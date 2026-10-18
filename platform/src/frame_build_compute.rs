//! Compute dispatches in the frame (`compute.rs`): a pass's queue goes to the HAL as a frame
//! packet of its own, a compute pass, submitted just before the pass's own packet, so the pass
//! and the passes that read it after (its parents) see what the kernels wrote. Names are
//! resolved here against the kernel as compiled for the running backend; D3D11's rules (never a
//! resource read and written by one dispatch) are checked here, with the kernel named.

use crate::{
    compute::{ComputeDispatch, DispatchSize},
    cx::Cx,
    draw_pass::DrawPassId,
    frame_build::FrameBuilder,
    makepad_script::shader_ir::IrScalar,
    texture::TextureFormat,
};
use makepad_hal::{pass_load as load, Dispatch, FrameMeta, Id, Pass, Slice};

/// The most workgroups a dispatch may run along one axis (D3D11's limit).
const MOST_GROUPS: u32 = 65535;

impl Cx {
    /// Sends pass `pass_id`'s waiting dispatches as a compute packet, in the order recorded, and
    /// empties its queue; or keeps the whole queue while one of their kernels still compiles
    /// (its PipelineReady paints the pass again), so a sequence never half-runs.
    pub(crate) fn fb_compute(&mut self, fb: &mut FrameBuilder, pass_id: DrawPassId) {
        if self.passes[pass_id].dispatches.is_empty() {
            return;
        }
        for d in &self.passes[pass_id].dispatches {
            let pipe = d.kernel.and_then(|k| self.compute_kernels.get(k)).and_then(|k| k.pipeline);
            if pipe.is_some_and(|p| fb.pipeline_waited_for(p)) {
                fb.stats.held_for_kernels += 1;
                return;
            }
        }
        let queue = std::mem::take(&mut self.passes[pass_id].dispatches);
        let mut dispatches = Vec::new();
        let mut clears: Vec<(Id, u32, u32)> = Vec::new();
        for d in &queue {
            match self.fb_resolve_dispatch(fb, d, &mut clears) {
                Ok(Some(dispatch)) => dispatches.push(dispatch),
                Ok(None) => {}
                Err(reason) => self.report_dispatch(d, &reason),
            }
        }
        if dispatches.is_empty() && clears.is_empty() {
            return;
        }
        let serial = self.textures.1.serials.submit();
        // `compute <pass>`: frame captures label it so, and the GPU frame timer reports it so
        let name = match self.passes[pass_id].debug_name.as_str() {
            "" => "compute".to_string(),
            pass => format!("compute {pass}"),
        };
        let interned = fb.intern(&name);
        fb.w.frame_begin(serial, FrameMeta::default());
        // a storage image just made holds nothing defined until cleared
        for (image, w, h) in clears {
            let mut pass = Pass { color: [image, Id::NONE, Id::NONE, Id::NONE], depth: Id::NONE, ..Default::default() };
            pass.load = [load::CLEAR, load::DONT_CARE, load::DONT_CARE, load::DONT_CARE, load::DONT_CARE];
            pass.viewport = [0.0, 0.0, w as f32, h as f32];
            pass.dpi = 1.0;
            pass.name = interned;
            fb.w.pass(pass);
        }
        if !dispatches.is_empty() {
            fb.w.pass(Pass::compute(interned));
            for dispatch in dispatches {
                fb.w.dispatch(dispatch);
                fb.stats.dispatches += 1;
            }
        }
        // no target: every pass in it is a compute pass or a clear of its images
        fb.w.frame_end(Id::NONE, makepad_hal::target::TEXTURE);
        fb.stats.frames += 1;
        fb.stats.compute_packets += 1;
        self.fb_time_packet(fb, pass_id, serial, name, false);
    }

    /// Dispatch `d` as the HAL takes it: its pipeline, params at their offsets, and its textures,
    /// images and buffers in the order the kernel binds them; `None` when it runs nothing (a zero
    /// size); an error naming what is wrong.
    fn fb_resolve_dispatch(&mut self, fb: &mut FrameBuilder, d: &ComputeDispatch, clears: &mut Vec<(Id, u32, u32)>) -> Result<Option<Dispatch>, String> {
        let kernel = d.kernel.and_then(|k| self.compute_kernels.get(k)).ok_or("its kernel did not compile")?;
        if let Some(error) = &kernel.error {
            return Err(error.clone());
        }
        let pipe = kernel.pipeline.ok_or("its kernel has no pipeline")?;
        let b = kernel.code.as_ref().and_then(|c| c.bindings()).ok_or("its kernel's backend binds by no names")?.clone();
        let owner = fb.intern_live(kernel.debug_id);
        // params, at the offsets the kernel was compiled with
        let mut params = vec![0u8; b.params_size as usize];
        for (name, value) in &d.params {
            let slot = b.param(*name).ok_or_else(|| format!("the kernel has no param {name}"))?;
            let crate::makepad_script::shader_ir_compute::ComputeSlotKind::Param { scalar, lanes } = slot.kind else {
                return Err(format!("{name} is not a param of the kernel"));
            };
            if (scalar, lanes) != value.kind() {
                return Err(format!("param {name} is {lanes} x {scalar:?} in the kernel, given {value:?}"));
            }
            let bytes = value.bytes();
            let at = slot.offset as usize;
            params[at..at + bytes.len()].copy_from_slice(&bytes);
        }
        // sampled textures, written images and buffers, in the kernel's order
        let mut tex = Vec::new();
        for (name, _) in &b.textures {
            let (_, t) = d.textures.iter().find(|(n, _)| n == name).ok_or_else(|| format!("no texture given for {name}"))?;
            tex.push((t.texture_id(), self.fb_sampled_for_kernel(fb, t)));
        }
        let mut images = Vec::new();
        for image in &b.images {
            let (_, t) = d.images.iter().find(|(n, _)| *n == image.name).ok_or_else(|| format!("no image given for {}", image.name))?;
            let tid = t.texture_id();
            if !self.textures[tid].storage {
                return Err(format!("image {} is not a storage texture (Texture::new_storage)", image.name));
            }
            if !texel_fits(&self.textures[tid].format, image.scalar, image.lanes) {
                return Err(format!("image {} is written as {} x {:?}, not what its texture holds", image.name, image.lanes, image.scalar));
            }
            if tex.iter().any(|(t, _)| *t == tid) {
                return Err(format!("image {} is sampled by the same dispatch (D3D11 never binds a texture to read and write at once)", image.name));
            }
            let (id, fresh) = self.fb_storage_image(fb, tid);
            if id == Id::NONE {
                return Err(format!("image {} could not be made", image.name));
            }
            if fresh {
                self.textures[tid].take_initial();
                if let Some(alloc) = self.textures[tid].alloc.as_ref() {
                    clears.push((id, alloc.width as u32, alloc.height as u32));
                }
            }
            images.push(id);
        }
        let mut buffers = Vec::new();
        let mut bound: Vec<(crate::storage_buffer::StorageBufferId, bool)> = Vec::new();
        for kb in &b.buffers {
            let (_, sb) = d.buffers.iter().find(|(n, _)| *n == kb.name).ok_or_else(|| format!("no buffer given for {}", kb.name))?;
            let sid = sb.storage_buffer_id();
            let stride = self.storage_buffers[sid].stride as u32;
            if stride != kb.stride {
                return Err(format!("buffer {} has {stride}-byte elements, the kernel's are {} bytes", kb.name, kb.stride));
            }
            if bound.iter().any(|(other, writable)| *other == sid && *writable != kb.writable) {
                return Err(format!("buffer {} is bound both read-only and written (D3D11 would read nothing)", kb.name));
            }
            bound.push((sid, kb.writable));
            buffers.push(self.fb_storage_buffer(fb, sid));
        }
        let groups = match d.size {
            DispatchSize::Groups(g) => g,
            DispatchSize::Threads(t) => std::array::from_fn(|i| t[i].div_ceil(b.workgroup[i].max(1))),
        };
        if groups.iter().any(|&g| g > MOST_GROUPS) {
            return Err(format!("{groups:?} workgroups: at most {MOST_GROUPS} a side"));
        }
        if groups.contains(&0) {
            return Ok(None);
        }
        fb.staging.push(params);
        let uniforms = Slice::from_bytes(fb.staging.last().expect("pushed"));
        let tex_ids: Vec<Id> = tex.iter().map(|(_, id)| *id).collect();
        Ok(Some(Dispatch {
            pipeline: fb.pipeline_id(pipe),
            groups,
            uniforms,
            tex: fb.w.put_ids(&tex_ids),
            images: fb.w.put_ids(&images),
            buffers: fb.w.put_ids(&buffers),
            owner,
        }))
    }

    /// A texture a kernel samples: as a draw samples it.
    fn fb_sampled_for_kernel(&mut self, fb: &mut FrameBuilder, t: &crate::texture::Texture) -> Id {
        self.fb_sampled_pub(fb, t.texture_id())
    }

    /// Logs once why dispatches of a kernel do not run.
    fn report_dispatch(&mut self, d: &ComputeDispatch, reason: &str) {
        let name = d.kernel.and_then(|k| self.compute_kernels.get(k)).map_or_else(|| "?".to_string(), |k| k.debug_id.to_string());
        let message = format!("a dispatch of compute kernel '{name}' did not run: {reason}");
        if self.compute_kernels.reported.insert(message.clone()) {
            crate::error!("{}", message);
        }
    }
}

/// Whether a texture of `format` holds what a kernel writes, `lanes` of `scalar`.
fn texel_fits(format: &TextureFormat, scalar: IrScalar, lanes: u32) -> bool {
    let float = matches!(scalar, IrScalar::F32 | IrScalar::F16);
    match format {
        TextureFormat::RenderRGBAf16 { .. } | TextureFormat::RenderRGBAf32 { .. } | TextureFormat::RenderBGRAu8 { .. } => float && lanes == 4,
        TextureFormat::RenderRf32 { .. } => float && lanes == 1,
        TextureFormat::RenderRGf16 { .. } => float && lanes == 2,
        TextureFormat::RenderRu32 { .. } => scalar == IrScalar::U32 && lanes == 1,
        _ => false,
    }
}
