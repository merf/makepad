//! Cutie's network, as `cutie-base-mega` runs it (the `merf/Cutie` fork, `cutie/model/*.py`): the
//! pieces every part shares (the checkpoint, convolutions with BatchNorm folded in, linears,
//! LayerNorm, the residual and fusion blocks, taps) and the assembled `CutieNet`. Each part is in
//! its own file and adds its methods to `CutieNet` there:
//!
//! - `encoder.rs`: the pixel encoder (ResNet-50 to stride 16), `encode_image`, the key
//!   projection, `transform_key`.
//! - `mask.rs`: the mask encoder (ResNet-18), the sensory deep update, the object
//!   summarizer, `encode_mask`.
//! - `transformer.rs`: pixel fusion (`pixel_fusion`) and the object transformer
//!   (`readout_query`).
//! - `decoder.rs`: the mask decoder, the sensory update, `segment`.
//!
//! The memory (working, long-term, sensory, object memory) is in `memory.rs` and the
//! per-frame step (Cutie's `InferenceCore`) in `processor.rs`.
//!
//! Every tensor is f32 with exact fp32 arithmetic: the port is checked against a PyTorch fp32
//! oracle with TF32 off (`oracle.rs`, `cutie-validate`).

use crate::torch_pth::PthStateDict;
use crate::{DiffusionError, Result};
use makepad_ai_common::gpu::{
    cutie_conv_out, gpu_add, gpu_birefnet_relu, gpu_concat_rows_many, gpu_cutie_conv2d, gpu_cutie_eca,
    gpu_download, gpu_layer_norm_pytorch, gpu_linear_f32_resident, gpu_slice_rows, gpu_upload, GpuTensor,
};
use std::path::Path;

/// The device weight cache's namespace for Cutie (`unload_cutie` evicts it).
pub(crate) const CUTIE_NAMESPACE: &str = "cutie";

/// BatchNorm's epsilon in Cutie's ResNets (torchvision's default).
pub(crate) const BN_EPS: f64 = 1e-5;

/// A device op's error, named by stage.
pub(crate) fn dev(stage: &str, error: String) -> DiffusionError {
    DiffusionError::model(format!("cutie {stage}: {error}"))
}

/// A part not ported yet (the skeleton's placeholder).
pub(crate) fn not_ported(what: &str) -> DiffusionError {
    DiffusionError::model(format!("cutie: {what} is not ported yet"))
}

// --- The checkpoint ---

/// `cutie-base-mega.pth`, read tensor by tensor by its state-dict name. Every read checks the
/// shape. The `num_batches_tracked` int64 entries are never read.
pub(crate) struct Weights {
    dict: PthStateDict,
    /// Prefixes every device cache key, so two checkpoints never share a cached weight.
    pub(crate) key_prefix: String,
}

impl Weights {
    pub(crate) fn load(path: &Path) -> Result<Self> {
        let dict = PthStateDict::load(path).map_err(|e| DiffusionError::model(format!("{}: {e}", path.display())))?;
        Ok(Self { dict, key_prefix: format!("{}:", path.display()) })
    }

    pub(crate) fn has(&self, name: &str) -> bool {
        self.dict.has(name)
    }

    pub(crate) fn shape(&self, name: &str) -> Result<Vec<usize>> {
        self.dict.shape(name).map_err(|e| DiffusionError::model(format!("cutie weight {name}: {e}")))
    }

    pub(crate) fn f32(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        self.dict.f32_shaped(name, shape).map_err(|e| DiffusionError::model(format!("cutie weight {name}: {e}")))
    }

    /// A Conv2d by its module name (`{name}.weight`, `{name}.bias` when `bias`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn conv(&mut self, name: &str, in_c: usize, out_c: usize, k: usize, pad: usize, stride: usize, bias: bool) -> Result<Conv2d> {
        let weights = self.f32(&format!("{name}.weight"), &[out_c, in_c, k, k])?;
        let bias = if bias { Some(self.f32(&format!("{name}.bias"), &[out_c])?) } else { None };
        Ok(Conv2d { key: format!("{}{name}", self.key_prefix), weights, bias, in_c, out_c, k, pad, stride })
    }

    /// A bias-free Conv2d followed by an eval-mode BatchNorm2d, the BatchNorm folded into the
    /// convolution (in f64): w' = w * g / sqrt(v + eps), b' = beta - m * g / sqrt(v + eps).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn conv_bn(&mut self, conv: &str, bn: &str, in_c: usize, out_c: usize, k: usize, pad: usize, stride: usize) -> Result<Conv2d> {
        let mut c = self.conv(conv, in_c, out_c, k, pad, stride, false)?;
        let gamma = self.f32(&format!("{bn}.weight"), &[out_c])?;
        let beta = self.f32(&format!("{bn}.bias"), &[out_c])?;
        let mean = self.f32(&format!("{bn}.running_mean"), &[out_c])?;
        let var = self.f32(&format!("{bn}.running_var"), &[out_c])?;
        let per = in_c * k * k;
        let mut bias = vec![0.0f32; out_c];
        for o in 0..out_c {
            let scale = gamma[o] as f64 / (var[o] as f64 + BN_EPS).sqrt();
            for w in &mut c.weights[o * per..(o + 1) * per] {
                *w = (*w as f64 * scale) as f32;
            }
            bias[o] = (beta[o] as f64 - mean[o] as f64 * scale) as f32;
        }
        c.bias = Some(bias);
        c.key = format!("{}{conv}+{bn}", self.key_prefix);
        Ok(c)
    }

    /// An nn.Linear (`{name}.weight` [out, in], `{name}.bias`), resident on the device.
    pub(crate) fn linear(&mut self, name: &str, in_f: usize, out_f: usize, bias: bool) -> Result<Linear> {
        let w = self.f32(&format!("{name}.weight"), &[out_f, in_f])?;
        let b = if bias { Some(self.f32(&format!("{name}.bias"), &[out_f])?) } else { None };
        Linear::new(&w, b.as_deref(), in_f, out_f)
    }

    /// An nn.LayerNorm over `dim` (`{name}.weight`, `{name}.bias`, eps 1e-5).
    pub(crate) fn layer_norm(&mut self, name: &str, dim: usize) -> Result<LayerNorm> {
        Ok(LayerNorm {
            gamma: self.f32(&format!("{name}.weight"), &[dim])?,
            beta: self.f32(&format!("{name}.bias"), &[dim])?,
            eps: 1e-5,
        })
    }
}

// --- Tensors ---

/// A batch of planar images: `t` is [n*c, h*w], each image's c channels contiguous rows, which
/// is the memory order of PyTorch's contiguous (n, c, h, w), so of Cutie's (batch=1, objects, c,
/// h, w) with n the objects.
pub(crate) struct Feat {
    pub(crate) t: GpuTensor,
    pub(crate) n: usize,
    pub(crate) c: usize,
    pub(crate) w: usize,
    pub(crate) h: usize,
}

impl Feat {
    pub(crate) fn new(t: GpuTensor, n: usize, c: usize, w: usize, h: usize) -> Result<Self> {
        if t.rows() != n * c || t.cols() != w * h {
            return Err(DiffusionError::model(format!(
                "cutie feat: {}x{} is not {n}x{c} at {w}x{h}",
                t.rows(),
                t.cols()
            )));
        }
        Ok(Self { t, n, c, w, h })
    }

    pub(crate) fn upload(values: &[f32], n: usize, c: usize, w: usize, h: usize) -> Result<Self> {
        Self::new(gpu_upload(values, n * c, w * h).map_err(|e| dev("upload", e))?, n, c, w, h)
    }

    pub(crate) fn download(&self) -> Result<Vec<f32>> {
        gpu_download(&self.t).map_err(|e| dev("download", e))
    }

    pub(crate) fn plane(&self) -> usize {
        self.w * self.h
    }

    /// A view of the same tensor (no copy).
    pub(crate) fn view(&self) -> Result<Self> {
        Ok(Self { t: gpu_slice_rows(&self.t, 0, self.t.rows()).map_err(|e| dev("view", e))?, ..*self })
    }

    /// A view of image `i` (no copy).
    pub(crate) fn image(&self, i: usize) -> Result<Self> {
        Ok(Self { t: gpu_slice_rows(&self.t, i * self.c, self.c).map_err(|e| dev("image", e))?, n: 1, ..*self })
    }

    /// Channels `start..start+len` of every image (a copy when n > 1).
    pub(crate) fn channels(&self, start: usize, len: usize) -> Result<Self> {
        let views = (0..self.n)
            .map(|i| gpu_slice_rows(&self.t, i * self.c + start, len))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| dev("channels", e))?;
        let parts: Vec<&GpuTensor> = views.iter().collect();
        Self::new(gpu_concat_rows_many(&parts).map_err(|e| dev("channels", e))?, self.n, len, self.w, self.h)
    }

    /// The images of `parts` one after another (same c, w, h).
    pub(crate) fn cat_images(parts: &[&Feat]) -> Result<Self> {
        let t: Vec<&GpuTensor> = parts.iter().map(|f| &f.t).collect();
        let n = parts.iter().map(|f| f.n).sum();
        Self::new(gpu_concat_rows_many(&t).map_err(|e| dev("cat images", e))?, n, parts[0].c, parts[0].w, parts[0].h)
    }

    /// Each image's channels followed by the same image's channels in `other` (same n, w, h):
    /// PyTorch's cat along the channel dimension.
    pub(crate) fn cat_channels(&self, other: &Feat) -> Result<Self> {
        let mut views = Vec::with_capacity(self.n * 2);
        for i in 0..self.n {
            views.push(gpu_slice_rows(&self.t, i * self.c, self.c).map_err(|e| dev("cat channels", e))?);
            views.push(gpu_slice_rows(&other.t, i * other.c, other.c).map_err(|e| dev("cat channels", e))?);
        }
        let parts: Vec<&GpuTensor> = views.iter().collect();
        Self::new(gpu_concat_rows_many(&parts).map_err(|e| dev("cat channels", e))?, self.n, self.c + other.c, self.w, self.h)
    }

    /// A single image repeated n times (Cutie's expand of shared pixel features to each object).
    pub(crate) fn repeat(&self, n: usize) -> Result<Self> {
        if self.n != 1 {
            return Err(DiffusionError::model("cutie feat repeat: not a single image"));
        }
        let parts: Vec<&Feat> = std::iter::repeat(self).take(n).collect();
        Self::cat_images(&parts)
    }
}

pub(crate) fn relu(x: &Feat) -> Result<Feat> {
    Feat::new(gpu_birefnet_relu(&x.t).map_err(|e| dev("relu", e))?, x.n, x.c, x.w, x.h)
}

pub(crate) fn add(a: &Feat, b: &Feat) -> Result<Feat> {
    Feat::new(gpu_add(&a.t, &b.t).map_err(|e| dev("add", e))?, a.n, a.c, a.w, a.h)
}

// --- Layers ---

/// A square Conv2d with symmetric padding and stride, applied to every image of a batch
/// (Cutie's GConv2d is the same over the objects).
pub(crate) struct Conv2d {
    pub(crate) key: String,
    pub(crate) weights: Vec<f32>,
    pub(crate) bias: Option<Vec<f32>>,
    pub(crate) in_c: usize,
    pub(crate) out_c: usize,
    pub(crate) k: usize,
    pub(crate) pad: usize,
    pub(crate) stride: usize,
}

impl Conv2d {
    pub(crate) fn forward(&self, x: &Feat) -> Result<Feat> {
        if x.c != self.in_c {
            return Err(DiffusionError::model(format!("cutie conv {}: {} channels in, wants {}", self.key, x.c, self.in_c)));
        }
        let t = gpu_cutie_conv2d(
            &x.t, x.n, x.w, x.h, CUTIE_NAMESPACE, &self.key, &self.weights, self.bias.as_deref(), self.out_c, self.k, self.pad, self.stride,
        )
        .map_err(|e| dev(&self.key, e))?;
        let (w, h) = (cutie_conv_out(x.w, self.k, self.pad, self.stride), cutie_conv_out(x.h, self.k, self.pad, self.stride));
        Feat::new(t, x.n, self.out_c, w, h)
    }
}

/// An nn.Linear on rows: x [rows, in] -> [rows, out] = x @ W^T + b, fp32 (cuBLAS SGEMM, no TF32).
pub(crate) struct Linear {
    pub(crate) weight: GpuTensor,
    pub(crate) bias: Option<GpuTensor>,
    pub(crate) in_f: usize,
    pub(crate) out_f: usize,
}

impl Linear {
    pub(crate) fn new(weight: &[f32], bias: Option<&[f32]>, in_f: usize, out_f: usize) -> Result<Self> {
        Ok(Self {
            weight: gpu_upload(weight, out_f, in_f).map_err(|e| dev("linear weight", e))?,
            bias: bias.map(|b| gpu_upload(b, 1, out_f)).transpose().map_err(|e| dev("linear bias", e))?,
            in_f,
            out_f,
        })
    }

    pub(crate) fn forward(&self, x: &GpuTensor) -> Result<GpuTensor> {
        gpu_linear_f32_resident(x, &self.weight, self.bias.as_ref()).map_err(|e| dev("linear", e))
    }
}

/// nn.LayerNorm over the last dimension of rows [rows, dim] (PyTorch's CUDA kernel's order).
pub(crate) struct LayerNorm {
    pub(crate) gamma: Vec<f32>,
    pub(crate) beta: Vec<f32>,
    pub(crate) eps: f32,
}

impl LayerNorm {
    pub(crate) fn forward(&self, x: &GpuTensor) -> Result<GpuTensor> {
        gpu_layer_norm_pytorch(x, &self.gamma, &self.beta, self.eps).map_err(|e| dev("layer norm", e))
    }
}

/// Cutie's channel-attention residual block (`channel_attn.CAResBlock`, residual, in == out):
/// conv2(relu(conv1(relu(x)))), scaled per channel by ECA (mean, 1-D conv over channels,
/// sigmoid), plus x.
pub(crate) struct CAResBlock {
    pub(crate) conv1: Conv2d,
    pub(crate) conv2: Conv2d,
    pub(crate) eca: Vec<f32>,
}

impl CAResBlock {
    pub(crate) fn load(w: &mut Weights, name: &str, dim: usize) -> Result<Self> {
        let k = w.shape(&format!("{name}.conv.weight"))?.last().copied().unwrap_or(0);
        Ok(Self {
            conv1: w.conv(&format!("{name}.conv1"), dim, dim, 3, 1, 1, true)?,
            conv2: w.conv(&format!("{name}.conv2"), dim, dim, 3, 1, 1, true)?,
            eca: w.f32(&format!("{name}.conv.weight"), &[1, 1, k])?,
        })
    }

    pub(crate) fn forward(&self, x: &Feat) -> Result<Feat> {
        let h = self.conv1.forward(&relu(x)?)?;
        let h = self.conv2.forward(&relu(&h)?)?;
        Feat::new(gpu_cutie_eca(&h.t, &x.t, h.n, h.c, &self.eca).map_err(|e| dev("eca", e))?, h.n, h.c, h.w, h.h)
    }
}

/// Cutie's GroupResBlock (the decoder's): conv2(relu(conv1(relu(g)))) + downsample(g), the
/// downsample a 1x1 conv when the width changes.
pub(crate) struct GroupResBlock {
    pub(crate) conv1: Conv2d,
    pub(crate) conv2: Conv2d,
    pub(crate) downsample: Option<Conv2d>,
}

impl GroupResBlock {
    pub(crate) fn load(w: &mut Weights, name: &str, in_c: usize, out_c: usize) -> Result<Self> {
        Ok(Self {
            conv1: w.conv(&format!("{name}.conv1"), in_c, out_c, 3, 1, 1, true)?,
            conv2: w.conv(&format!("{name}.conv2"), out_c, out_c, 3, 1, 1, true)?,
            downsample: if in_c != out_c { Some(w.conv(&format!("{name}.downsample"), in_c, out_c, 1, 0, 1, true)?) } else { None },
        })
    }

    pub(crate) fn forward(&self, g: &Feat) -> Result<Feat> {
        let out = self.conv1.forward(&relu(g)?)?;
        let out = self.conv2.forward(&relu(&out)?)?;
        match &self.downsample {
            Some(d) => add(&out, &d.forward(g)?),
            None => add(&out, g),
        }
    }
}

/// Cutie's GroupFeatureFusionBlock: the shared image features `x` (one image) through a 1x1
/// conv, the objects' features `g` (n images) through another, added per object (x + g), then
/// two channel-attention blocks.
pub(crate) struct FusionBlock {
    pub(crate) x_transform: Conv2d,
    pub(crate) g_transform: Conv2d,
    pub(crate) block1: CAResBlock,
    pub(crate) block2: CAResBlock,
}

impl FusionBlock {
    pub(crate) fn load(w: &mut Weights, name: &str, x_in: usize, g_in: usize, out: usize) -> Result<Self> {
        Ok(Self {
            x_transform: w.conv(&format!("{name}.distributor.x_transform"), x_in, out, 1, 0, 1, true)?,
            g_transform: w.conv(&format!("{name}.distributor.g_transform"), g_in, out, 1, 0, 1, true)?,
            block1: CAResBlock::load(w, &format!("{name}.block1"), out)?,
            block2: CAResBlock::load(w, &format!("{name}.block2"), out)?,
        })
    }

    /// Taps `{name}.distributor`, `{name}.block1`, `{name}.block2` (the oracle's module names).
    pub(crate) fn forward(&self, x: &Feat, g: &Feat, name: &str, tap: &mut Tap) -> Result<Feat> {
        let x = self.x_transform.forward(x)?.repeat(g.n)?;
        let g = add(&x, &self.g_transform.forward(g)?)?;
        tap.put(&format!("{name}.distributor"), &g.t)?;
        let g = self.block1.forward(&g)?;
        tap.put(&format!("{name}.block1"), &g.t)?;
        let g = self.block2.forward(&g)?;
        tap.put(&format!("{name}.block2"), &g.t)?;
        Ok(g)
    }
}

// --- Taps ---

/// Where a forward pass shows its intermediate tensors, under the oracle's names
/// (tools/cutie-oracle in RationalVisions: `taps/<step>/manifest.tsv`) and in PyTorch's memory
/// order (transpose token-major tensors before putting them when the oracle's are channel-major,
/// and the other way round). Off unless a sink is given; then each put downloads.
pub(crate) struct Tap<'a> {
    sink: Option<&'a mut dyn FnMut(&str, &GpuTensor) -> Result<()>>,
}

impl<'a> Tap<'a> {
    pub(crate) fn none() -> Self {
        Self { sink: None }
    }

    pub(crate) fn new(sink: &'a mut dyn FnMut(&str, &GpuTensor) -> Result<()>) -> Self {
        Self { sink: Some(sink) }
    }

    pub(crate) fn active(&self) -> bool {
        self.sink.is_some()
    }

    pub(crate) fn put(&mut self, name: &str, t: &GpuTensor) -> Result<()> {
        match &mut self.sink {
            Some(sink) => sink(name, t),
            None => Ok(()),
        }
    }
}

// --- The network ---

use crate::decoder::MaskDecoder;
use crate::encoder::{KeyProjection, PixelEncoder};
use crate::mask::{MaskEncoder, ObjectSummarizer};
use crate::transformer::{PixelFuser, QueryTransformer};

/// What `encode_image` gives a frame: the pixel encoder's three scales and the projected pixel
/// features, all one image.
pub(crate) struct ImageFeatures {
    /// The frame normalised (ImageNet mean and std), padded: [3, H*W].
    pub(crate) image: Feat,
    pub(crate) f16: Feat,
    pub(crate) f8: Feat,
    pub(crate) f4: Feat,
    pub(crate) pix_feat: Feat,
}

/// The key projection's outputs for a frame (one image at stride 16).
pub(crate) struct Keys {
    pub(crate) key: Feat,
    pub(crate) shrinkage: Feat,
    pub(crate) selection: Feat,
}

/// Cutie's network (`CUTIE`), every part loaded.
pub(crate) struct CutieNet {
    pub(crate) pixel_encoder: PixelEncoder,
    pub(crate) key_proj: KeyProjection,
    pub(crate) mask_encoder: MaskEncoder,
    pub(crate) object_summarizer: ObjectSummarizer,
    pub(crate) pixel_fuser: PixelFuser,
    pub(crate) object_transformer: QueryTransformer,
    pub(crate) mask_decoder: MaskDecoder,
}

impl CutieNet {
    pub(crate) fn load(w: &mut Weights) -> Result<Self> {
        Ok(Self {
            pixel_encoder: PixelEncoder::load(w)?,
            key_proj: KeyProjection::load(w)?,
            mask_encoder: MaskEncoder::load(w)?,
            object_summarizer: ObjectSummarizer::load(w)?,
            pixel_fuser: PixelFuser::load(w)?,
            object_transformer: QueryTransformer::load(w)?,
            mask_decoder: MaskDecoder::load(w)?,
        })
    }
}
