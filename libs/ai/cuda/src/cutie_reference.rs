// Cutie's host references (models/cutie): each `gpu_cutie_X_reference` downloads, computes on the
// host in PyTorch's order and uploads, through the public gpu_* surface only, so it builds against
// any backend's GpuTensor. Included by cutie_ops.rs inside the CUDA `mod imp` (where `cutie-validate
// ops` checks each kernel against its reference) and by models/common/src/gpu_cutie_host.rs on
// builds without Cutie's CUDA kernels: there the references are the ops on Linux and Windows, and
// on macOS the path MAKEPAD_CUTIE_METAL_REF forces beside Metal's kernels.

fn cutie_check_f32(op: &str, x: &GpuTensor) -> Result<(), String> {
    if x.is_half() {
        return Err(format!("{op} is f32-only"));
    }
    Ok(())
}

/// PyTorch's order for a sum or product over a dimension that is not the innermost (Cutie's
/// objects, the top-k picks), as the kernels' `cutie_reduce4`: four accumulators from `ident`,
/// term o into o % 4, then ((a0 op a1) op a2) op a3 (Reduce.cuh's thread_reduce_impl with
/// vt0 = 4, one thread an output, which holds for fewer than 64 terms). From 5 terms on it rounds
/// differently from a sequential sum, and PyTorch's results are this order's to the bit.
fn cutie_reduce4(terms: impl IntoIterator<Item = f32>, ident: f32, op: impl Fn(f32, f32) -> f32) -> f32 {
    let mut acc = [ident; 4];
    for (o, t) in terms.into_iter().enumerate() {
        acc[o % 4] = op(acc[o % 4], t);
    }
    op(op(op(acc[0], acc[1]), acc[2]), acc[3])
}

/// 1 / sqrt(ck) as PyTorch applies `x / math.sqrt(ck)` on CUDA (div_true with a CPU scalar): the
/// reciprocal of the Python float taken in f64, rounded to f32, then a multiply (torch 2.11 gives
/// exactly this for every ck tried; dividing by sqrt(ck) in f32 differs at every one but 64).
fn cutie_inv_sqrt(ck: usize) -> f32 {
    (1.0 / (ck as f64).sqrt()) as f32
}

/// The larger of two values, a NaN in either winning (PyTorch's `max_propagate_nan`).
fn cutie_max_nan(a: f32, b: f32) -> f32 {
    if b.is_nan() {
        b
    } else if a.is_nan() {
        a
    } else {
        a.max(b)
    }
}

/// PyTorch's radix key of a float (`TopKTypeConfig<float>::convert`), as the top-k kernel's: the
/// keys' unsigned order is torch.topk's order of the values, so -0 ranks below +0 and every NaN
/// (either sign) above every number.
fn cutie_topk_key(v: f32) -> u32 {
    if v.is_nan() {
        return u32::MAX;
    }
    let x = v.to_bits();
    x ^ if x & 0x8000_0000 != 0 { u32::MAX } else { 0x8000_0000 }
}

/// A convolution's output size along one axis (PyTorch's floor rule).
pub fn cutie_conv_out(size: usize, k: usize, pad: usize, stride: usize) -> usize {
    (size + 2 * pad - k) / stride + 1
}

/// Reference: the existing direct fp32 kernel (`gpu_conv2d_planar_strided`) once per image.
pub fn gpu_cutie_conv2d_reference(
    x: &GpuTensor, batch: usize, in_w: usize, in_h: usize, cache_namespace: &str, key: &str,
    weights: &[f32], bias: Option<&[f32]>, out_channels: usize, k: usize, pad: usize, stride: usize,
) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_conv2d", x)?;
    if batch == 0 || x.rows() % batch != 0 || x.cols() != in_w * in_h {
        return Err(format!("gpu_cutie_conv2d {key}: x {}x{} for batch {batch} at {in_w}x{in_h}", x.rows(), x.cols()));
    }
    let in_channels = x.rows() / batch;
    if weights.len() != out_channels * in_channels * k * k {
        return Err(format!("gpu_cutie_conv2d {key}: {} weights for {out_channels}x{in_channels}x{k}x{k}", weights.len()));
    }
    let zeros;
    let bias = match bias {
        Some(b) => b,
        None => {
            zeros = vec![0.0f32; out_channels];
            &zeros
        }
    };
    let (ow, oh) = (cutie_conv_out(in_w, k, pad, stride), cutie_conv_out(in_h, k, pad, stride));
    let mut outs = Vec::with_capacity(batch);
    for b in 0..batch {
        let view = gpu_slice_rows(x, b * in_channels, in_channels)?;
        outs.push(gpu_conv2d_planar_strided(
            &view, in_w, in_h, ow, oh, cache_namespace, key, weights, bias, out_channels, k, k, pad, pad, stride, stride,
        )?);
    }
    if batch == 1 {
        return Ok(outs.pop().unwrap());
    }
    let parts: Vec<&GpuTensor> = outs.iter().collect();
    gpu_concat_rows_many(&parts)
}

/// Reference: `gpu_cutie_conv2d_reference`, then `gpu_add` with the residual and
/// `gpu_birefnet_relu`, the separate ops the fused epilogue replaces.
#[allow(clippy::too_many_arguments)]
pub fn gpu_cutie_conv2d_fused_reference(
    x: &GpuTensor, batch: usize, in_w: usize, in_h: usize, cache_namespace: &str, key: &str,
    weights: &[f32], bias: Option<&[f32]>, out_channels: usize, k: usize, pad: usize, stride: usize,
    residual: Option<&GpuTensor>, relu: bool,
) -> Result<GpuTensor, String> {
    let mut out = gpu_cutie_conv2d_reference(x, batch, in_w, in_h, cache_namespace, key, weights, bias, out_channels, k, pad, stride)?;
    if let Some(r) = residual {
        out = gpu_add(&out, r)?;
    }
    if relu {
        out = gpu_birefnet_relu(&out)?;
    }
    Ok(out)
}

pub fn gpu_cutie_maxpool3x3s2_reference(x: &GpuTensor, w: usize, h: usize) -> Result<GpuTensor, String> {
    let v = gpu_download(x)?;
    let (ow, oh) = (cutie_conv_out(w, 3, 1, 2), cutie_conv_out(h, 3, 1, 2));
    let mut out = vec![0.0f32; x.rows() * ow * oh];
    for p in 0..x.rows() {
        for oy in 0..oh {
            for ox in 0..ow {
                let mut best = f32::NEG_INFINITY;
                for ky in 0..3 {
                    let y = (oy * 2 + ky) as isize - 1;
                    if y < 0 || y >= h as isize {
                        continue;
                    }
                    for kx in 0..3 {
                        let xx = (ox * 2 + kx) as isize - 1;
                        if xx < 0 || xx >= w as isize {
                            continue;
                        }
                        let tap = v[p * w * h + y as usize * w + xx as usize];
                        if tap > best || tap.is_nan() {
                            best = tap;
                        }
                    }
                }
                out[p * ow * oh + oy * ow + ox] = best;
            }
        }
    }
    gpu_upload(&out, x.rows(), ow * oh)
}

pub fn gpu_cutie_sigmoid_reference(x: &GpuTensor) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_sigmoid", x)?;
    let v: Vec<f32> = gpu_download(x)?.into_iter().map(|z| 1.0 / (1.0 + (-z).exp())).collect();
    gpu_upload(&v, x.rows(), x.cols())
}

pub fn gpu_cutie_gru_reference(h: &GpuTensor, values: &GpuTensor, n: usize, c: usize) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_gru", h)?;
    if h.rows() != n * c || values.rows() != n * 3 * c || h.cols() != values.cols() {
        return Err("gpu_cutie_gru shape mismatch".into());
    }
    let p = h.cols();
    let hv = gpu_download(h)?;
    let vv = gpu_download(values)?;
    let mut out = vec![0.0f32; n * c * p];
    for o in 0..n {
        for ch in 0..c {
            for i in 0..p {
                let f = 1.0 / (1.0 + (-vv[((o * 3 * c) + ch) * p + i]).exp());
                let u = 1.0 / (1.0 + (-vv[((o * 3 * c) + c + ch) * p + i]).exp());
                let nv = vv[((o * 3 * c) + 2 * c + ch) * p + i].tanh();
                let hh = hv[(o * c + ch) * p + i];
                out[(o * c + ch) * p + i] = f * hh * (1.0 - u) + u * nv;
            }
        }
    }
    gpu_upload(&out, n * c, p)
}

pub fn gpu_cutie_eca_reference(x: &GpuTensor, residual: &GpuTensor, n: usize, c: usize, conv_w: &[f32]) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_eca", x)?;
    if x.rows() != n * c || residual.rows() != x.rows() || residual.cols() != x.cols() || conv_w.len() % 2 == 0 {
        return Err("gpu_cutie_eca shape mismatch".into());
    }
    let p = x.cols();
    let xv = gpu_download(x)?;
    let rv = gpu_download(residual)?;
    let k = conv_w.len();
    let pad = (k - 1) / 2;
    let mut out = vec![0.0f32; n * c * p];
    for o in 0..n {
        let means: Vec<f32> = (0..c)
            .map(|ch| {
                let row = &xv[(o * c + ch) * p..(o * c + ch + 1) * p];
                (row.iter().map(|&v| v as f64).sum::<f64>() / p as f64) as f32
            })
            .collect();
        for ch in 0..c {
            let mut y = 0.0f32;
            for j in 0..k {
                let src = ch as isize + j as isize - pad as isize;
                if src >= 0 && (src as usize) < c {
                    y += conv_w[j] * means[src as usize];
                }
            }
            let s = 1.0 / (1.0 + (-y).exp());
            for i in 0..p {
                let at = (o * c + ch) * p + i;
                out[at] = xv[at] * s + rv[at];
            }
        }
    }
    gpu_upload(&out, n * c, p)
}

pub fn gpu_cutie_others_reference(masks: &GpuTensor) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_others", masks)?;
    let (n, p) = (masks.rows(), masks.cols());
    let v = gpu_download(masks)?;
    let mut out = vec![0.0f32; n * p];
    for i in 0..p {
        let sum = cutie_reduce4((0..n).map(|o| v[o * p + i]), 0.0, |a, b| a + b);
        for o in 0..n {
            out[o * p + i] = (sum - v[o * p + i]).clamp(0.0, 1.0);
        }
    }
    gpu_upload(&out, n, p)
}

pub fn gpu_cutie_pad_reference(x: &GpuTensor, w: usize, h: usize, left: usize, right: usize, top: usize, bottom: usize) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_pad", x)?;
    if x.cols() != w * h {
        return Err("gpu_cutie_pad shape mismatch".into());
    }
    let (nw, nh) = (w + left + right, h + top + bottom);
    let v = gpu_download(x)?;
    let mut out = vec![0.0f32; x.rows() * nw * nh];
    for p in 0..x.rows() {
        for y in 0..h {
            let src = &v[p * w * h + y * w..p * w * h + (y + 1) * w];
            let at = p * nw * nh + (y + top) * nw + left;
            out[at..at + w].copy_from_slice(src);
        }
    }
    gpu_upload(&out, x.rows(), nw * nh)
}

pub fn gpu_cutie_crop_reference(x: &GpuTensor, w: usize, h: usize, left: usize, top: usize, out_w: usize, out_h: usize) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_crop", x)?;
    if x.cols() != w * h || left + out_w > w || top + out_h > h {
        return Err("gpu_cutie_crop shape mismatch".into());
    }
    let v = gpu_download(x)?;
    let mut out = vec![0.0f32; x.rows() * out_w * out_h];
    for p in 0..x.rows() {
        for y in 0..out_h {
            let src = p * w * h + (y + top) * w + left;
            let at = p * out_w * out_h + y * out_w;
            out[at..at + out_w].copy_from_slice(&v[src..src + out_w]);
        }
    }
    gpu_upload(&out, x.rows(), out_w * out_h)
}

pub fn gpu_cutie_matmul_reference(a: &GpuTensor, trans_a: bool, b: &GpuTensor, trans_b: bool) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_matmul", a)?;
    cutie_check_f32("gpu_cutie_matmul", b)?;
    let (m, k) = if trans_a { (a.cols(), a.rows()) } else { (a.rows(), a.cols()) };
    let (k2, n) = if trans_b { (b.cols(), b.rows()) } else { (b.rows(), b.cols()) };
    if k != k2 {
        return Err(format!("gpu_cutie_matmul inner dims {k} != {k2}"));
    }
    let av = gpu_download(a)?;
    let bv = gpu_download(b)?;
    let at = |i: usize, j: usize| if trans_a { av[j * a.cols() + i] } else { av[i * a.cols() + j] };
    let bt = |i: usize, j: usize| if trans_b { bv[j * b.cols() + i] } else { bv[i * b.cols() + j] };
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f64;
            for l in 0..k {
                s += at(i, l) as f64 * bt(l, j) as f64;
            }
            out[i * n + j] = s as f32;
        }
    }
    gpu_upload(&out, m, n)
}

pub fn gpu_cutie_similarity_reference(mk: &GpuTensor, ms: Option<&GpuTensor>, qk: &GpuTensor, qe: Option<&GpuTensor>) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_similarity", mk)?;
    let (ck, m, p) = (mk.rows(), mk.cols(), qk.cols());
    if qk.rows() != ck || qe.is_some_and(|e| e.rows() != ck || e.cols() != p) || ms.is_some_and(|s| s.rows() * s.cols() != m) {
        return Err("gpu_cutie_similarity shape mismatch".into());
    }
    let mkv = gpu_download(mk)?;
    let qkv = gpu_download(qk)?;
    let qev = qe.map(gpu_download).transpose()?;
    let msv = ms.map(gpu_download).transpose()?;
    let inv_scale = cutie_inv_sqrt(ck);
    let mut out = vec![0.0f32; m * p];
    for j in 0..m {
        for i in 0..p {
            let sim = if let Some(qev) = &qev {
                let (mut a_sq, mut two_ab, mut b_sq) = (0.0f64, 0.0f64, 0.0f64);
                for c in 0..ck {
                    let (mkc, qkc, qec) = (mkv[c * m + j] as f64, qkv[c * p + i] as f64, qev[c * p + i] as f64);
                    a_sq += mkc * mkc * qec;
                    two_ab += mkc * (qkc * qec);
                    b_sq += qec * qkc * qkc;
                }
                -(a_sq as f32) + 2.0 * (two_ab as f32) - b_sq as f32
            } else {
                let (mut a_sq, mut two_ab) = (0.0f64, 0.0f64);
                for c in 0..ck {
                    let (mkc, qkc) = (mkv[c * m + j] as f64, qkv[c * p + i] as f64);
                    a_sq += mkc * mkc;
                    two_ab += mkc * qkc;
                }
                -(a_sq as f32) + 2.0 * (two_ab as f32)
            };
            out[j * p + i] = match &msv {
                Some(s) => sim * s[j] * inv_scale,
                None => sim * inv_scale,
            };
        }
    }
    gpu_upload(&out, m, p)
}

pub fn gpu_cutie_topk_softmax_reference(sim: &GpuTensor, k: usize) -> Result<(GpuTensor, GpuTensor), String> {
    cutie_check_f32("gpu_cutie_topk_softmax", sim)?;
    let (m, p) = (sim.rows(), sim.cols());
    let k = k.min(m);
    let v = gpu_download(sim)?;
    let mut aff = vec![0.0f32; m * p];
    let mut col: Vec<(u32, usize)> = Vec::with_capacity(m);
    for i in 0..p {
        col.clear();
        col.extend((0..m).map(|j| (cutie_topk_key(v[j * p + i]), j)));
        // torch.topk's order: the largest key first, equal keys by the lower row.
        col.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        let exps: Vec<f32> = col[..k].iter().map(|&(_, j)| v[j * p + i].exp()).collect();
        let sum = cutie_reduce4(exps.iter().copied(), 0.0, |a, b| a + b);
        for (e, &(_, j)) in exps.iter().zip(&col[..k]) {
            aff[j * p + i] = e / sum;
        }
    }
    let usage: Vec<f32> = (0..m).map(|j| aff[j * p..(j + 1) * p].iter().sum()).collect();
    Ok((gpu_upload(&aff, m, p)?, gpu_upload(&usage, 1, m)?))
}

/// Reference: exp(x - max) summed in f64 and rounded once, so it stands for the exact softmax
/// whichever order the kernel sums in (a sequential f32 sum of 35k peaked terms is itself off
/// by 5e-5).
pub fn gpu_cutie_softmax_axis0_reference(x: &GpuTensor) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_softmax_axis0", x)?;
    let (m, p) = (x.rows(), x.cols());
    let v = gpu_download(x)?;
    let mut out = vec![0.0f32; m * p];
    for i in 0..p {
        let max = (0..m).map(|j| v[j * p + i]).fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f64;
        for j in 0..m {
            let e = (v[j * p + i] - max).exp();
            out[j * p + i] = e;
            sum += e as f64;
        }
        let sum = sum as f32;
        for j in 0..m {
            out[j * p + i] /= sum;
        }
    }
    gpu_upload(&out, m, p)
}

pub fn gpu_cutie_aggregate_reference(prob: &GpuTensor) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_aggregate", prob)?;
    let (n, p) = (prob.rows(), prob.cols());
    let v = gpu_download(prob)?;
    let mut out = vec![0.0f32; (n + 1) * p];
    let logit = |q: f32| {
        let q = q.clamp(1e-7, 1.0 - 1e-7);
        (q / (1.0 - q)).ln()
    };
    for i in 0..p {
        let bg = cutie_reduce4((0..n).map(|o| 1.0 - v[o * p + i]), 1.0, |a, b| a * b);
        out[i] = logit(bg);
        for o in 0..n {
            out[(o + 1) * p + i] = logit(v[o * p + i]);
        }
    }
    gpu_upload(&out, n + 1, p)
}

pub fn gpu_cutie_attn_mask_reference(aux: &GpuTensor, heads: usize, queries: usize) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_attn_mask", aux)?;
    let (n, p) = (aux.rows(), aux.cols());
    let prob = gpu_cutie_sigmoid_reference(aux)?;
    let agg = gpu_download(&gpu_cutie_aggregate_reference(&prob)?)?;
    let mut fg = vec![false; n * p];
    for i in 0..p {
        let max = (0..=n).map(|r| agg[r * p + i]).fold(f32::NEG_INFINITY, cutie_max_nan);
        for o in 0..n {
            fg[o * p + i] = agg[(o + 1) * p + i] >= max;
        }
    }
    let mut out = vec![0.0f32; n * heads * queries * p];
    for o in 0..n {
        let any_fg = (0..p).any(|i| fg[o * p + i]);
        let any_bg = (0..p).any(|i| !fg[o * p + i]);
        for h in 0..heads {
            for q in 0..queries {
                let foreground_query = q < queries / 2;
                // A row blocked everywhere is let through everywhere.
                let open_all = if foreground_query { !any_fg } else { !any_bg };
                let row = ((o * heads + h) * queries + q) * p;
                for i in 0..p {
                    let blocked = !open_all && (fg[o * p + i] != foreground_query);
                    out[row + i] = if blocked { f32::NEG_INFINITY } else { 0.0 };
                }
            }
        }
    }
    gpu_upload(&out, n * heads * queries, p)
}

pub fn gpu_cutie_add_broadcast_reference(a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_add_broadcast", a)?;
    if a.rows() == 0 || a.cols() != b.cols() || b.rows() % a.rows() != 0 {
        return Err("gpu_cutie_add_broadcast shape mismatch".into());
    }
    let av = gpu_download(a)?;
    let bv = gpu_download(b)?;
    let out: Vec<f32> = bv.iter().enumerate().map(|(i, &g)| av[i % av.len()] + g).collect();
    gpu_upload(&out, b.rows(), b.cols())
}

pub fn gpu_cutie_normalise_rows_reference(x: &GpuTensor, sub: &GpuTensor, div: &GpuTensor) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_normalise_rows", x)?;
    if sub.rows() * sub.cols() != x.rows() || div.rows() * div.cols() != x.rows() {
        return Err("gpu_cutie_normalise_rows shape mismatch".into());
    }
    let (s, d) = (gpu_download(sub)?, gpu_download(div)?);
    let mut v = gpu_download(x)?;
    for (r, row) in v.chunks_mut(x.cols().max(1)).enumerate() {
        for value in row {
            *value = (*value - s[r]) / d[r];
        }
    }
    gpu_upload(&v, x.rows(), x.cols())
}

pub fn gpu_cutie_rgb_planes_reference(rgb: &[u8], plane: usize, scale: f32) -> Result<GpuTensor, String> {
    if plane == 0 || rgb.len() != plane * 3 {
        return Err("gpu_cutie_rgb_planes shape mismatch".into());
    }
    let mut planes = vec![0.0f32; 3 * plane];
    for (i, px) in rgb.chunks_exact(3).enumerate() {
        for c in 0..3 {
            planes[c * plane + i] = px[c] as f32 * scale;
        }
    }
    gpu_upload(&planes, 3, plane)
}

/// `gpu_cutie_resize_bilinear`'s scale for one axis: Python's in / out (a double), made f32 as
/// PyTorch's CUDA multiply takes a scalar.
fn cutie_bilinear_scale(size_in: usize, size_out: usize) -> f32 {
    (size_in as f64 / size_out as f64) as f32
}

/// One axis of `gpu_cutie_resize_bilinear` as torch._decomp's `_upsample_linear` computes it in
/// f32 (size given, align_corners false): for each output index, its lower and upper source
/// indices and the upper one's lambda.
fn cutie_bilinear_axis(size_in: usize, size_out: usize) -> Vec<(usize, usize, f32)> {
    let scale = cutie_bilinear_scale(size_in, size_out);
    (0..size_out)
        .map(|dst| {
            let source = scale * (dst as f32 + 0.5) - 0.5;
            let source = if source < 0.0 { 0.0 } else { source };
            let lower = source as usize;
            let upper = if lower + 1 < size_in { lower + 1 } else { lower };
            (lower, upper, source - lower as f32)
        })
        .collect()
}

pub fn gpu_cutie_resize_bilinear_reference(x: &GpuTensor, in_w: usize, in_h: usize, out_w: usize, out_h: usize) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_resize_bilinear", x)?;
    if x.cols() != in_w * in_h || in_w == 0 || in_h == 0 || out_w == 0 || out_h == 0 {
        return Err("gpu_cutie_resize_bilinear shape mismatch".into());
    }
    let (ax, ay) = (cutie_bilinear_axis(in_w, out_w), cutie_bilinear_axis(in_h, out_h));
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let src = gpu_download(x)?;
    let (in_plane, out_plane) = (in_w * in_h, out_w * out_h);
    let mut out = vec![0.0f32; x.rows() * out_plane];
    for (p, dst) in out.chunks_exact_mut(out_plane).enumerate() {
        let src = &src[p * in_plane..(p + 1) * in_plane];
        for (&(y0, y1, h), dst) in ay.iter().zip(dst.chunks_exact_mut(out_w)) {
            let (row0, row1) = (&src[y0 * in_w..][..in_w], &src[y1 * in_w..][..in_w]);
            for (&(x0, x1, w), d) in ax.iter().zip(dst.iter_mut()) {
                *d = lerp(lerp(row0[x0], row0[x1], w), lerp(row1[x0], row1[x1], w), h);
            }
        }
    }
    gpu_upload(&out, x.rows(), out_plane)
}

pub fn gpu_cutie_object_values_reference(memory: &GpuTensor, objects: usize, eps: f32) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_object_values", memory)?;
    if objects == 0 || memory.rows() % objects != 0 || memory.cols() < 2 {
        return Err("gpu_cutie_object_values shape mismatch".into());
    }
    let (queries, width) = (memory.rows() / objects, memory.cols() - 1);
    let v = gpu_download(memory)?;
    let mut out = vec![0.0f32; objects * queries * width];
    for o in 0..objects {
        for q in 0..queries {
            let row = &v[(o * queries + q) * (width + 1)..(o * queries + q + 1) * (width + 1)];
            let area = row[width] + eps;
            for (dst, &s) in out[(q * objects + o) * width..(q * objects + o + 1) * width].iter_mut().zip(row) {
                *dst = s / area;
            }
        }
    }
    gpu_upload(&out, objects * queries, width)
}

/// Reference: sin and cos in f64, rounded once. CUDA's sinf and cosf are not correctly rounded
/// (about 1 entry in 7 of Cutie's encodings is 1 ulp from this), so the two are fp32 rounding
/// apart.
pub fn gpu_cutie_sine_encoding_reference(inv_freq: &[f32], w: usize, h: usize, scale: f32, eps: f32) -> Result<GpuTensor, String> {
    if inv_freq.is_empty() || w == 0 || h == 0 {
        return Err("gpu_cutie_sine_encoding shape mismatch".into());
    }
    let half = inv_freq.len() * 2;
    let axis = |len: usize| -> Vec<f32> {
        let last = (len - 1) as f32 + eps;
        let mut emb = Vec::with_capacity(len * half);
        for i in 0..len {
            let pos = i as f32 / last * scale;
            for &f in inv_freq {
                let angle = (pos * f) as f64;
                emb.extend([angle.sin() as f32, angle.cos() as f32]);
            }
        }
        emb
    };
    let (ex, ey) = (axis(w), axis(h));
    let mut pe = Vec::with_capacity(h * w * 2 * half);
    for y in 0..h {
        for x in 0..w {
            pe.extend_from_slice(&ex[x * half..(x + 1) * half]);
            pe.extend_from_slice(&ey[y * half..(y + 1) * half]);
        }
    }
    gpu_upload(&pe, h * w, 2 * half)
}

/// Reference: `gpu_layer_norm_pytorch` itself (the same kernel, its affine uploaded per call).
pub fn gpu_cutie_layer_norm_reference(
    x: &GpuTensor, _cache_namespace: &str, _key: &str, gamma: &[f32], beta: &[f32], eps: f32,
) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_layer_norm", x)?;
    gpu_layer_norm_pytorch(x, gamma, beta, eps)
}
