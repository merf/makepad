// Cutie's device ops (models/vision/src/cutie*.rs), included inside launch.rs's CUDA `mod imp` so
// they can use its private helpers. The stubs for builds without kernels are in
// models/common/src/gpu_cutie_stubs.rs, with the same signatures.
//
// Every op is f32 with exact fp32 arithmetic: Cutie is checked against a PyTorch fp32 oracle with
// TF32 off. Each `gpu_cutie_X` has a host reference `gpu_cutie_X_reference` (download, compute on
// the host, upload): obviously correct and slow. `gpu_cutie_X` calls its reference until it has a
// kernel in kernels/cutie.cu; a kernel must then match its reference (`cutie-validate ops`).
//
// Layouts: tensors are row-major [rows, cols]. A batch of planar images is [batch*channels,
// height*width], each image's channels contiguous rows, which is the memory order of PyTorch's
// contiguous (batch, channels, height, width).

unsafe extern "C" {
    fn makepad_cuda_cutie_maxpool3x3s2_f32(
        x: *const f32, out: *mut f32, width: u32, height: u32, ow: u32, oh: u32, planes: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
}

fn cutie_check_f32(op: &str, x: &GpuTensor) -> Result<(), String> {
    if x.half {
        return Err(format!("{op} is f32-only"));
    }
    Ok(())
}

/// A convolution's output size along one axis (PyTorch's floor rule).
pub fn cutie_conv_out(size: usize, k: usize, pad: usize, stride: usize) -> usize {
    (size + 2 * pad - k) / stride + 1
}

// --- Convolution ---

/// Conv2d over a batch of planar images: `x` is [batch*in_channels, in_h*in_w], `weights` PyTorch's
/// [out][in][k][k], `bias` per output channel (None: no bias). Square kernel, symmetric padding and
/// stride. Out: [batch*out_channels, out_h*out_w], out_h = (in_h + 2*pad - k)/stride + 1. Weights
/// are cached on the device under `cache_namespace::key` (one key per layer).
pub fn gpu_cutie_conv2d(
    x: &GpuTensor, batch: usize, in_w: usize, in_h: usize, cache_namespace: &str, key: &str,
    weights: &[f32], bias: Option<&[f32]>, out_channels: usize, k: usize, pad: usize, stride: usize,
) -> Result<GpuTensor, String> {
    gpu_cutie_conv2d_reference(x, batch, in_w, in_h, cache_namespace, key, weights, bias, out_channels, k, pad, stride)
}

/// Reference: the existing direct fp32 kernel (`gpu_conv2d_planar_strided`) once per image.
pub fn gpu_cutie_conv2d_reference(
    x: &GpuTensor, batch: usize, in_w: usize, in_h: usize, cache_namespace: &str, key: &str,
    weights: &[f32], bias: Option<&[f32]>, out_channels: usize, k: usize, pad: usize, stride: usize,
) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_conv2d", x)?;
    if batch == 0 || x.rows % batch != 0 || x.cols != in_w * in_h {
        return Err(format!("gpu_cutie_conv2d {key}: x {}x{} for batch {batch} at {in_w}x{in_h}", x.rows, x.cols));
    }
    let in_channels = x.rows / batch;
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

// --- Pooling ---

/// MaxPool2d(3, stride 2, padding 1) on planar [planes, h*w]. Out [planes, oh*ow], oh = (h-1)/2+1.
pub fn gpu_cutie_maxpool3x3s2(x: &GpuTensor, w: usize, h: usize) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_maxpool3x3s2", x)?;
    if x.cols != w * h || w == 0 || h == 0 {
        return Err("gpu_cutie_maxpool3x3s2 shape mismatch".into());
    }
    let (ow, oh) = (cutie_conv_out(w, 3, 1, 2), cutie_conv_out(h, 3, 1, 2));
    with_dense_linear_backend(|backend| {
        backend.prepare_device()?;
        let out = GpuTensor::from_pool(x.rows, ow * oh)?;
        gpu_check(unsafe {
            makepad_cuda_cutie_maxpool3x3s2_f32(
                x.device_ptr()?, out.device_ptr()?, w as u32, h as u32, ow as u32, oh as u32, x.rows as u32,
                backend.stream,
            )
        })?;
        Ok(out)
    })
}

pub fn gpu_cutie_maxpool3x3s2_reference(x: &GpuTensor, w: usize, h: usize) -> Result<GpuTensor, String> {
    let v = gpu_download(x)?;
    let (ow, oh) = (cutie_conv_out(w, 3, 1, 2), cutie_conv_out(h, 3, 1, 2));
    let mut out = vec![0.0f32; x.rows * ow * oh];
    for p in 0..x.rows {
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
                        best = best.max(v[p * w * h + y as usize * w + xx as usize]);
                    }
                }
                out[p * ow * oh + oy * ow + ox] = best;
            }
        }
    }
    gpu_upload(&out, x.rows, ow * oh)
}

// --- Elementwise ---

/// 1 / (1 + exp(-x)), elementwise.
pub fn gpu_cutie_sigmoid(x: &GpuTensor) -> Result<GpuTensor, String> {
    gpu_cutie_sigmoid_reference(x)
}

pub fn gpu_cutie_sigmoid_reference(x: &GpuTensor) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_sigmoid", x)?;
    let v: Vec<f32> = gpu_download(x)?.into_iter().map(|z| 1.0 / (1.0 + (-z).exp())).collect();
    gpu_upload(&v, x.rows, x.cols)
}

/// Cutie's GRU-like sensory update, per object: `values` [n*3c, p] holds each object's forget,
/// update and new-value rows (c each); `h` [n*c, p]. new_h = sigmoid(f)*h*(1-sigmoid(u)) +
/// sigmoid(u)*tanh(v), evaluated in that order.
pub fn gpu_cutie_gru(h: &GpuTensor, values: &GpuTensor, n: usize, c: usize) -> Result<GpuTensor, String> {
    gpu_cutie_gru_reference(h, values, n, c)
}

pub fn gpu_cutie_gru_reference(h: &GpuTensor, values: &GpuTensor, n: usize, c: usize) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_gru", h)?;
    if h.rows != n * c || values.rows != n * 3 * c || h.cols != values.cols {
        return Err("gpu_cutie_gru shape mismatch".into());
    }
    let p = h.cols;
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

/// The tail of Cutie's channel-attention block (ECA): per image b of `x` [n*c, p], the mean of
/// each channel, a 1-D convolution over channels (`conv_w`, odd length k, zero padding (k-1)/2,
/// no bias), a sigmoid, then x * that + `residual` ([n*c, p]).
pub fn gpu_cutie_eca(x: &GpuTensor, residual: &GpuTensor, n: usize, c: usize, conv_w: &[f32]) -> Result<GpuTensor, String> {
    gpu_cutie_eca_reference(x, residual, n, c, conv_w)
}

pub fn gpu_cutie_eca_reference(x: &GpuTensor, residual: &GpuTensor, n: usize, c: usize, conv_w: &[f32]) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_eca", x)?;
    if x.rows != n * c || residual.rows != x.rows || residual.cols != x.cols || conv_w.len() % 2 == 0 {
        return Err("gpu_cutie_eca shape mismatch".into());
    }
    let p = x.cols;
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

/// For each object's mask in `masks` [n, p], the sum of the other objects' masks, clamped to 0..1
/// (Cutie's `_get_others`).
pub fn gpu_cutie_others(masks: &GpuTensor) -> Result<GpuTensor, String> {
    gpu_cutie_others_reference(masks)
}

pub fn gpu_cutie_others_reference(masks: &GpuTensor) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_others", masks)?;
    let (n, p) = (masks.rows, masks.cols);
    let v = gpu_download(masks)?;
    let mut out = vec![0.0f32; n * p];
    for i in 0..p {
        let mut sum = 0.0f32;
        for o in 0..n {
            sum += v[o * p + i];
        }
        for o in 0..n {
            out[o * p + i] = (sum - v[o * p + i]).clamp(0.0, 1.0);
        }
    }
    gpu_upload(&out, n, p)
}

// --- Shapes ---

/// Zero padding of planar [planes, h*w] by (left, right, top, bottom). Out [planes, (h+t+b)*(w+l+r)].
pub fn gpu_cutie_pad(x: &GpuTensor, w: usize, h: usize, left: usize, right: usize, top: usize, bottom: usize) -> Result<GpuTensor, String> {
    gpu_cutie_pad_reference(x, w, h, left, right, top, bottom)
}

pub fn gpu_cutie_pad_reference(x: &GpuTensor, w: usize, h: usize, left: usize, right: usize, top: usize, bottom: usize) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_pad", x)?;
    if x.cols != w * h {
        return Err("gpu_cutie_pad shape mismatch".into());
    }
    let (nw, nh) = (w + left + right, h + top + bottom);
    let v = gpu_download(x)?;
    let mut out = vec![0.0f32; x.rows * nw * nh];
    for p in 0..x.rows {
        for y in 0..h {
            let src = &v[p * w * h + y * w..p * w * h + (y + 1) * w];
            let at = p * nw * nh + (y + top) * nw + left;
            out[at..at + w].copy_from_slice(src);
        }
    }
    gpu_upload(&out, x.rows, nw * nh)
}

/// A window of planar [planes, h*w]: `out_w` x `out_h` from (`left`, `top`).
pub fn gpu_cutie_crop(x: &GpuTensor, w: usize, h: usize, left: usize, top: usize, out_w: usize, out_h: usize) -> Result<GpuTensor, String> {
    gpu_cutie_crop_reference(x, w, h, left, top, out_w, out_h)
}

pub fn gpu_cutie_crop_reference(x: &GpuTensor, w: usize, h: usize, left: usize, top: usize, out_w: usize, out_h: usize) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_crop", x)?;
    if x.cols != w * h || left + out_w > w || top + out_h > h {
        return Err("gpu_cutie_crop shape mismatch".into());
    }
    let v = gpu_download(x)?;
    let mut out = vec![0.0f32; x.rows * out_w * out_h];
    for p in 0..x.rows {
        for y in 0..out_h {
            let src = p * w * h + (y + top) * w + left;
            let at = p * out_w * out_h + y * out_w;
            out[at..at + out_w].copy_from_slice(&v[src..src + out_w]);
        }
    }
    gpu_upload(&out, x.rows, out_w * out_h)
}

// --- Matrix products ---

/// op(a) @ op(b), op transposing when asked: op(a) [m, k], op(b) [k, n]; out [m, n]. fp32
/// multiply-accumulate (no TF32).
pub fn gpu_cutie_matmul(a: &GpuTensor, trans_a: bool, b: &GpuTensor, trans_b: bool) -> Result<GpuTensor, String> {
    gpu_cutie_matmul_reference(a, trans_a, b, trans_b)
}

pub fn gpu_cutie_matmul_reference(a: &GpuTensor, trans_a: bool, b: &GpuTensor, trans_b: bool) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_matmul", a)?;
    cutie_check_f32("gpu_cutie_matmul", b)?;
    let (m, k) = if trans_a { (a.cols, a.rows) } else { (a.rows, a.cols) };
    let (k2, n) = if trans_b { (b.cols, b.rows) } else { (b.rows, b.cols) };
    if k != k2 {
        return Err(format!("gpu_cutie_matmul inner dims {k} != {k2}"));
    }
    let av = gpu_download(a)?;
    let bv = gpu_download(b)?;
    let at = |i: usize, j: usize| if trans_a { av[j * a.cols + i] } else { av[i * a.cols + j] };
    let bt = |i: usize, j: usize| if trans_b { bv[j * b.cols + i] } else { bv[i * b.cols + j] };
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

// --- Memory reading ---

/// Cutie's anisotropic L2 similarity (XMem's appendix): memory keys `mk` [ck, m], shrinkage `ms`
/// [1, m] (None: 1), query keys `qk` [ck, p], selection `qe` [ck, p] (None: plain L2 as STCN).
/// Out [m, p]: (-a_sq + two_ab - b_sq) * ms / sqrt(ck), with a_sq = mk^2 . qe, two_ab =
/// 2 mk . (qk*qe), b_sq = sum_c qe*qk^2 (without qe: a_sq = |mk|^2, two_ab = 2 mk . qk, no b_sq).
pub fn gpu_cutie_similarity(mk: &GpuTensor, ms: Option<&GpuTensor>, qk: &GpuTensor, qe: Option<&GpuTensor>) -> Result<GpuTensor, String> {
    gpu_cutie_similarity_reference(mk, ms, qk, qe)
}

pub fn gpu_cutie_similarity_reference(mk: &GpuTensor, ms: Option<&GpuTensor>, qk: &GpuTensor, qe: Option<&GpuTensor>) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_similarity", mk)?;
    let (ck, m, p) = (mk.rows, mk.cols, qk.cols);
    if qk.rows != ck || qe.is_some_and(|e| e.rows != ck || e.cols != p) || ms.is_some_and(|s| s.rows * s.cols != m) {
        return Err("gpu_cutie_similarity shape mismatch".into());
    }
    let mkv = gpu_download(mk)?;
    let qkv = gpu_download(qk)?;
    let qev = qe.map(gpu_download).transpose()?;
    let msv = ms.map(gpu_download).transpose()?;
    let scale = (ck as f32).sqrt();
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
                Some(s) => sim * s[j] / scale,
                None => sim / scale,
            };
        }
    }
    gpu_upload(&out, m, p)
}

/// Cutie's top-k softmax down each column of `sim` [m, p]: the k largest (k = min(k, m)), their
/// exp normalised to sum 1 (no max subtracted, as Cutie's `do_softmax`), every other entry 0.
/// Also each row's sum of the result ([1, m]): how much each memory token was used.
pub fn gpu_cutie_topk_softmax(sim: &GpuTensor, k: usize) -> Result<(GpuTensor, GpuTensor), String> {
    gpu_cutie_topk_softmax_reference(sim, k)
}

pub fn gpu_cutie_topk_softmax_reference(sim: &GpuTensor, k: usize) -> Result<(GpuTensor, GpuTensor), String> {
    cutie_check_f32("gpu_cutie_topk_softmax", sim)?;
    let (m, p) = (sim.rows, sim.cols);
    let k = k.min(m);
    let v = gpu_download(sim)?;
    let mut aff = vec![0.0f32; m * p];
    let mut col: Vec<(f32, usize)> = Vec::with_capacity(m);
    for i in 0..p {
        col.clear();
        col.extend((0..m).map(|j| (v[j * p + i], j)));
        // Largest first; equal values by lower index.
        col.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        let exps: Vec<f32> = col[..k].iter().map(|&(s, _)| s.exp()).collect();
        let sum: f32 = exps.iter().sum();
        for (e, &(_, j)) in exps.iter().zip(&col[..k]) {
            aff[j * p + i] = e / sum;
        }
    }
    let usage: Vec<f32> = (0..m).map(|j| aff[j * p..(j + 1) * p].iter().sum()).collect();
    Ok((gpu_upload(&aff, m, p)?, gpu_upload(&usage, 1, m)?))
}

/// Softmax down each column of `x` [m, p] (over the rows), the max subtracted first.
pub fn gpu_cutie_softmax_axis0(x: &GpuTensor) -> Result<GpuTensor, String> {
    gpu_cutie_softmax_axis0_reference(x)
}

pub fn gpu_cutie_softmax_axis0_reference(x: &GpuTensor) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_softmax_axis0", x)?;
    let (m, p) = (x.rows, x.cols);
    let v = gpu_download(x)?;
    let mut out = vec![0.0f32; m * p];
    for i in 0..p {
        let max = (0..m).map(|j| v[j * p + i]).fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for j in 0..m {
            let e = (v[j * p + i] - max).exp();
            out[j * p + i] = e;
            sum += e;
        }
        for j in 0..m {
            out[j * p + i] /= sum;
        }
    }
    gpu_upload(&out, m, p)
}

// --- Objects ---

/// Cutie's `aggregate`: objects' probabilities `prob` [n, p] to logits with a background row first
/// ([n+1, p]): the background is prod(1 - prob), every row clamped to 1e-7..1-1e-7, then
/// log(q / (1 - q)).
pub fn gpu_cutie_aggregate(prob: &GpuTensor) -> Result<GpuTensor, String> {
    gpu_cutie_aggregate_reference(prob)
}

pub fn gpu_cutie_aggregate_reference(prob: &GpuTensor) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_aggregate", prob)?;
    let (n, p) = (prob.rows, prob.cols);
    let v = gpu_download(prob)?;
    let mut out = vec![0.0f32; (n + 1) * p];
    let logit = |q: f32| {
        let q = q.clamp(1e-7, 1.0 - 1e-7);
        (q / (1.0 - q)).ln()
    };
    for i in 0..p {
        let mut bg = 1.0f32;
        for o in 0..n {
            bg *= 1.0 - v[o * p + i];
        }
        out[i] = logit(bg);
        for o in 0..n {
            out[(o + 1) * p + i] = logit(v[o * p + i]);
        }
    }
    gpu_upload(&out, n + 1, p)
}

/// The object transformer's attention mask from its auxiliary logits `aux` [n, p] (one row per
/// object): sigmoid, aggregated with a background (as `gpu_cutie_aggregate`); a pixel is an
/// object's foreground where its logit is the largest of all rows. For each object, each of
/// `heads` heads and each of `queries` queries a row of p: the first half of the queries may only
/// see the object's foreground, the second half only the rest; a row that would see nothing sees
/// everything. Out [n*heads*queries, p], additive: 0 where seen, -inf where blocked.
pub fn gpu_cutie_attn_mask(aux: &GpuTensor, heads: usize, queries: usize) -> Result<GpuTensor, String> {
    gpu_cutie_attn_mask_reference(aux, heads, queries)
}

pub fn gpu_cutie_attn_mask_reference(aux: &GpuTensor, heads: usize, queries: usize) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_attn_mask", aux)?;
    let (n, p) = (aux.rows, aux.cols);
    let prob = gpu_cutie_sigmoid_reference(aux)?;
    let agg = gpu_download(&gpu_cutie_aggregate_reference(&prob)?)?;
    let mut fg = vec![false; n * p];
    for i in 0..p {
        let max = (0..=n).map(|r| agg[r * p + i]).fold(f32::NEG_INFINITY, f32::max);
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
