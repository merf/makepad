//! Host versions of the generic `gpu_*` ops Cutie calls (`gpu_slice_cols`, `gpu_concat_cols`,
//! `gpu_pixal_pool`, `gpu_planar_tokens_transpose`, `gpu_realesrgan_scale_add`, `gpu_gather_cols`,
//! `gpu_add_cols_broadcast`, `gpu_birefnet_relu`, `gpu_rife_fill`, `gpu_layer_norm_pytorch`), in
//! plain Rust on row-major f32 values: the truth Cutie's ops check (models/cutie/src/ops_check.rs)
//! holds each backend's op to, CUDA's kernels and Metal's alike (RationalVisions
//! docs/cutie-metal.md, step 2). Platform-free: no device, every OS.
//!
//! Each makes CUDA's checks (cuda/src/launch.rs) with its messages, and computes what CUDA's
//! kernel does: the moves and the one add to the bit; the pool's window summed in row order and
//! divided once; the scale add as CUDA's one `fmaf`. The layer norm takes its moments and its
//! result in f64 and rounds once, so it is within fp32 rounding of PyTorch's Welford order and
//! CUDA's `rsqrtf`, not their bits.
//!
//! Where CUDA's kernels and PyTorch differ, these follow CUDA. The pool divides its sum once by
//! the window's area; PyTorch's area interpolate (`adaptive_avg_pool2d`) divides by the window's
//! height, then by its width. The two agree wherever the area is a power of two, as at Cutie's
//! strides 4 and 16, but can differ by an ulp at odd windows. ReLU is `fmaxf`, so NaN gives 0;
//! `torch.relu` keeps NaN.

/// A row-major `[rows, cols]` matrix of f32 values, borrowed.
#[derive(Clone, Copy, Debug)]
pub struct HostMat<'a> {
    values: &'a [f32],
    rows: usize,
    cols: usize,
}

impl<'a> HostMat<'a> {
    /// `values` as `[rows, cols]`; `Err` unless there are exactly `rows * cols` of them.
    pub fn new(values: &'a [f32], rows: usize, cols: usize) -> Result<Self, String> {
        if rows.checked_mul(cols) != Some(values.len()) {
            return Err(format!("host_ops: {} values are not {rows}x{cols}", values.len()));
        }
        Ok(Self { values, rows, cols })
    }

    pub fn values(&self) -> &'a [f32] {
        self.values
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    fn row(&self, r: usize) -> &'a [f32] {
        &self.values[r * self.cols..(r + 1) * self.cols]
    }
}

/// The largest side `gpu_pixal_pool` takes (CUDA's), so its window arithmetic fits a u32.
const PIXAL_POOL_MAX_SIDE: usize = 65535;

/// Columns `start..start + len` of `x`: `[rows, len]` (CUDA's `gpu_slice_cols`).
pub fn slice_cols(x: HostMat, start: usize, len: usize) -> Result<Vec<f32>, String> {
    if start.checked_add(len).is_none_or(|end| end > x.cols) {
        return Err("gpu_slice_cols out of range".to_string());
    }
    Ok((0..x.rows).flat_map(|r| &x.row(r)[start..start + len]).copied().collect())
}

/// `parts` side by side: `[rows, sum of their cols]`, each part's columns after the one before's;
/// a part may appear more than once (CUDA's `gpu_concat_cols`: at least one row, the same in
/// every part).
pub fn concat_cols(parts: &[HostMat]) -> Result<Vec<f32>, String> {
    let rows = parts.first().map_or(0, |part| part.rows);
    if rows == 0 || parts.iter().any(|part| part.rows != rows) {
        return Err("gpu_concat_cols shape mismatch".to_string());
    }
    let cols: usize = parts.iter().map(|part| part.cols).sum();
    let mut out = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for part in parts {
            out.extend_from_slice(part.row(r));
        }
    }
    Ok(out)
}

/// Adaptive average pooling of planar `x` `[channels, height*width]` to `[channels,
/// out_height*out_width]` (CUDA's `gpu_pixal_pool`, `pixal_pool_kernel`): each output the mean of
/// its window `[floor(i*in/out), ceil((i+1)*in/out))` in each direction, summed in f32 row by row
/// and divided once by the window's size (PyTorch divides by its height, then its width).
pub fn pixal_pool(x: HostMat, width: usize, height: usize, out_width: usize, out_height: usize) -> Result<Vec<f32>, String> {
    if width == 0
        || height == 0
        || out_width == 0
        || out_height == 0
        || width.checked_mul(height) != Some(x.cols)
        || out_width.checked_mul(out_height).is_none()
        || [width, height, out_width, out_height, x.rows].iter().any(|&n| n > PIXAL_POOL_MAX_SIDE)
    {
        return Err("gpu_pixal_pool shape/dtype mismatch".to_string());
    }
    let mut out = Vec::with_capacity(x.rows * out_width * out_height);
    for c in 0..x.rows {
        let plane = x.row(c);
        for py in 0..out_height {
            let (y0, y1) = (py * height / out_height, ((py + 1) * height).div_ceil(out_height));
            for px in 0..out_width {
                let (x0, x1) = (px * width / out_width, ((px + 1) * width).div_ceil(out_width));
                let mut sum = 0.0f32;
                for y in y0..y1 {
                    for v in &plane[y * width + x0..y * width + x1] {
                        sum += v;
                    }
                }
                out.push(sum / ((x1 - x0) * (y1 - y0)) as f32);
            }
        }
    }
    Ok(out)
}

/// `x` `[rows, cols]` transposed to `[cols, rows]`: planar `[C, HW]` to tokens `[HW, C]` and back
/// (CUDA's `gpu_planar_tokens_transpose`).
pub fn planar_tokens_transpose(x: HostMat) -> Vec<f32> {
    let mut out = vec![0.0f32; x.values.len()];
    for r in 0..x.rows {
        for (c, &v) in x.row(r).iter().enumerate() {
            out[c * x.rows + r] = v;
        }
    }
    out
}

/// `base + scale * delta` elementwise with one rounding, CUDA's `fmaf(scale, delta, base)` (its
/// `gpu_realesrgan_scale_add`; Cutie's `1 - mask` is `scale = -1`).
pub fn realesrgan_scale_add(base: HostMat, delta: HostMat, scale: f32) -> Result<Vec<f32>, String> {
    if base.rows != delta.rows || base.cols != delta.cols {
        return Err(format!("gpu_realesrgan_scale_add {}x{} vs {}x{}", base.rows, base.cols, delta.rows, delta.cols));
    }
    Ok(base.values.iter().zip(delta.values).map(|(&b, &d)| scale.mul_add(d, b)).collect())
}

/// The columns `indices` of `x`, in that order, repeats allowed: `[rows, indices.len()]` (CUDA's
/// `gpu_gather_cols`, which checks every index first).
pub fn gather_cols(x: HostMat, indices: &[u32]) -> Result<Vec<f32>, String> {
    if indices.is_empty() {
        return Err("gpu_gather_cols: empty index table".to_string());
    }
    if let Some(&index) = indices.iter().find(|&&index| index as usize >= x.cols) {
        return Err(format!("gpu_gather_cols index {index} out of range (cols {})", x.cols));
    }
    Ok((0..x.rows).flat_map(|r| indices.iter().map(move |&i| x.row(r)[i as usize])).collect())
}

/// `x[r, c] + bias[c]`, the bias's `rows * cols` values read as one row as wide as `x` (CUDA's
/// `gpu_add_cols_broadcast`).
pub fn add_cols_broadcast(x: HostMat, bias: HostMat) -> Result<Vec<f32>, String> {
    if bias.values.len() != x.cols {
        return Err(format!(
            "gpu_add_cols_broadcast shape mismatch {}x{} half=false vs bias {}x{} half=false",
            x.rows, x.cols, bias.rows, bias.cols
        ));
    }
    Ok((0..x.rows).flat_map(|r| x.row(r).iter().zip(bias.values).map(|(&v, &b)| v + b)).collect())
}

/// `fmaxf(x, 0)` (CUDA's `gpu_birefnet_relu`): NaN and -inf give 0 (`torch.relu` keeps NaN).
pub fn birefnet_relu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v.max(0.0)).collect()
}

/// `[rows, cols]` of `value` (CUDA's `gpu_rife_fill`, which refuses an empty extent).
pub fn rife_fill(rows: usize, cols: usize, value: f32) -> Result<Vec<f32>, String> {
    if rows == 0 || cols == 0 {
        return Err("gpu_rife_fill needs a non-empty extent".to_string());
    }
    let n = rows.checked_mul(cols).ok_or_else(|| format!("gpu_rife_fill: {rows}x{cols} overflows"))?;
    Ok(vec![value; n])
}

/// nn.LayerNorm over each row, `scale * ((x - mean) / sqrt(var + eps)) + bias` with the biased
/// variance (CUDA's `gpu_layer_norm_pytorch`, which takes rows a multiple of 4 wide): the moments
/// and the result in f64, rounded once.
pub fn layer_norm_pytorch(x: HostMat, scale: &[f32], bias: &[f32], eps: f32) -> Result<Vec<f32>, String> {
    if scale.len() != x.cols || bias.len() != x.cols || !x.cols.is_multiple_of(4) {
        return Err("gpu_layer_norm_pytorch width mismatch".to_string());
    }
    let n = x.cols as f64;
    let mut out = Vec::with_capacity(x.values.len());
    for r in 0..x.rows {
        let row = x.row(r);
        let mean = row.iter().map(|&v| v as f64).sum::<f64>() / n;
        let var = row.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
        let rstd = 1.0 / (var + eps as f64).sqrt();
        out.extend(row.iter().zip(scale.iter().zip(bias)).map(|(&v, (&g, &b))| (g as f64 * (rstd * (v as f64 - mean)) + b as f64) as f32));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mat(values: &[f32], rows: usize, cols: usize) -> HostMat<'_> {
        HostMat::new(values, rows, cols).unwrap()
    }

    /// 0, 1, 2, ... as f32: `[rows, cols]` whose value is its index.
    fn iota(n: usize) -> Vec<f32> {
        (0..n).map(|i| i as f32).collect()
    }

    #[test]
    fn a_matrix_holds_rows_times_cols_values() {
        assert_eq!(HostMat::new(&[0.0; 6], 2, 4).unwrap_err(), "host_ops: 6 values are not 2x4");
        assert!(HostMat::new(&[], usize::MAX, 2).is_err(), "rows * cols overflows");
        let m = mat(&[1.0, 2.0, 3.0], 3, 1);
        assert_eq!((m.rows(), m.cols(), m.values()), (3, 1, &[1.0, 2.0, 3.0][..]));
        assert_eq!(mat(&[], 0, 5).rows(), 0);
    }

    #[test]
    fn slices_and_concatenations_move_every_value_where_cuda_does() {
        let xs = iota(12);
        let x = mat(&xs, 3, 4);
        assert_eq!(slice_cols(x, 1, 2).unwrap(), [1.0, 2.0, 5.0, 6.0, 9.0, 10.0]);
        assert_eq!(slice_cols(x, 0, 4).unwrap(), xs);
        assert_eq!(slice_cols(x, 4, 0).unwrap(), Vec::<f32>::new());
        assert_eq!(slice_cols(x, 3, 2).unwrap_err(), "gpu_slice_cols out of range");
        assert_eq!(slice_cols(x, usize::MAX, 2).unwrap_err(), "gpu_slice_cols out of range", "start + len overflows");

        let ys = [-1.0, -2.0, -3.0];
        let y = mat(&ys, 3, 1);
        assert_eq!(
            concat_cols(&[x, y, x]).unwrap(),
            [
                0.0, 1.0, 2.0, 3.0, -1.0, 0.0, 1.0, 2.0, 3.0, //
                4.0, 5.0, 6.0, 7.0, -2.0, 4.0, 5.0, 6.0, 7.0, //
                8.0, 9.0, 10.0, 11.0, -3.0, 8.0, 9.0, 10.0, 11.0,
            ]
        );
        assert_eq!(concat_cols(&[y]).unwrap(), ys);
        assert_eq!(concat_cols(&[x, mat(&[], 3, 0), y]).unwrap().len(), 15, "a part with no columns adds none");
        let mismatch = "gpu_concat_cols shape mismatch";
        assert_eq!(concat_cols(&[]).unwrap_err(), mismatch);
        assert_eq!(concat_cols(&[mat(&[], 0, 4)]).unwrap_err(), mismatch, "no rows");
        assert_eq!(concat_cols(&[x, mat(&[0.0; 2], 2, 1)]).unwrap_err(), mismatch);
    }

    #[test]
    fn the_pool_averages_cudas_windows() {
        // 4x2 to 2x1: two windows of 2x2.
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert_eq!(pixal_pool(mat(&xs, 1, 8), 4, 2, 2, 1).unwrap(), [3.5, 5.5]);
        // 3 to 2 wide: the windows [0, 2) and [1, 3) overlap.
        assert_eq!(pixal_pool(mat(&[1.0, 2.0, 4.0], 1, 3), 3, 1, 2, 1).unwrap(), [1.5, 3.0]);
        // 1x1 up to 3x2: every window the one pixel; two planes.
        assert_eq!(pixal_pool(mat(&[7.0, -1.0], 2, 1), 1, 1, 3, 2).unwrap(), [7.0, 7.0, 7.0, 7.0, 7.0, 7.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]);
        // 5 to 3: [0, 2), [1, 4), [3, 5); summed in order, divided once.
        let row = [0.1f32, 0.2, 0.3, 0.4, 0.5];
        let want = [(0.1f32 + 0.2) / 2.0, (0.2f32 + 0.3 + 0.4) / 3.0, (0.4f32 + 0.5) / 2.0];
        assert_eq!(pixal_pool(mat(&row, 1, 5), 5, 1, 3, 1).unwrap(), want);
        // A NaN poisons only its windows.
        let out = pixal_pool(mat(&[f32::NAN, 1.0, 2.0, 3.0], 1, 4), 4, 1, 2, 1).unwrap();
        assert!(out[0].is_nan() && out[1] == 2.5);

        let mismatch = "gpu_pixal_pool shape/dtype mismatch";
        let x = mat(&[0.0; 12], 1, 12);
        assert_eq!(pixal_pool(x, 4, 3, 0, 2).unwrap_err(), mismatch);
        assert_eq!(pixal_pool(x, 4, 3, 2, 0).unwrap_err(), mismatch);
        assert_eq!(pixal_pool(x, 5, 3, 2, 2).unwrap_err(), mismatch, "width * height is not cols");
        assert_eq!(pixal_pool(x, 4, 3, 65536, 1).unwrap_err(), mismatch, "CUDA's 65535");
        assert_eq!(pixal_pool(mat(&[], 0, 12), 4, 3, 2, 2).unwrap(), Vec::<f32>::new(), "no channels");
    }

    #[test]
    fn the_transpose_swaps_rows_and_columns() {
        let xs = iota(6);
        assert_eq!(planar_tokens_transpose(mat(&xs, 2, 3)), [0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
        assert_eq!(planar_tokens_transpose(mat(&xs, 1, 6)), xs);
        let back = planar_tokens_transpose(mat(&planar_tokens_transpose(mat(&xs, 3, 2)), 2, 3));
        assert_eq!(back, xs);
    }

    #[test]
    fn the_scale_add_rounds_once_as_cudas_fmaf() {
        // (1 + 2^-23)(1 - 2^-23) = 1 - 2^-46: rounded to 1 before the add it would give 0.
        let (scale, d, b) = (f32::from_bits(0x3f80_0001), f32::from_bits(0x3f7f_fffe), -1.0f32);
        assert_ne!(scale * d + b, -(2.0f32).powi(-46), "separately rounded");
        let got = realesrgan_scale_add(mat(&[b, 0.25], 1, 2), mat(&[d, 0.5], 1, 2), scale).unwrap();
        assert_eq!(got[0], -(2.0f32).powi(-46));
        // Cutie's 1 - mask.
        let ones = [1.0; 3];
        let got = realesrgan_scale_add(mat(&ones, 3, 1), mat(&[0.0, 0.25, 1.0], 3, 1), -1.0).unwrap();
        assert_eq!(got, [1.0, 0.75, 0.0]);
        let e = realesrgan_scale_add(mat(&ones, 3, 1), mat(&ones, 1, 3), 1.0).unwrap_err();
        assert_eq!(e, "gpu_realesrgan_scale_add 3x1 vs 1x3");
    }

    #[test]
    fn gathers_check_every_index_first() {
        let xs = iota(8);
        let x = mat(&xs, 2, 4);
        assert_eq!(gather_cols(x, &[3, 0, 3]).unwrap(), [3.0, 0.0, 3.0, 7.0, 4.0, 7.0]);
        assert_eq!(gather_cols(x, &[]).unwrap_err(), "gpu_gather_cols: empty index table");
        assert_eq!(gather_cols(x, &[0, 4, 9]).unwrap_err(), "gpu_gather_cols index 4 out of range (cols 4)");
    }

    #[test]
    fn the_bias_is_added_to_every_row() {
        let xs = iota(8);
        let bias = [10.0, 20.0, 30.0, 40.0];
        let want = [10.0, 21.0, 32.0, 43.0, 14.0, 25.0, 36.0, 47.0];
        assert_eq!(add_cols_broadcast(mat(&xs, 2, 4), mat(&bias, 1, 4)).unwrap(), want);
        assert_eq!(add_cols_broadcast(mat(&xs, 2, 4), mat(&bias, 2, 2)).unwrap(), want, "the bias's shape is not its width");
        let e = add_cols_broadcast(mat(&xs, 2, 4), mat(&bias[..3], 3, 1)).unwrap_err();
        assert_eq!(e, "gpu_add_cols_broadcast shape mismatch 2x4 half=false vs bias 3x1 half=false");
    }

    #[test]
    fn relu_is_fmaxf_and_fill_refuses_nothing() {
        let got = birefnet_relu(&[-1.0, 2.0, f32::NAN, f32::NEG_INFINITY, f32::INFINITY, -0.0]);
        assert_eq!(got[..5], [0.0, 2.0, 0.0, 0.0, f32::INFINITY]);
        assert_eq!(got[5], 0.0);

        assert_eq!(rife_fill(2, 3, 0.5).unwrap(), [0.5; 6]);
        assert_eq!(rife_fill(0, 3, 0.5).unwrap_err(), "gpu_rife_fill needs a non-empty extent");
        assert_eq!(rife_fill(3, 0, 0.5).unwrap_err(), "gpu_rife_fill needs a non-empty extent");
        assert!(rife_fill(usize::MAX, 2, 0.0).is_err());
    }

    #[test]
    fn the_layer_norm_is_pytorchs_in_f64() {
        // [1, 2, 3, 4]: mean 2.5, biased variance 1.25.
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0, 5.0];
        let x = mat(&xs, 2, 4);
        let (gamma, beta) = ([1.0, 2.0, -1.0, 0.5], [0.0, 1.0, 0.0, -1.0]);
        let got = layer_norm_pytorch(x, &gamma, &beta, 1e-5).unwrap();
        let rstd = 1.0 / (1.25f64 + 1e-5f32 as f64).sqrt();
        for i in 0..4 {
            let want = (gamma[i] as f64 * (rstd * (xs[i] as f64 - 2.5)) + beta[i] as f64) as f32;
            assert_eq!(got[i], want, "{i}");
        }
        assert!((got[0] + 1.341635).abs() < 1e-5, "{}", got[0]);
        // A constant row is its bias.
        assert_eq!(got[4..], beta);

        let mismatch = "gpu_layer_norm_pytorch width mismatch";
        assert_eq!(layer_norm_pytorch(x, &gamma[..3], &beta, 1e-5).unwrap_err(), mismatch);
        assert_eq!(layer_norm_pytorch(x, &gamma, &beta[..3], 1e-5).unwrap_err(), mismatch);
        let odd = [0.0; 6];
        assert_eq!(layer_norm_pytorch(mat(&odd, 1, 6), &odd, &odd, 1e-5).unwrap_err(), mismatch, "CUDA's 4 floats a lane");
    }
}
