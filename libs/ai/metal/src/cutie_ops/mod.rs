//! Cutie's ops as Makepad's Metal kernels (RationalVisions docs/cutie-metal.md §8, step 8): one
//! module and one kernel library a family, each op at the signature of its CUDA twin
//! (cuda/src/cutie_ops.rs), with CUDA's checks and messages, its kernels ported from
//! cuda/kernels/cutie.cu. models/common's gpu_cutie_host.rs calls them on macOS, or each op's host
//! reference (cuda/src/cutie_reference.rs) where MAKEPAD_CUTIE_METAL_REF names it.
//!
//! - `elementwise`: sigmoid, gru, others, pad, crop, add_broadcast, normalise_rows, rgb_planes,
//!   object_values;
//! - `pool_resize`: maxpool3x3s2, resize_bilinear;
//! - `softmax_norm`: softmax_axis0, aggregate, eca, layer_norm, sine_encoding; the layer norm on
//!   [`crate::layer_norm`]'s kernel, a library of its own, which `gpu_layer_norm_pytorch` runs for
//!   every model;
//! - `matmul_conv`: matmul, similarity, conv2d, conv2d_fused, on `MetalStream::gemm`;
//! - `topk_mask`: topk_softmax, attn_mask.
//!
//! Cutie's linear is `crate::linear::linear_f32_resident`, as CUDA's is `gpu_linear_f32_resident`.
//! Each family's library, kernels, params and checks compile on every OS and are unit-tested
//! anywhere; the ops are macOS only.

use crate::stream::{Kernel, KernelLib};

pub mod elementwise;
pub mod matmul_conv;
pub mod pool_resize;
pub mod softmax_norm;
pub mod topk_mask;

#[cfg(target_os = "macos")]
pub use elementwise::{
    gpu_cutie_add_broadcast, gpu_cutie_crop, gpu_cutie_gru, gpu_cutie_normalise_rows, gpu_cutie_object_values,
    gpu_cutie_others, gpu_cutie_pad, gpu_cutie_rgb_planes, gpu_cutie_sigmoid,
};
#[cfg(target_os = "macos")]
pub use matmul_conv::{gpu_cutie_conv2d, gpu_cutie_conv2d_fused, gpu_cutie_matmul, gpu_cutie_similarity};
#[cfg(target_os = "macos")]
pub use pool_resize::{gpu_cutie_maxpool3x3s2, gpu_cutie_resize_bilinear};
#[cfg(target_os = "macos")]
pub use softmax_norm::{
    gpu_cutie_aggregate, gpu_cutie_eca, gpu_cutie_layer_norm, gpu_cutie_sine_encoding, gpu_cutie_softmax_axis0,
};
pub use softmax_norm::CUTIE_ECA_MAX_TAPS;
#[cfg(target_os = "macos")]
pub use topk_mask::{gpu_cutie_attn_mask, gpu_cutie_topk_softmax};

/// Every library of the kernels Cutie's own ops launch (the five families' and the layer norm's),
/// for the gate that asks whether they all compile (step 14) and the Mac's bring-up test. The GEMM
/// (`stream::gemm::GEMM`, under matmul, similarity, the convolutions and the linear), the generic
/// ops' (`generic_ops::GENERIC_OPS`) and the stream's own (`stream::kernels::TENSOR`) are listed
/// where they are defined.
pub static CUTIE_LIBRARIES: &[&KernelLib] = &[
    &elementwise::CUTIE_ELEMENTWISE,
    &pool_resize::CUTIE_POOL_RESIZE,
    &softmax_norm::CUTIE_SOFTMAX_NORM,
    &crate::layer_norm::LAYER_NORM,
    &matmul_conv::CUTIE_MATMUL_CONV,
    &topk_mask::CUTIE_TOPK_MASK,
];

/// Every Cutie kernel, family by family, for the Mac's test that builds every pipeline.
pub fn cutie_kernels() -> impl Iterator<Item = &'static Kernel> {
    [
        elementwise::ELEMENTWISE_KERNELS,
        pool_resize::POOL_RESIZE_KERNELS,
        softmax_norm::SOFTMAX_NORM_KERNELS,
        crate::layer_norm::LAYER_NORM_KERNELS,
        matmul_conv::MATMUL_CONV_KERNELS,
        topk_mask::TOPK_MASK_KERNELS,
    ]
    .into_iter()
    .flatten()
    .copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_kernel_is_in_its_familys_library_and_every_library_is_listed() {
        let mut names = std::collections::HashSet::new();
        for kernel in cutie_kernels() {
            assert!(
                CUTIE_LIBRARIES.iter().any(|lib| std::ptr::eq(*lib, kernel.lib)),
                "{} is in a library not listed",
                kernel.function
            );
            assert!(names.insert((kernel.lib.name, kernel.function)), "{} is listed twice", kernel.function);
        }
        for lib in CUTIE_LIBRARIES {
            assert!(cutie_kernels().any(|kernel| std::ptr::eq(kernel.lib, *lib)), "{} has no kernel", lib.name);
        }
        let mut libs: Vec<&str> = CUTIE_LIBRARIES.iter().map(|lib| lib.name).collect();
        libs.sort_unstable();
        libs.dedup();
        assert_eq!(libs.len(), CUTIE_LIBRARIES.len(), "library names are the stream's cache keys");
    }
}
