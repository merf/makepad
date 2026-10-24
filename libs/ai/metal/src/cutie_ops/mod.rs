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
pub mod graph;
pub mod matmul_conv;
pub mod pool_resize;
pub mod softmax_norm;
pub mod topk_mask;
pub mod transformer_graph;

#[cfg(target_os = "macos")]
pub use elementwise::{
    gpu_cutie_add_broadcast, gpu_cutie_crop, gpu_cutie_gru, gpu_cutie_normalise_rows, gpu_cutie_object_values,
    gpu_cutie_others, gpu_cutie_pad, gpu_cutie_rgb_planes, gpu_cutie_sigmoid,
};
#[cfg(target_os = "macos")]
pub use matmul_conv::{gpu_cutie_conv2d, gpu_cutie_conv2d_fused, gpu_cutie_conv2d_fused_f16, gpu_cutie_matmul, gpu_cutie_similarity};
pub use matmul_conv::{set_conv_mpscnn_override, CUTIE_MPS, MPS_CONV_KERNELS};
pub use graph::{CutieGraphConv, CutieGraphOp};
pub use transformer_graph::{CutieGraphAttention, CutieGraphBlock, CutieGraphLinear, CutieGraphNorm, CutieGraphTransformer};
#[cfg(target_os = "macos")]
pub use graph::gpu_cutie_graph_f16;
#[cfg(target_os = "macos")]
pub use transformer_graph::gpu_cutie_transformer_graph_f16;
#[cfg(target_os = "macos")]
pub use pool_resize::{gpu_cutie_maxpool3x3s2, gpu_cutie_resize_bilinear};
#[cfg(target_os = "macos")]
pub use softmax_norm::{
    gpu_cutie_aggregate, gpu_cutie_eca, gpu_cutie_layer_norm, gpu_cutie_sine_encoding, gpu_cutie_softmax_axis0,
};
pub use softmax_norm::CUTIE_ECA_MAX_TAPS;
#[cfg(target_os = "macos")]
pub use topk_mask::{gpu_cutie_attn_mask, gpu_cutie_topk_softmax};
#[cfg(target_os = "macos")]
pub use topk_mask::gpu_cutie_topk_sparse_readout;

/// Every library of the kernels Cutie's own ops launch (the five families' and the layer norm's),
/// for the Mac's bring-up test. [`libraries_cutie_launches`] adds the libraries they share.
pub static CUTIE_LIBRARIES: &[&KernelLib] = &[
    &elementwise::CUTIE_ELEMENTWISE,
    &pool_resize::CUTIE_POOL_RESIZE,
    &softmax_norm::CUTIE_SOFTMAX_NORM,
    &crate::layer_norm::LAYER_NORM,
    &matmul_conv::CUTIE_MATMUL_CONV,
    &topk_mask::CUTIE_TOPK_MASK,
];

/// Every library Cutie's ops launch: its own ([`CUTIE_LIBRARIES`]), the generic ops' it calls
/// (`generic_ops::GENERIC_OPS`), the GEMM's (`stream::gemm::GEMM`: matmul, similarity, the
/// convolutions, the linear and the cross-attentions) and the stream's fills, copies, adds, muls
/// and ReLU (`stream::kernels::TENSOR`). What [`cutie_ready`] compiles.
pub fn libraries_cutie_launches() -> impl Iterator<Item = &'static KernelLib> {
    CUTIE_LIBRARIES.iter().copied().chain([
        &crate::generic_ops::GENERIC_OPS,
        &crate::stream::gemm::GEMM,
        &crate::stream::kernels::TENSOR,
    ])
}

/// The kernels Cutie's ops launch that refuse SIMD groups of any width but 32 (each launch checks
/// its own; [`cutie_ready`] checks them all first): the layer norm's, top-k's merge and row sum,
/// the ECA's row mean, the tiled GEMM and attention's row softmax.
pub static SIMD_32_KERNELS: &[&Kernel] = &[
    &crate::layer_norm::LAYER_NORM_PYTORCH,
    &topk_mask::CUTIE_TOPK_MERGE,
    &topk_mask::CUTIE_ROW_SUM,
    &softmax_norm::CUTIE_ECA_ROW_MEAN,
    &crate::stream::gemm::GEMM_TILED,
    &crate::stream::gemm::GEMM_SOFTMAX_ROWS,
];

/// The gate (step 14): `Ok` when the calling thread's stream starts, every library Cutie's ops
/// launch compiles there ([`libraries_cutie_launches`]) and the kernels of [`SIMD_32_KERNELS`]
/// get the SIMD groups they are written for. Libraries and pipelines are compiled per thread, now
/// if they have not been, so ask on the thread that runs Cutie. `Err` names the first library or
/// kernel that fails, with Metal's message.
#[cfg(target_os = "macos")]
pub fn cutie_ready() -> Result<(), String> {
    libraries_cutie_launches().try_for_each(crate::stream::library_ready)?;
    crate::stream::with_metal_stream("cutie_ready", |s| {
        SIMD_32_KERNELS.iter().try_for_each(|kernel| crate::stream::gemm::check_simd_width(s, "cutie_ready", kernel))
    })
}

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

    #[test]
    fn the_gate_compiles_every_library_once() {
        let mut libs: Vec<&str> = libraries_cutie_launches().map(|lib| lib.name).collect();
        assert_eq!(libs.len(), CUTIE_LIBRARIES.len() + 3);
        libs.sort_unstable();
        libs.dedup();
        assert_eq!(libs.len(), CUTIE_LIBRARIES.len() + 3, "a library is listed twice");
        for kernel in SIMD_32_KERNELS {
            assert!(
                libraries_cutie_launches().any(|lib| std::ptr::eq(lib, kernel.lib)),
                "{} is in a library the gate does not compile",
                kernel.function
            );
        }
    }
}
