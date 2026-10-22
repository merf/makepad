//! What a kernel is to the stream: its library, its function, the function constants it is
//! specialised with, and the params struct it reads at `[[buffer(0)]]`.

/// A Makepad kernel library: compiled once per thread's stream (fast math off), cached by `name`.
/// MAKEPAD_METAL_KERNELS_SRC=<dir> reads `<dir>/<name>.metal` instead of `source`.
pub struct KernelLib {
    pub name: &'static str,
    pub source: &'static str,
}

impl std::fmt::Debug for KernelLib {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KernelLib").field("name", &self.name).finish_non_exhaustive()
    }
}

/// One kernel function in a library.
#[derive(Clone, Copy, Debug)]
pub struct Kernel {
    pub lib: &'static KernelLib,
    pub function: &'static str,
}

/// [[function_constant(index)]] values; part of the pipeline key.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FnConst {
    Bool(u16, bool),
    I32(u16, i32),
    U32(u16, u32),
    F32(u16, f32),
}

/// A function constant as the pipeline cache compares it: its index, its type and its bits (an
/// `f32` by its bits, so `-0.0` and `0.0` are two pipelines and a NaN finds its own).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct FnConstKey {
    pub index: u16,
    pub kind: u8,
    pub bits: u32,
}

impl FnConst {
    pub(crate) fn key(self) -> FnConstKey {
        let (index, kind, bits) = match self {
            FnConst::Bool(index, value) => (index, 0, value as u32),
            FnConst::I32(index, value) => (index, 1, value as u32),
            FnConst::U32(index, value) => (index, 2, value),
            FnConst::F32(index, value) => (index, 3, value.to_bits()),
        };
        FnConstKey { index, kind, bits }
    }
}

/// Whether `consts` are the constants a cached pipeline was made with, in the same order.
pub(crate) fn consts_match(consts: &[FnConst], keys: &[FnConstKey]) -> bool {
    consts.len() == keys.len() && consts.iter().zip(keys).all(|(c, k)| c.key() == *k)
}

/// The params struct a kernel reads at `[[buffer(0)]]`, passed with `setBytes`.
///
/// # Safety
/// #[repr(C)], at most 4096 bytes, no pointers, field for field its MSL twin.
pub unsafe trait KernelParams: Copy + 'static {}

// SAFETY: zero-sized; a kernel without params binds nothing at index 0.
unsafe impl KernelParams for () {}
