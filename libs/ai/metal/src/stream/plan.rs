//! The checks a launch makes on its params and bindings before it encodes anything, over a plain
//! description of each binding and of what the kernel reads, so they run on any OS. A failure is
//! an `Err` with nothing encoded (cutie-metal.md §8, rules for kernel authors).

use super::MetalResult;

/// The most bytes `setBytes` takes: the params at index 0 and every `Bind::Bytes`.
pub const MAX_SET_BYTES: usize = 4096;

/// The most buffer bindings after the params: Metal gives a compute function 31 buffer slots, and
/// slot 0 is the params.
pub const MAX_BINDS: usize = 30;

/// One binding as the checks see it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum SlotKind {
    In,
    Out,
    InOut,
    /// Weights read from `offset` of `floats` values.
    Weights { offset: usize, floats: usize },
    /// `setBytes` of this many bytes.
    Bytes(usize),
    Absent,
}

/// A binding: its kind, the address of the tensor it binds (0 for none), the address of that
/// tensor's storage (0 for none) and the words it covers there, `[start, end)`, and whether this op
/// allocated the tensor.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SlotPlan {
    pub kind: SlotKind,
    pub tensor: usize,
    pub storage: usize,
    pub words: (usize, usize),
    pub allocated_here: bool,
}

impl SlotPlan {
    /// Whether `self` and `other` bind memory in common: the same tensor, or tensors on one
    /// storage whose words overlap (a view of the other, or of what the other is a view of).
    fn overlaps(&self, other: &SlotPlan) -> bool {
        if self.tensor != 0 && self.tensor == other.tensor {
            return true;
        }
        let ((a0, a1), (b0, b1)) = (self.words, other.words);
        self.storage != 0 && self.storage == other.storage && a0 < a1 && b0 < b1 && a0 < b1 && b0 < a1
    }
}

impl SlotKind {
    fn writes(self) -> bool {
        matches!(self, SlotKind::Out | SlotKind::InOut)
    }
}

/// Checks a launch of `function` in op `op` with `params_bytes` of params and `slots`:
/// - params and every `Bytes` at most [`MAX_SET_BYTES`], at most [`MAX_BINDS`] bindings;
/// - `Weights` read from an offset inside them;
/// - every `Out` and `InOut` a tensor this op allocated (rule 2), and written through one slot
///   only: no other slot binds it, as input or output, nor a view overlapping it (a dispatch's
///   reads and writes of the same memory are not ordered).
pub(crate) fn check_slots(op: &str, function: &str, params_bytes: usize, slots: &[SlotPlan]) -> MetalResult<()> {
    if params_bytes > MAX_SET_BYTES {
        return Err(format!("{op} ({function}): {params_bytes} bytes of params, setBytes takes at most {MAX_SET_BYTES}"));
    }
    if slots.len() > MAX_BINDS {
        return Err(format!("{op} ({function}): {} bindings, a kernel takes at most {MAX_BINDS}", slots.len()));
    }
    for (index, slot) in slots.iter().enumerate() {
        let at = index + 1;
        match slot.kind {
            SlotKind::Bytes(len) if len > MAX_SET_BYTES => {
                return Err(format!(
                    "{op} ({function}): {len} bytes at [[buffer({at})]], setBytes takes at most {MAX_SET_BYTES}"
                ));
            }
            SlotKind::Weights { offset, floats } if offset > floats => {
                return Err(format!(
                    "{op} ({function}): weights at [[buffer({at})]] read from float {offset} of {floats}"
                ));
            }
            kind if kind.writes() => {
                if !slot.allocated_here {
                    return Err(format!(
                        "{op} ({function}): [[buffer({at})]] writes a tensor this op did not allocate \
                         (GpuTensor::alloc inside the op)"
                    ));
                }
                let shared = slots.iter().enumerate().any(|(other, s)| other != index && s.overlaps(slot));
                if shared {
                    return Err(format!(
                        "{op} ({function}): the tensor written at [[buffer({at})]] is bound again, \
                         or a view overlapping it is; read and write it through one InOut"
                    ));
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// How a pipeline's reflection lists one of the kernel's arguments.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ArgKind {
    /// `[[buffer(index)]]`.
    Buffer,
    /// `[[threadgroup(index)]]`.
    Threadgroup,
    /// Textures, samplers and the rest: the stream binds none.
    Other,
}

/// What a kernel reads, from its pipeline's reflection: bit `i` of `buffers` for each
/// `[[buffer(i)]]` it uses, the size of the params struct it reads at `[[buffer(0)]]` (0 when
/// unknown), and bit `i` of `threadgroups` for each `[[threadgroup(i)]]` it uses.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct KernelArgs {
    pub buffers: u64,
    pub params_bytes: usize,
    pub threadgroups: u32,
}

impl KernelArgs {
    /// Adds one argument from the reflection: its kind, its index, whether the kernel uses it, and
    /// for a buffer the size of the type it points at. Arguments the kernel does not use are left
    /// out (it never reads them). An index past what a launch can bind is kept as the highest bit,
    /// so the launch check refuses it.
    pub(crate) fn add(&mut self, kind: ArgKind, index: u64, used: bool, data_bytes: u64) {
        if !used {
            return;
        }
        match kind {
            ArgKind::Buffer => {
                self.buffers |= 1 << index.min(63);
                if index == 0 {
                    self.params_bytes = data_bytes as usize;
                }
            }
            ArgKind::Threadgroup => self.threadgroups |= 1 << index.min(31),
            ArgKind::Other => {}
        }
    }
}

/// Checks a launch of `function` gives it everything `args` says it reads: params at
/// `[[buffer(0)]]` at least as big as its struct, a binding at every other buffer index it uses
/// (`binds` bindings cover 1 to `binds`), and threadgroup memory at every threadgroup index it
/// uses. A compute encoder keeps its bindings, params and threadgroup lengths from one dispatch to
/// the next, so once dispatches share an encoder (the resident binding) whatever a launch left out
/// would be read from the dispatch before it; refusing it here makes both bindings fail alike.
pub(crate) fn check_args(
    op: &str,
    function: &str,
    args: &KernelArgs,
    params_bytes: usize,
    binds: usize,
    threadgroup_bytes: &[u32; 4],
) -> MetalResult<()> {
    if args.buffers & 1 != 0 {
        if params_bytes == 0 {
            return Err(format!(
                "{op} ({function}): the kernel reads params at [[buffer(0)]] and the launch passes none"
            ));
        }
        if params_bytes < args.params_bytes {
            return Err(format!(
                "{op} ({function}): {params_bytes} bytes of params, the kernel's struct at [[buffer(0)]] \
                 is {} bytes",
                args.params_bytes
            ));
        }
    }
    if let Some(index) = (1..64).find(|&index| args.buffers & (1 << index) != 0 && index > binds) {
        return Err(format!(
            "{op} ({function}): the kernel reads [[buffer({index})]] and the launch binds only {binds} \
             (Bind::Absent fills a slot the kernel does not read)"
        ));
    }
    if let Some(index) = (0..32).find(|&index| {
        args.threadgroups & (1 << index) != 0 && threadgroup_bytes.get(index).is_none_or(|&bytes| bytes == 0)
    }) {
        return Err(format!(
            "{op} ({function}): the kernel uses [[threadgroup({index})]] and the grid gives it no memory \
             (Grid::threadgroup)"
        ));
    }
    Ok(())
}
