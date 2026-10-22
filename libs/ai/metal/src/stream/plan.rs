//! The checks a launch makes on its params and bindings before it compiles or encodes anything,
//! over a plain description of each binding so they run on any OS. A failure is an `Err` with
//! nothing encoded (cutie-metal.md §8, rules for kernel authors).

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

/// A binding: its kind, the address of the tensor it binds (0 for none), and whether this op
/// allocated that tensor.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SlotPlan {
    pub kind: SlotKind,
    pub tensor: usize,
    pub allocated_here: bool,
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
///   only: no other slot binds it, as input or output.
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
                let shared = slots
                    .iter()
                    .enumerate()
                    .any(|(other, s)| other != index && s.tensor == slot.tensor && s.tensor != 0);
                if shared {
                    return Err(format!(
                        "{op} ({function}): the tensor written at [[buffer({at})]] is bound again; \
                         read and write it through one InOut"
                    ));
                }
            }
            _ => {}
        }
    }
    Ok(())
}
