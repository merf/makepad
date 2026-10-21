//! Stage `ops`: each Cutie device op (cuda/src/cutie_ops.rs) against its host reference, on
//! random inputs at Cutie's shapes (several objects, odd sizes, the padded oracle sizes) and the
//! edge cases each op has (k > rows for top-k, ties, a mask row blocked everywhere, padding at
//! every border for max-pool and the strided convs).

use crate::model::not_ported;
use crate::oracle::Report;
use crate::Result;

pub(crate) fn check(_report: &mut Report) -> Result<()> {
    Err(not_ported("the ops check"))
}
