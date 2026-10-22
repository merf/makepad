//! A dispatch's shape: threadgroups, threads per threadgroup and threadgroup memory, built and
//! checked on the host before anything is encoded.

use super::MetalResult;

/// What a launch dispatches: `groups` threadgroups of `threads` threads each, and the
/// threadgroup memory at `[[threadgroup(i)]]` (bytes, multiples of 16; 0 for none).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grid {
    pub groups: [u32; 3],
    pub threads: [u32; 3],
    pub threadgroup_bytes: [u32; 4],
}

/// A pipeline's limits: the most threads one threadgroup may have, and the SIMD-group width.
#[derive(Clone, Copy, Debug)]
pub struct KernelLimits {
    pub max_threads: u32,
    pub simd_width: u32,
}

impl Grid {
    /// `n` threads in a line, `threads` to a threadgroup, the last threadgroup partly idle (kernels
    /// check their index). `n == 0` gives an empty grid.
    pub fn linear(n: usize, threads: u32) -> MetalResult<Grid> {
        Grid::cover([n, 1, 1], [threads, 1, 1])
    }

    /// Enough threadgroups of `threads` to cover `size` in each dimension; any zero size gives an
    /// empty grid.
    pub fn cover(size: [usize; 3], threads: [u32; 3]) -> MetalResult<Grid> {
        check_threads(threads)?;
        let mut groups = [0usize; 3];
        for axis in 0..3 {
            groups[axis] = size[axis].div_ceil(threads[axis] as usize);
        }
        Grid::groups(groups, threads)
    }

    /// Exactly `groups` threadgroups of `threads` threads.
    pub fn groups(groups: [usize; 3], threads: [u32; 3]) -> MetalResult<Grid> {
        check_threads(threads)?;
        let mut counts = [0u32; 3];
        for axis in 0..3 {
            counts[axis] = u32::try_from(groups[axis]).map_err(|_| {
                format!("Metal grid: {} threadgroups along axis {axis} is more than a dispatch takes", groups[axis])
            })?;
        }
        Ok(Grid { groups: counts, threads, threadgroup_bytes: [0; 4] })
    }

    /// The same grid with `bytes` of threadgroup memory at `[[threadgroup(index)]]`, rounded up to
    /// a multiple of 16 as Metal requires. `index` is below 4.
    pub fn threadgroup(mut self, index: usize, bytes: u32) -> Grid {
        assert!(index < self.threadgroup_bytes.len(), "Metal grid: threadgroup memory index {index} is not below 4");
        self.threadgroup_bytes[index] = bytes.div_ceil(16).saturating_mul(16);
        self
    }

    /// True when it has no threadgroups: a launch then encodes nothing.
    pub fn is_empty(&self) -> bool {
        self.groups.contains(&0)
    }

    /// Threads in one threadgroup.
    pub(crate) fn threads_per_group(&self) -> u64 {
        self.threads.iter().map(|&t| t as u64).product()
    }

    /// Threadgroup memory the grid asks for, over every index.
    pub(crate) fn threadgroup_total(&self) -> u64 {
        self.threadgroup_bytes.iter().map(|&b| b as u64).sum()
    }
}

fn check_threads(threads: [u32; 3]) -> MetalResult<()> {
    if threads.contains(&0) {
        return Err(format!("Metal grid: a threadgroup of {threads:?} threads has an empty side"));
    }
    Ok(())
}

/// Checks `grid` against the pipeline it dispatches: no empty threadgroup side, at most
/// `max_threads` threads to a threadgroup, and at most `max_threadgroup_bytes` of threadgroup
/// memory with the kernel's own static share.
pub(crate) fn check_grid(
    op: &str,
    function: &str,
    grid: &Grid,
    limits: KernelLimits,
    static_threadgroup_bytes: u32,
    max_threadgroup_bytes: u64,
) -> MetalResult<()> {
    check_threads(grid.threads).map_err(|e| format!("{op} ({function}): {e}"))?;
    if grid.threads_per_group() > limits.max_threads as u64 {
        return Err(format!(
            "{op} ({function}): {} threads to a threadgroup, the pipeline takes at most {}",
            grid.threads_per_group(),
            limits.max_threads
        ));
    }
    let threadgroup = grid.threadgroup_total() + static_threadgroup_bytes as u64;
    if threadgroup > max_threadgroup_bytes {
        return Err(format!(
            "{op} ({function}): {threadgroup} bytes of threadgroup memory, the device has {max_threadgroup_bytes}"
        ));
    }
    Ok(())
}
