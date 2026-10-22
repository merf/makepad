//! The text a kernel library compiles from: its source with the shared header spliced in (Metal
//! compiles from a string at run time, so there is no include path), read from disk instead when
//! MAKEPAD_METAL_KERNELS_SRC names a folder (an edit loop on the Mac with no cargo rebuild).

use super::kernel::KernelLib;
use std::path::{Path, PathBuf};

/// The header every Makepad kernel library includes.
pub const COMMON_HEADER_NAME: &str = "makepad_common.h";

/// The header's text as built into the crate.
pub static COMMON_HEADER: &str = include_str!("../../shaders/makepad/makepad_common.h");

/// The line a source includes the header with.
const COMMON_INCLUDE: &str = "#include \"makepad_common.h\"";

/// The folder MAKEPAD_METAL_KERNELS_SRC names, when it is set and not empty.
pub(crate) fn kernels_src_dir() -> Option<PathBuf> {
    std::env::var_os("MAKEPAD_METAL_KERNELS_SRC")
        .filter(|dir| !dir.is_empty())
        .map(PathBuf::from)
}

/// The text `lib` compiles from: from `dir` (`<dir>/<name>.metal` and `<dir>/makepad_common.h`)
/// when given, otherwise as built in, with the header spliced in. A file missing from `dir` is an
/// error, never a quiet fall back to the built-in text.
pub(crate) fn library_source(lib: &KernelLib, dir: Option<&Path>) -> Result<String, String> {
    match dir {
        Some(dir) => {
            let source = read(&dir.join(format!("{}.metal", lib.name)))?;
            let header = read(&dir.join(COMMON_HEADER_NAME))?;
            Ok(splice_header(lib.name, &source, &header))
        }
        None => Ok(splice_header(lib.name, lib.source, COMMON_HEADER)),
    }
}

fn read(path: &Path) -> Result<String, String> {
    std::fs::read_to_string(path).map_err(|e| {
        format!("MAKEPAD_METAL_KERNELS_SRC: cannot read {}: {e}", path.display())
    })
}

/// `source` with every line that is exactly `#include "makepad_common.h"` replaced by `header`,
/// framed by `#line` directives so the compiler's messages name the right file and line.
pub(crate) fn splice_header(name: &str, source: &str, header: &str) -> String {
    let mut out = String::with_capacity(source.len() + header.len() + 128);
    for (index, line) in source.lines().enumerate() {
        if line.trim() == COMMON_INCLUDE {
            out.push_str("#line 1 \"");
            out.push_str(COMMON_HEADER_NAME);
            out.push_str("\"\n");
            out.push_str(header);
            if !header.ends_with('\n') {
                out.push('\n');
            }
            // The source's next line is line index + 2 (lines count from 1).
            out.push_str(&format!("#line {} \"{name}.metal\"\n", index + 2));
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}
