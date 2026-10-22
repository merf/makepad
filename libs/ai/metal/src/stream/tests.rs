//! The stream's platform-free parts, on any OS: grids, function-constant keys, library sources
//! and the override folder, the launch checks, the host binding's mirror and the weight map. The
//! stream itself is checked on the Mac (`tests/stream.rs`, `stream_fault.rs`, `stream_probe.rs`).

use super::grid::check_grid;
use super::kernel::consts_match;
use super::kernels::*;
use super::mirror::Lru;
use super::plan::{check_slots, SlotKind, SlotPlan};
use super::source::{library_source, splice_header, COMMON_HEADER};
use super::weight_map::WeightMap;
use super::*;

#[test]
fn linear_grid_rounds_up_and_zero_is_empty() {
    let grid = Grid::linear(1000, 256).unwrap();
    assert_eq!(grid.groups, [4, 1, 1]);
    assert_eq!(grid.threads, [256, 1, 1]);
    assert_eq!(grid.threadgroup_bytes, [0; 4]);
    assert!(!grid.is_empty());
    assert_eq!(Grid::linear(1024, 256).unwrap().groups, [4, 1, 1]);
    assert_eq!(Grid::linear(1025, 256).unwrap().groups, [5, 1, 1]);
    assert_eq!(Grid::linear(1, 256).unwrap().groups, [1, 1, 1]);
    let empty = Grid::linear(0, 256).unwrap();
    assert!(empty.is_empty());
    assert_eq!(empty.groups, [0, 1, 1]);
}

#[test]
fn grid_refuses_empty_threadgroups_and_too_many_groups() {
    assert!(Grid::linear(10, 0).is_err());
    assert!(Grid::cover([4, 4, 1], [8, 0, 1]).is_err());
    assert!(Grid::groups([1, 1, 1], [0, 1, 1]).is_err());
    let too_many = (u32::MAX as usize) + 1;
    assert!(Grid::groups([too_many, 1, 1], [1, 1, 1]).is_err());
    assert!(Grid::linear(too_many * 2, 1).is_err());
    assert!(Grid::groups([u32::MAX as usize, 1, 1], [1, 1, 1]).is_ok());
}

#[test]
fn cover_covers_each_axis_and_any_zero_side_is_empty() {
    let grid = Grid::cover([100, 7, 3], [16, 4, 1]).unwrap();
    assert_eq!(grid.groups, [7, 2, 3]);
    assert_eq!(grid.threads, [16, 4, 1]);
    assert!(Grid::cover([100, 0, 1], [16, 4, 1]).unwrap().is_empty());
    assert!(Grid::cover([0, 7, 1], [16, 4, 1]).unwrap().is_empty());
    let exact = Grid::groups([3, 2, 1], [32, 4, 1]).unwrap();
    assert_eq!(exact.groups, [3, 2, 1]);
}

#[test]
fn threadgroup_memory_rounds_up_to_16() {
    let grid = Grid::linear(64, 64).unwrap().threadgroup(0, 1).threadgroup(3, 32).threadgroup(1, 0);
    assert_eq!(grid.threadgroup_bytes, [16, 0, 0, 32]);
    assert_eq!(Grid::linear(64, 64).unwrap().threadgroup(2, 17).threadgroup_bytes, [0, 0, 32, 0]);
    assert_eq!(grid.threadgroup_total(), 48);
}

#[test]
#[should_panic(expected = "not below 4")]
fn threadgroup_memory_index_is_below_4() {
    let _ = Grid::linear(64, 64).unwrap().threadgroup(4, 16);
}

#[test]
fn grid_checks_against_the_pipeline_and_device() {
    let limits = KernelLimits { max_threads: 1024, simd_width: 32 };
    let fits = Grid::cover([64, 64, 1], [32, 32, 1]).unwrap();
    assert!(check_grid("op", "k", &fits, limits, 0, 32768).is_ok());
    let wide = Grid::cover([64, 64, 1], [64, 32, 1]).unwrap();
    let err = check_grid("op", "k", &wide, limits, 0, 32768).unwrap_err();
    assert!(err.contains("2048 threads") && err.starts_with("op (k)"), "{err}");
    let tg = Grid::linear(64, 64).unwrap().threadgroup(0, 16384).threadgroup(1, 16384);
    assert!(check_grid("op", "k", &tg, limits, 0, 32768).is_ok());
    assert!(check_grid("op", "k", &tg, limits, 16, 32768).is_err(), "the kernel's static share counts");
    let hand_made = Grid { groups: [1, 1, 1], threads: [0, 1, 1], threadgroup_bytes: [0; 4] };
    assert!(check_grid("op", "k", &hand_made, limits, 0, 32768).is_err());
}

#[test]
fn function_constants_key_by_index_type_and_bits() {
    let key = |c: FnConst| c.key();
    assert_eq!(key(FnConst::F32(1, 0.5)), key(FnConst::F32(1, 0.5)));
    assert_ne!(key(FnConst::F32(1, 0.0)), key(FnConst::F32(1, -0.0)));
    assert_eq!(key(FnConst::F32(1, f32::NAN)), key(FnConst::F32(1, f32::NAN)));
    assert_ne!(key(FnConst::U32(0, 1)), key(FnConst::I32(0, 1)));
    assert_ne!(key(FnConst::U32(0, 1)), key(FnConst::Bool(0, true)));
    assert_ne!(key(FnConst::U32(0, 1)), key(FnConst::U32(1, 1)));
    let keys: Vec<_> = [FnConst::U32(0, 3), FnConst::Bool(1, true)].iter().map(|c| c.key()).collect();
    assert!(consts_match(&[FnConst::U32(0, 3), FnConst::Bool(1, true)], &keys));
    assert!(!consts_match(&[FnConst::Bool(1, true), FnConst::U32(0, 3)], &keys), "order is part of the key");
    assert!(!consts_match(&[FnConst::U32(0, 3)], &keys));
    assert!(consts_match(&[], &[]));
}

#[test]
fn every_kernel_is_in_its_library_and_every_library_includes_the_header() {
    for kernel in ALL_KERNELS {
        let declared = format!("kernel void {}(", kernel.function);
        assert!(
            kernel.lib.source.contains(&declared),
            "{} is not declared in {}.metal",
            kernel.function,
            kernel.lib.name
        );
    }
    for lib in [&TENSOR, &PROBE, &PROBE_CONTRACT] {
        assert!(lib.source.contains("#include \"makepad_common.h\""), "{} does not include the header", lib.name);
        let spliced = library_source(lib, None).unwrap();
        assert!(!spliced.lines().any(|line| line.trim() == "#include \"makepad_common.h\""));
        assert!(spliced.contains("#include <metal_stdlib>"));
        assert!(spliced.contains("MAKEPAD_COMMON_H"));
    }
}

#[test]
fn params_structs_have_their_msl_sizes() {
    assert_eq!(std::mem::size_of::<TensorCount>(), 4);
    assert_eq!(std::mem::size_of::<TensorFill>(), 8);
    assert_eq!(std::mem::size_of::<TensorCopy2d>(), 24);
    assert_eq!(std::mem::size_of::<ProbeBinds>(), 8);
    assert_eq!(std::mem::size_of::<ProbeFault>(), 8);
    assert_eq!(std::mem::size_of::<()>(), 0);
}

#[test]
fn splice_keeps_line_numbers_for_the_compiler() {
    let source = "// a\n#include \"makepad_common.h\"\nkernel void k() {}\n";
    let spliced = splice_header("demo", source, "HEADER 1\nHEADER 2\n");
    assert_eq!(
        spliced,
        "// a\n#line 1 \"makepad_common.h\"\nHEADER 1\nHEADER 2\n#line 3 \"demo.metal\"\nkernel void k() {}\n"
    );
    // A header without a final newline, CRLF sources, and sources without the include.
    let spliced = splice_header("demo", "#include \"makepad_common.h\"\r\nx\r\n", "H");
    assert_eq!(spliced, "#line 1 \"makepad_common.h\"\nH\n#line 2 \"demo.metal\"\nx\n");
    assert_eq!(splice_header("demo", "x\ny\n", "H"), "x\ny\n");
    assert!(COMMON_HEADER.contains("#ifndef MAKEPAD_COMMON_H"));
}

#[test]
fn the_source_folder_overrides_both_source_and_header() {
    let dir = std::env::temp_dir().join(format!("makepad-metal-stream-src-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let lib = KernelLib { name: "override_demo", source: "BUILT IN\n" };
    let missing = library_source(&lib, Some(&dir)).unwrap_err();
    assert!(missing.contains("override_demo.metal"), "{missing}");
    std::fs::write(dir.join("override_demo.metal"), "#include \"makepad_common.h\"\nFROM DISK\n").unwrap();
    let no_header = library_source(&lib, Some(&dir)).unwrap_err();
    assert!(no_header.contains("makepad_common.h"), "a missing header is an error too: {no_header}");
    std::fs::write(dir.join("makepad_common.h"), "DISK HEADER\n").unwrap();
    let spliced = library_source(&lib, Some(&dir)).unwrap();
    assert!(spliced.contains("DISK HEADER") && spliced.contains("FROM DISK") && !spliced.contains("BUILT IN"));
    assert_eq!(library_source(&lib, None).unwrap(), "BUILT IN\n");
    std::fs::remove_dir_all(&dir).unwrap();
}

/// A slot plan for tests: the tensor's "address" is a small number.
fn slot(kind: SlotKind, tensor: usize, allocated_here: bool) -> SlotPlan {
    SlotPlan { kind, tensor, allocated_here }
}

#[test]
fn launch_checks_params_and_bytes_sizes() {
    let out = slot(SlotKind::Out, 2, true);
    assert!(check_slots("op", "k", 4096, &[out]).is_ok());
    assert!(check_slots("op", "k", 4097, &[out]).is_err());
    assert!(check_slots("op", "k", 0, &[slot(SlotKind::Bytes(4096), 0, false), out]).is_ok());
    let err = check_slots("op", "k", 0, &[slot(SlotKind::Bytes(4097), 0, false), out]).unwrap_err();
    assert!(err.contains("[[buffer(1)]]"), "{err}");
    let many = vec![slot(SlotKind::Absent, 0, false); MAX_BINDS + 1];
    assert!(check_slots("op", "k", 0, &many).is_err());
    assert!(check_slots("op", "k", 0, &many[..MAX_BINDS]).is_ok());
}

#[test]
fn launch_checks_weights_offsets() {
    let weights = |offset, floats| slot(SlotKind::Weights { offset, floats }, 0, false);
    assert!(check_slots("op", "k", 0, &[weights(10, 10)]).is_ok());
    assert!(check_slots("op", "k", 0, &[weights(0, 0)]).is_ok());
    assert!(check_slots("op", "k", 0, &[weights(11, 10)]).is_err());
}

#[test]
fn launch_writes_only_tensors_the_op_allocated_through_one_slot() {
    let x = slot(SlotKind::In, 1, false);
    let out = slot(SlotKind::Out, 2, true);
    assert!(check_slots("op", "k", 0, &[x, x, out]).is_ok(), "an input may be bound twice");
    let foreign = check_slots("op", "k", 0, &[x, slot(SlotKind::Out, 2, false)]).unwrap_err();
    assert!(foreign.contains("did not allocate") && foreign.contains("[[buffer(2)]]"), "{foreign}");
    assert!(check_slots("op", "k", 0, &[slot(SlotKind::InOut, 3, false)]).is_err());
    let aliased = check_slots("op", "k", 0, &[slot(SlotKind::In, 2, false), out]).unwrap_err();
    assert!(aliased.contains("bound again"), "{aliased}");
    assert!(check_slots("op", "k", 0, &[out, out]).is_err(), "written twice");
    assert!(check_slots("op", "k", 0, &[slot(SlotKind::InOut, 2, true), x]).is_ok());
}

#[test]
fn mirror_keeps_the_most_recently_used_under_its_cap() {
    let mut lru = Lru::new(100);
    assert!(lru.insert(1, "a", 40).is_empty());
    assert!(lru.insert(2, "b", 40).is_empty());
    assert_eq!(lru.get(1), Some(&"a")); // 1 is now more recent than 2
    let dropped = lru.insert(3, "c", 40);
    assert_eq!(dropped, vec!["b"]);
    assert_eq!(lru.get(2), None);
    assert_eq!(lru.bytes(), 80);
    // Replacing a key gives the old value back; the bytes follow.
    assert_eq!(lru.insert(1, "a2", 10), vec!["a"]);
    assert_eq!(lru.bytes(), 50);
    // A value alone over the cap is not kept.
    assert_eq!(lru.insert(4, "huge", 101), vec!["huge"]);
    assert_eq!(lru.get(4), None);
    // Room for a big one: everything older goes, least recently used first.
    lru.get(3);
    let dropped = lru.insert(5, "e", 90);
    assert_eq!(dropped, vec!["a2", "c"]);
    assert_eq!(lru.len(), 1);
    assert_eq!(lru.clear(), vec!["e"]);
    assert_eq!((lru.len(), lru.bytes()), (0, 0));
}

#[test]
fn weight_map_is_size_checked_and_evicts_by_prefix() {
    let mut map = WeightMap::new();
    assert!(map.insert("cutie::enc.conv1::cutie_conv", 1, 100).is_none());
    assert!(map.insert("cutie::enc.ln::cutie_ln", 2, 8).is_none());
    assert!(map.insert("sam3::x::w", 3, 4).is_none());
    assert_eq!(map.get("cutie::enc.conv1::cutie_conv", 100), Some(&1));
    assert_eq!(map.get("cutie::enc.conv1::cutie_conv", 101), None, "another size is a miss");
    assert_eq!(map.insert("cutie::enc.conv1::cutie_conv", 4, 101), Some(1));
    assert_eq!(map.get("cutie::enc.conv1::cutie_conv", 101), Some(&4));
    assert_eq!(map.evict_prefix("cutie::"), 2);
    assert_eq!(map.len(), 1);
    assert_eq!(map.evict_prefix("cutie::"), 0);
    assert_eq!(map.get("sam3::x::w", 4), Some(&3));
}
