//! The stream's platform-free parts, on any OS: grids, function-constant keys, library sources
//! and the override folder, the launch checks and the weight map. The stream itself is checked on
//! the Mac (`tests/stream.rs`, `residency.rs`, `stream_fault.rs`, `stream_probe.rs`).

use super::grid::check_grid;
use super::kernel::consts_match;
use super::kernels::*;
use super::plan::{check_args, check_slots, ArgKind, KernelArgs, SlotKind, SlotPlan};
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
        assert!(
            !spliced.lines().any(|line| line.trim_start().starts_with("#include") && line.contains("makepad_common.h")),
            "{} still includes the header after splicing",
            lib.name
        );
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
    assert_eq!(std::mem::size_of::<ProbeSpin>(), 8);
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
    // A comment after the include, as cutie-metal.md's example writes it, and indentation.
    let commented = "x\n  #include \"makepad_common.h\"   // contraction, cutie_reduce4\ny\n";
    assert_eq!(
        splice_header("demo", commented, "H\n"),
        "x\n#line 1 \"makepad_common.h\"\nH\n#line 3 \"demo.metal\"\ny\n"
    );
    // Not the header: left alone.
    for other in ["#include \"makepad_common.h.bak\"", "#include \"makepad_common.h\" x", "#include <metal_stdlib>"] {
        assert_eq!(splice_header("demo", other, "H"), format!("{other}\n"));
    }
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

/// What `tensor_add_f32` reads, as its reflection lists it: params of 4 bytes, three buffers.
fn add_args() -> KernelArgs {
    let mut args = KernelArgs::default();
    args.add(ArgKind::Buffer, 0, true, 4);
    for index in 1..=3 {
        args.add(ArgKind::Buffer, index, true, 4);
    }
    args
}

#[test]
fn the_unwritten_word_is_a_quiet_nan_that_arithmetic_keeps() {
    let unwritten = f32::from_bits(UNWRITTEN_BITS);
    assert!(unwritten.is_nan());
    assert_ne!(UNWRITTEN_BITS & 0x0040_0000, 0, "quiet, so no FP exception and no change on a copy");
    assert!((unwritten * 0.0 + 1.0).is_nan(), "accumulating onto it stays NaN");
}

#[test]
fn kernel_args_keep_what_the_kernel_uses() {
    assert_eq!(add_args(), KernelArgs { buffers: 0b1111, params_bytes: 4, threadgroups: 0 });
    let mut args = KernelArgs::default();
    args.add(ArgKind::Buffer, 3, false, 4); // declared, never read
    args.add(ArgKind::Buffer, 2, true, 4);
    args.add(ArgKind::Threadgroup, 1, true, 4);
    args.add(ArgKind::Threadgroup, 2, false, 4);
    args.add(ArgKind::Other, 0, true, 0);
    assert_eq!(args, KernelArgs { buffers: 0b100, params_bytes: 0, threadgroups: 0b10 });
    // Indices past what a launch can bind keep the highest bit, which no launch satisfies.
    let mut far = KernelArgs::default();
    far.add(ArgKind::Buffer, 200, true, 4);
    far.add(ArgKind::Threadgroup, 40, true, 4);
    assert_eq!((far.buffers, far.threadgroups), (1 << 63, 1 << 31));
    assert!(check_args("op", "k", &far, 0, MAX_BINDS, &[16; 4]).is_err());
}

#[test]
fn a_launch_gives_the_kernel_everything_it_reads() {
    let args = add_args();
    assert!(check_args("op", "k", &args, 4, 3, &[0; 4]).is_ok());
    assert!(check_args("op", "k", &args, 8, 5, &[0; 4]).is_ok(), "bigger params and extra bindings are fine");
    let none = check_args("op", "k", &args, 0, 3, &[0; 4]).unwrap_err();
    assert!(none.starts_with("op (k)") && none.contains("[[buffer(0)]]") && none.contains("passes none"), "{none}");
    let mut wide = add_args();
    wide.params_bytes = 24;
    let small = check_args("op", "k", &wide, 8, 3, &[0; 4]).unwrap_err();
    assert!(small.contains("8 bytes of params") && small.contains("is 24 bytes"), "{small}");
    let missing = check_args("op", "k", &args, 4, 2, &[0; 4]).unwrap_err();
    assert!(missing.contains("[[buffer(3)]]") && missing.contains("binds only 2"), "{missing}");
    // A kernel that reads no params needs none; one whose struct size is unknown needs some.
    let no_params = KernelArgs { buffers: 0b10, params_bytes: 0, threadgroups: 0 };
    assert!(check_args("op", "k", &no_params, 0, 1, &[0; 4]).is_ok());
    let unknown = KernelArgs { buffers: 0b11, params_bytes: 0, threadgroups: 0 };
    assert!(check_args("op", "k", &unknown, 4, 1, &[0; 4]).is_ok());
    assert!(check_args("op", "k", &unknown, 0, 1, &[0; 4]).is_err());
}

#[test]
fn a_launch_gives_every_threadgroup_index_its_memory() {
    let args = KernelArgs { buffers: 0b111, params_bytes: 4, threadgroups: 0b1001 };
    let grid = Grid::linear(64, 64).unwrap();
    let err = check_args("op", "k", &args, 4, 2, &grid.threadgroup_bytes).unwrap_err();
    assert!(err.contains("[[threadgroup(0)]]") && err.contains("Grid::threadgroup"), "{err}");
    let half = grid.threadgroup(0, 256);
    let err = check_args("op", "k", &args, 4, 2, &half.threadgroup_bytes).unwrap_err();
    assert!(err.contains("[[threadgroup(3)]]"), "{err}");
    assert!(check_args("op", "k", &args, 4, 2, &half.threadgroup(3, 4).threadgroup_bytes).is_ok());
    // Memory at an index the kernel does not use is harmless.
    let none = KernelArgs { threadgroups: 0, ..args };
    assert!(check_args("op", "k", &none, 4, 2, &half.threadgroup_bytes).is_ok());
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
