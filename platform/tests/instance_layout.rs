//! The instance layout law the Script derive checks for `#[repr(C)]` draw
//! types: a type's own `#[live]` fields after `#[deref]` start exactly where
//! the base's instance fields end. The misaligned case is the
//! `compile_fail` doctest on `DrawVars` (an odd base without a pad lane).

use makepad_platform::*;

/// One instance lane after `DrawVars`: odd, so the struct gets 4 bytes of
/// tail padding, and the pad lane fills it.
#[derive(Script, ScriptHook)]
#[repr(C)]
pub struct DrawOddPadded {
    #[deref]
    pub draw_vars: DrawVars,
    #[live]
    pub a: f32,
    #[live]
    pub pad: f32,
}

#[derive(Script, ScriptHook)]
#[repr(C)]
pub struct DrawOddPaddedExt {
    #[deref]
    pub draw_super: DrawOddPadded,
    #[live]
    pub b: f32,
}

/// The same base without the pad: fine on its own, but extending it would
/// put the first field 4 bytes past the base's instance end.
#[derive(Script, ScriptHook)]
#[repr(C)]
pub struct DrawOdd {
    #[deref]
    pub draw_vars: DrawVars,
    #[live]
    pub a: f32,
}

#[test]
fn draw_vars_instance_end_is_its_size() {
    assert_eq!(DrawVars::SCRIPT_INSTANCE_END, std::mem::size_of::<DrawVars>());
}

#[test]
fn odd_base_has_tail_padding() {
    let end = DrawOdd::SCRIPT_INSTANCE_END;
    assert_eq!(end, std::mem::size_of::<DrawVars>() + 4);
    assert_eq!(std::mem::size_of::<DrawOdd>(), end + 4, "the padding the check guards against");
}

#[test]
fn padded_extension_is_contiguous() {
    assert_eq!(DrawOddPadded::SCRIPT_INSTANCE_END, std::mem::size_of::<DrawOddPadded>());
    assert_eq!(
        std::mem::offset_of!(DrawOddPaddedExt, b),
        DrawOddPadded::SCRIPT_INSTANCE_END
    );
    assert_eq!(
        DrawOddPaddedExt::SCRIPT_INSTANCE_END,
        DrawOddPadded::SCRIPT_INSTANCE_END + 4
    );
}
