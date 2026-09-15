//! Local-first vinyl catalogue: speak or paste, structure, keep a SQLite file.

pub use ::makepad_widgets;

use makepad_widgets::*;

mod extract;
mod model;
mod view;

#[cfg(feature = "native")]
mod db;
#[cfg(feature = "native")]
mod discogs;
#[cfg(feature = "native")]
mod llm;
#[cfg(feature = "native")]
mod weights;

app_main!(App);

script_mod! {
    use mod.prelude.widgets.*
    use mod.widgets.*

    startup() do #(App::script_component(vm)){
        ui: Root{
            main_window := Window{
                window.inner_size: vec2(1200, 820)
                window.title: "Vinyl Catalogue"
                pass.clear_color: vec4(0.047, 0.051, 0.071, 1.0)
                // Caption VoiceWave still receives KeyDown/Permission while the
                // caption bar is invisible — that second pipeline can steal the
                // mic callback (Cmd+1 / F1) from capture_wave. Keep it inert.
                caption_bar +: {
                    voice_wave.visible: false
                }
                body +: {
                    Vinyl{}
                }
            }
        }
    }
}

#[derive(Script, ScriptHook)]
pub struct App {
    #[live]
    ui: WidgetRef,
}

impl MatchEvent for App {}

impl AppMain for App {
    fn script_mod(vm: &mut ScriptVm) -> ScriptValue {
        crate::makepad_widgets::script_mod(vm);
        makepad_wm_theme::apply(vm);
        crate::view::script_mod(vm);
        self::script_mod(vm)
    }

    fn handle_event(&mut self, cx: &mut Cx, event: &Event) {
        self.match_event(cx, event);
        self.ui.handle_event(cx, event, &mut Scope::empty());
    }
}
