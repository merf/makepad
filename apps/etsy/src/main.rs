//! Etsy greeting-card market research: paste bookmarklet JSON, store locally.

pub use ::makepad_widgets;

use makepad_widgets::*;

mod analysis;
mod ask_tools;
mod db;
mod llm;
mod model;
mod normalise;
mod parse;
mod shop_reviews;
mod view;

app_main!(App);

script_mod! {
    use mod.prelude.widgets.*
    use mod.widgets.*

    startup() do #(App::script_component(vm)){
        ui: Root{
            main_window := Window{
                window.inner_size: vec2(1200, 820)
                window.title: "Etsy Market Research"
                pass.clear_color: vec4(0.047, 0.051, 0.071, 1.0)
                body +: {
                    Etsy{}
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
