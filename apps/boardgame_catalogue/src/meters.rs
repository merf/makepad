//! Rating ring + compact player-count strip — registered before the main UI script_mod.

use crate::model::{FitAtN, PlayerCount};
use makepad_widgets::*;

script_mod! {
    use mod.prelude.widgets_internal.*
    use mod.widgets.*

    // Live fields on DrawRatingRing become shader instances; only the pixel
    // program belongs in the script_shader block (see finance DrawMeter).
    set_type_default() do #(DrawRatingRing::script_shader(vm)){
        ..mod.draw.DrawQuad
        pixel: fn() {
            // Ring via distance-to-circle + angular mask (HTML mockup style).
            // Avoid arc_round_caps / arc_to for the value: both mis-draw when
            // the sweep is past π, which is most BGG scores.
            let size = self.rect_size
            let center = size * 0.5
            let p = self.pos * size - center
            let half = self.stroke_width * 0.5
            // Match the HTML mockup (r=14 in a 36 box): leave ~2px padding
            // so the stroke isn't clipped by the widget bounds.
            let radius = min(center.x, center.y) - half - 2.0
            let ring = abs(length(p) - radius) - half
            let aa = 0.75
            let cover = 1.0 - smoothstep(-aa, aa, ring)
            // Screen y grows down: atan2(y,x) is 0 at +x and increases
            // clockwise. Start at top (-PI/2) and sweep clockwise.
            let ang = atan2(p.y, p.x)
            let start = -0.5 * PI
            let mut rel = ang - start
            if rel < 0.0 {
                rel = rel + 2.0 * PI
            }
            let frac = clamp(self.rating_frac, 0.0, 1.0)
            let sweep = frac * 2.0 * PI
            // Soft cut at the end of the arc (~1px of circumference).
            let ang_aa = aa / max(radius, 1.0)
            let on_fill = 1.0 - smoothstep(sweep - ang_aa, sweep + ang_aa, rel)
            // Round caps: discs at start and end of the value arc.
            let mut cap = 0.0
            if frac > 0.001 {
                let a0 = start
                let a1 = start + sweep
                let c0 = vec2(cos(a0), sin(a0)) * radius
                let c1 = vec2(cos(a1), sin(a1)) * radius
                let d0 = length(p - c0) - half
                let d1 = length(p - c1) - half
                cap = max(
                    1.0 - smoothstep(-aa, aa, d0),
                    1.0 - smoothstep(-aa, aa, d1)
                )
            }
            let fill_a = max(on_fill, cap)
            let col = self.color_track.mix(self.color_fill, fill_a)
            return vec4(col.xyz * col.w * cover, col.w * cover)
        }
    }

    mod.widgets.RatingMeterBase = #(RatingMeter::register_widget(vm))
    mod.widgets.RatingMeter = set_type_default() do mod.widgets.RatingMeterBase{
        width: 36
        height: 36
        draw_ring +: {
            color_fill: #xc45c26
            color_track: #x2a2f3a
            stroke_width: 3.0
        }
        draw_text +: {
            color: #xf2f4f8
            text_style: theme.font_bold{font_size: 10}
        }
    }

    mod.widgets.PlayerCountStripBase = #(PlayerCountStrip::register_widget(vm))
    mod.widgets.PlayerCountStrip = set_type_default() do mod.widgets.PlayerCountStripBase{
        width: Fill
        height: 36
        color_bg: #00000000
        color_best: #x3d9a5f
        color_rec: #x7a9a4a
        color_not: #x5a3a3a
        color_empty: #x252833
        color_track: #x1c1f28
        color_label: #x9aa3b2
        color_label_hi: #xf2f4f8
        draw_text +: {
            color: #x9aa3b2
            text_style: theme.font_regular{font_size: 9}
        }
    }
}

#[derive(Script, ScriptHook)]
#[repr(C)]
pub struct DrawRatingRing {
    #[deref]
    draw_super: DrawQuad,
    #[live]
    rating_frac: f32,
    #[live]
    color_fill: Vec4f,
    #[live]
    color_track: Vec4f,
    #[live(3.0)]
    stroke_width: f32,
}

#[derive(Script, ScriptHook, Widget)]
pub struct RatingMeter {
    #[uid]
    uid: WidgetUid,
    #[source]
    source: ScriptObjectRef,
    #[walk]
    walk: Walk,
    #[redraw]
    #[live]
    draw_ring: DrawRatingRing,
    #[live]
    draw_text: DrawText,
    #[rust]
    rating: f64,
}

impl RatingMeter {
    pub fn set_rating(&mut self, rating: f64) {
        self.rating = rating.clamp(0.0, 10.0);
        self.draw_ring.rating_frac = (self.rating / 10.0) as f32;
    }
}

impl Widget for RatingMeter {
    fn handle_event(&mut self, _cx: &mut Cx, _event: &Event, _scope: &mut Scope) {}

    fn draw_walk(&mut self, cx: &mut Cx2d, _scope: &mut Scope, walk: Walk) -> DrawStep {
        let rect = cx.walk_turtle(walk);
        self.draw_ring.rating_frac = (self.rating / 10.0) as f32;
        self.draw_ring.draw_abs(cx, rect);

        let label = format!("{:.1}", self.rating);
        let (tw, th) = if let Some(run) = self.draw_text.prepare_single_line_run(cx, &label) {
            (
                run.width_in_lpxs as f64,
                (run.ascender_in_lpxs - run.descender_in_lpxs) as f64,
            )
        } else {
            (label.len() as f64 * 5.5, 11.0)
        };
        // draw_abs pos is the TOP of the ink box (first row origin.y is the
        // ascender), not the baseline — center that box in the ring.
        let lx = rect.pos.x + (rect.size.x - tw) * 0.5;
        let ly = rect.pos.y + (rect.size.y - th) * 0.5;
        self.draw_text.draw_abs(cx, dvec2(lx, ly), &label);
        DrawStep::done()
    }
}

#[derive(Script, ScriptHook, Widget)]
pub struct PlayerCountStrip {
    #[uid]
    uid: WidgetUid,
    #[source]
    source: ScriptObjectRef,
    #[walk]
    walk: Walk,
    #[redraw]
    #[live]
    draw_quad: DrawColor,
    #[live]
    draw_text: DrawText,
    #[live]
    color_bg: Vec4f,
    #[live]
    color_best: Vec4f,
    #[live]
    color_rec: Vec4f,
    #[live]
    color_not: Vec4f,
    #[live]
    color_empty: Vec4f,
    #[live]
    color_track: Vec4f,
    #[live]
    color_label: Vec4f,
    #[live]
    color_label_hi: Vec4f,
    #[rust]
    counts: Vec<PlayerCount>,
    #[rust]
    highlight_n: i32,
    #[rust]
    min_players: i32,
    #[rust]
    max_players: i32,
}

impl PlayerCountStrip {
    pub fn set_data(
        &mut self,
        counts: &[PlayerCount],
        highlight_n: i32,
        min_players: i32,
        max_players: i32,
    ) {
        self.highlight_n = highlight_n;
        self.min_players = min_players;
        self.max_players = max_players;
        // Compact: only counts that touch the printed player range.
        self.counts = counts
            .iter()
            .filter(|c| {
                if min_players == 0 || max_players == 0 {
                    return true;
                }
                if c.plus {
                    c.player_n <= max_players
                } else {
                    c.player_n >= min_players && c.player_n <= max_players
                }
            })
            .cloned()
            .collect();
        if self.counts.is_empty() {
            self.counts = counts.to_vec();
        }
    }

    fn is_legal(&self, count: &PlayerCount) -> bool {
        if self.min_players == 0 || self.max_players == 0 {
            true
        } else if count.plus {
            self.max_players >= count.player_n
        } else {
            count.player_n >= self.min_players && count.player_n <= self.max_players
        }
    }

    fn is_highlight(&self, count: &PlayerCount, has_exact: bool) -> bool {
        (!count.plus && count.player_n == self.highlight_n)
            || (count.plus && count.player_n <= self.highlight_n && !has_exact)
    }
}

impl Widget for PlayerCountStrip {
    fn handle_event(&mut self, _cx: &mut Cx, _event: &Event, _scope: &mut Scope) {}

    fn draw_walk(&mut self, cx: &mut Cx2d, _scope: &mut Scope, mut walk: Walk) -> DrawStep {
        // Ensure we actually receive the row height — a Fit walk was collapsing
        // to the bar-only ink box (~19px) and the digit labels drew below the
        // clip of the card.
        if !matches!(walk.height, Size::Fill { .. } | Size::Fixed(_)) {
            walk.height = Size::Fixed(36.0);
        }
        let rect = cx.walk_turtle(walk);
        self.draw_quad.color = self.color_bg;
        self.draw_quad.draw_abs(cx, rect);
        if self.counts.is_empty() {
            return DrawStep::done();
        }

        let n = self.counts.len() as f64;
        let gap = 3.0;
        // draw_abs y is text top; keep the band tight — glyph AA can spill
        // a pixel into card padding via unbounded draw_clip below.
        let label_band = 11.0;
        let label_gap = 2.0;
        let cell_w = ((rect.size.x - gap * (n - 1.0)) / n).min(16.0).max(10.0);
        let used_w = cell_w * n + gap * (n - 1.0);
        let origin_x = rect.pos.x + (rect.size.x - used_w).max(0.0);
        let track_h = (rect.size.y - label_band - label_gap).max(10.0);
        let track_y = rect.pos.y;
        let label_top = rect.pos.y + rect.size.y - label_band;
        let has_exact = self
            .counts
            .iter()
            .any(|c| !c.plus && c.player_n == self.highlight_n);

        for (i, count) in self.counts.iter().enumerate() {
            let x = origin_x + i as f64 * (cell_w + gap);
            let hi = self.is_highlight(count, has_exact);
            let legal = self.is_legal(count);
            let score = count.recommend_score();
            let fit = count.fit_level();

            let track = Rect {
                pos: dvec2(x, track_y),
                size: dvec2(cell_w, track_h),
            };
            self.draw_quad.color = self.color_track;
            self.draw_quad.draw_abs(cx, track);

            let fill_h = if fit == FitAtN::Empty {
                0.0
            } else {
                let t = match fit {
                    FitAtN::Best => score.max(0.55),
                    FitAtN::Recommended => score.max(0.35).min(0.75),
                    FitAtN::NotRecommended => score.min(0.28).max(0.12),
                    FitAtN::Empty => 0.0,
                };
                track_h * t
            };
            if fill_h > 0.5 {
                let mut color = match fit {
                    FitAtN::Best => self.color_best,
                    FitAtN::Recommended => self.color_rec,
                    FitAtN::NotRecommended => self.color_not,
                    FitAtN::Empty => self.color_empty,
                };
                if !legal {
                    color.w *= 0.35;
                }
                self.draw_quad.color = color;
                self.draw_quad.draw_abs(
                    cx,
                    Rect {
                        pos: dvec2(track.pos.x, track.pos.y + track.size.y - fill_h),
                        size: dvec2(track.size.x, fill_h),
                    },
                );
            }

            self.draw_text.color = if hi {
                self.color_label_hi
            } else {
                self.color_label
            };
            let label = count.numplayers.as_str();
            let tw = self
                .draw_text
                .prepare_single_line_run(cx, label)
                .map(|r| r.width_in_lpxs as f64)
                .unwrap_or(label.len() as f64 * 5.5);
            let lx = x + (cell_w - tw) * 0.5;
            // Don't scissor glyph AA against the strip's tight walk clip.
            self.draw_text.draw_clip = vec4(-1.0e6, -1.0e6, 1.0e6, 1.0e6);
            self.draw_text.draw_abs(cx, dvec2(lx, label_top), label);
        }
        DrawStep::done()
    }
}
