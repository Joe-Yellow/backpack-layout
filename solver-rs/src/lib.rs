//! Exact, provably optimal 2D backpack layout solver, with a fast approximate mode.
//!
//! `solve_json` is the whole public surface: problem JSON in, solution JSON out. It is shared by the
//! native benchmark binary and the WebAssembly build (see the `wasm` module).

pub mod beam;
pub mod bits;
pub mod exact;
pub mod geometry;
pub mod problem;
pub mod reduce;
pub mod sweep;
pub mod verifier;

use exact::Outcome;
use problem::{Goal, Problem, EXACT_MAX_ITEMS, FAST_MAX_ITEMS};

/// Fast mode also runs a wide beam over this many items picked by value per cell (see solve_fast).
/// On 100-600 random items it beat the beam over all of them on 5 of 6 instances.
const FAST_FOCUS_ITEMS: usize = 40;
use serde::Serialize;

/// Attribute arrays are [performance, quality, efficiency].
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PlacedItem {
    pub index: usize,
    pub name: String,
    pub r#type: Option<String>,
    pub base: [i32; 3],
    pub values: [i64; 3],
    pub cells: Vec<[usize; 2]>,
}

#[derive(Serialize)]
pub struct PlacedPiece {
    pub cells: Vec<[usize; 2]>,
    pub values: [i64; 3],
}

/// Why no layout exists.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Infeasible {
    /// The required items alone cannot all be placed.
    pub required_impossible: bool,
    /// For each target: the best total its attribute can reach on its own (with the required
    /// items, ignoring the other targets).
    pub target_best: Vec<TargetBest>,
    /// Fast mode found no layout meeting everything (not a proof that none exists).
    pub not_found: bool,
}

#[derive(Serialize)]
pub struct TargetBest {
    pub attr: usize,
    pub target: i64,
    pub best: i64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SolutionOut {
    pub feasible: bool,
    pub totals: [i64; 3],
    pub empty_cells: i64,
    pub rounds: u32,
    pub nodes: u64,
    pub items: Vec<PlacedItem>,
    pub pieces: Vec<PlacedPiece>,
    pub infeasible: Option<Infeasible>,
    /// Fast mode: a good layout, not proven optimal.
    pub approximate: bool,
    /// Items left after dropping those that can't matter (and, in fast mode, the greedy cut).
    pub items_considered: usize,
}

/// Solves a problem given as JSON; reports (round, pos, cells) after every DP layer.
pub fn solve_json(
    json: &str,
    on_progress: &mut dyn FnMut(u32, usize, usize),
) -> Result<SolutionOut, String> {
    let full = problem::parse(json)?;
    if full.fast {
        return solve_fast(&full, on_progress);
    }
    let (p, index) = reduce::reduce(&full, usize::MAX);
    if !p.fast && p.items.len() > EXACT_MAX_ITEMS {
        return Err(format!(
            "精确模式最多支持 {EXACT_MAX_ITEMS} 件可能用上的物品（去掉不可能用上的之后还有 {} 件），请改用快速模式",
            p.items.len()
        ));
    }
    let mut out = match exact::solve(&p, on_progress)? {
        Outcome::Solved(sol) => solved(&p, sol)?,
        Outcome::Infeasible => explain_infeasible(&p, on_progress)?,
    };
    for it in &mut out.items {
        it.index = index[it.index];
    }
    out.items_considered = p.items.len();
    Ok(out)
}

/// Fast mode. With many items, a beam over all of them is narrow, while one over the best few by
/// value per cell can be wide; each wins on some instances, so both run and the better one counts.
fn solve_fast(
    full: &Problem,
    on_progress: &mut dyn FnMut(u32, usize, usize),
) -> Result<SolutionOut, String> {
    let (all, index) = reduce::reduce(full, FAST_MAX_ITEMS);
    let mut runs = vec![(all, index)];
    if runs[0].0.items.len() > FAST_FOCUS_ITEMS {
        let (focus, sub) = reduce::reduce(&runs[0].0, FAST_FOCUS_ITEMS);
        let index = sub.iter().map(|&i| runs[0].1[i]).collect();
        runs.push((focus, index));
    }
    let mut best: Option<(i64, SolutionOut)> = None;
    for (p, index) in &runs {
        let Some(sol) = beam::solve(p, on_progress) else {
            continue;
        };
        let value = sol.value;
        let mut out = solved(p, sol)?;
        for it in &mut out.items {
            it.index = index[it.index];
        }
        out.items_considered = runs[0].0.items.len();
        if best.as_ref().is_none_or(|(v, _)| value > *v) {
            best = Some((value, out));
        }
    }
    let mut out = best.map_or_else(not_found, |(_, out)| out);
    out.approximate = true;
    Ok(out)
}

fn not_found() -> SolutionOut {
    SolutionOut {
        feasible: false,
        totals: [0; 3],
        empty_cells: 0,
        rounds: 0,
        nodes: 0,
        items: Vec::new(),
        pieces: Vec::new(),
        infeasible: Some(Infeasible {
            required_impossible: false,
            target_best: Vec::new(),
            not_found: true,
        }),
        approximate: true,
        items_considered: 0,
    }
}

fn solved(p: &Problem, sol: exact::Solution) -> Result<SolutionOut, String> {
    let score = verifier::evaluate(p, &sol.layout)?;
    if !score.feasible(p, &sol.layout) || score.objective(p) != sol.value {
        return Err(format!(
            "internal error: solver value {} != verifier score {} (or targets unmet)",
            sol.value,
            score.objective(p)
        ));
    }
    let w = p.width as usize;
    let xy = |cells: &[usize]| cells.iter().map(|&c| [c % w, c / w]).collect::<Vec<_>>();
    let items = sol
        .layout
        .item_cells
        .iter()
        .enumerate()
        .filter_map(|(i, cells)| {
            let cells = cells.as_ref()?;
            let it = &p.items[i];
            Some(PlacedItem {
                index: i,
                name: it.name.clone(),
                r#type: it.r#type.clone(),
                base: it.attrs,
                values: score.item_values[i],
                cells: xy(cells),
            })
        })
        .collect();
    let pieces = sol
        .layout
        .dominoes
        .iter()
        .zip(&score.domino_values)
        .map(|(d, &values)| PlacedPiece {
            cells: xy(d),
            values,
        })
        .collect();
    Ok(SolutionOut {
        feasible: true,
        totals: score.totals,
        empty_cells: score.empty_cells,
        rounds: sol.rounds,
        nodes: sol.nodes,
        items,
        pieces,
        infeasible: None,
        approximate: false,
        items_considered: 0,
    })
}

/// Re-solves without the targets: the required items alone may not fit; otherwise report how far
/// each target's attribute can go on its own.
fn explain_infeasible(
    p: &Problem,
    on_progress: &mut dyn FnMut(u32, usize, usize),
) -> Result<SolutionOut, String> {
    let mut target_best = Vec::new();
    let mut required_impossible = false;
    for &(attr, target) in &p.goal.targets {
        let alone = Problem {
            goal: Goal {
                maximize: Some(attr),
                targets: Vec::new(),
            },
            ..p.clone()
        };
        match exact::solve(&alone, on_progress)? {
            Outcome::Solved(sol) => target_best.push(TargetBest {
                attr,
                target,
                best: sol.value,
            }),
            Outcome::Infeasible => required_impossible = true,
        }
    }
    if p.goal.targets.is_empty() {
        required_impossible = true; // nothing but the required items can make it infeasible
    }
    Ok(SolutionOut {
        feasible: false,
        totals: [0; 3],
        empty_cells: 0,
        rounds: 0,
        nodes: 0,
        items: Vec::new(),
        pieces: Vec::new(),
        infeasible: Some(Infeasible {
            required_impossible,
            target_best: if required_impossible {
                Vec::new()
            } else {
                target_best
            },
            not_found: false,
        }),
        approximate: false,
        items_considered: 0,
    })
}

#[cfg(target_arch = "wasm32")]
pub mod wasm {
    use wasm_bindgen::prelude::*;
    pub use wasm_bindgen_rayon::init_thread_pool;

    /// Solves a problem JSON; calls `progress(round, pos, cells)` after every DP layer.
    /// Must run in a Web Worker (rayon blocks while waiting for its threads).
    #[wasm_bindgen]
    pub fn solve(json: &str, progress: &js_sys::Function) -> Result<String, JsValue> {
        let mut report = |round: u32, pos: usize, cells: usize| {
            let _ = progress.call3(
                &JsValue::NULL,
                &round.into(),
                &(pos as u32).into(),
                &(cells as u32).into(),
            );
        };
        let out = crate::solve_json(json, &mut report).map_err(|e| JsValue::from_str(&e))?;
        serde_json::to_string(&out).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}
