//! Exact, provably optimal 2D backpack layout solver.
//!
//! `solve_json` is the whole public surface: problem JSON in, solution JSON out. It is shared by the
//! native benchmark binary and the WebAssembly build (see the `wasm` module).

pub mod exact;
pub mod geometry;
pub mod problem;
pub mod sweep;
pub mod verifier;

use exact::Outcome;
use problem::{Goal, Problem};
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
}

/// Solves a problem given as JSON; reports (round, pos, cells) after every DP layer.
pub fn solve_json(
    json: &str,
    on_progress: &mut dyn FnMut(u32, usize, usize),
) -> Result<SolutionOut, String> {
    let p = problem::parse(json)?;
    match exact::solve(&p, on_progress)? {
        Outcome::Solved(sol) => solved(&p, sol),
        Outcome::Infeasible => explain_infeasible(&p, on_progress),
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
        }),
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
