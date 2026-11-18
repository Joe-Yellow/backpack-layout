//! Exact, provably optimal 2D backpack layout solver.
//!
//! `solve_json` is the whole public surface: problem JSON in, solution JSON out. It is shared by the
//! native benchmark binary and the WebAssembly build (see the `wasm` module).

pub mod exact;
pub mod geometry;
pub mod problem;
pub mod sweep;
pub mod verifier;

use serde::Serialize;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PlacedItem {
    pub index: usize,
    pub name: String,
    pub r#type: Option<String>,
    pub value: i32,
    pub final_value: i64,
    pub cells: Vec<[usize; 2]>,
}

#[derive(Serialize)]
pub struct PlacedPiece {
    pub cells: Vec<[usize; 2]>,
    pub value: i64,
}

#[derive(Serialize)]
pub struct SolutionOut {
    pub total: i64,
    pub rounds: u32,
    pub nodes: u64,
    pub items: Vec<PlacedItem>,
    pub pieces: Vec<PlacedPiece>,
}

/// Solves a problem given as JSON; reports (round, pos, cells) after every DP layer.
pub fn solve_json(
    json: &str,
    on_progress: &mut dyn FnMut(u32, usize, usize),
) -> Result<SolutionOut, String> {
    let p = problem::parse(json)?;
    let sol = exact::solve(&p, on_progress)?;
    let score = verifier::evaluate(&p, &sol.layout)?;
    if score.total != sol.value {
        return Err(format!(
            "internal error: solver value {} != verifier score {}",
            sol.value, score.total
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
                value: it.value,
                final_value: score.item_values[i],
                cells: xy(cells),
            })
        })
        .collect();
    let pieces = sol
        .layout
        .dominoes
        .iter()
        .zip(&score.domino_values)
        .map(|(d, &value)| PlacedPiece {
            cells: xy(d),
            value,
        })
        .collect();
    Ok(SolutionOut {
        total: score.total,
        rounds: sol.rounds,
        nodes: sol.nodes,
        items,
        pieces,
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
