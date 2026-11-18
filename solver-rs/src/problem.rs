//! Input model, validation and the built-in shapes. Coordinates are [x, y], origin top-left.

use serde::Deserialize;
use std::collections::HashSet;

pub type Cell = [i32; 2];

pub const MAX_ITEMS: usize = 63;
pub const MAX_SIDE: i32 = 12;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Border {
    Top,
    Bottom,
    Left,
    Right,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ItemInput {
    pub name: String,
    pub value: i64,
    #[serde(default)]
    pub r#type: Option<String>,
    #[serde(default)]
    pub shape: Option<Vec<Cell>>,
    #[serde(default)]
    pub border: Option<Border>,
    #[serde(default)]
    pub adj_bonus: bool,
}

#[derive(Debug, Deserialize)]
pub struct ProblemInput {
    pub width: i32,
    pub height: i32,
    #[serde(default)]
    pub blocked: Vec<Cell>,
    pub items: Vec<ItemInput>,
}

#[derive(Clone, Debug)]
pub struct Item {
    pub name: String,
    pub value: i32,
    pub r#type: Option<String>,
    pub shape: Vec<Cell>,
    pub border: Option<Border>,
    pub adj_bonus: bool,
}

#[derive(Clone, Debug)]
pub struct Problem {
    pub width: i32,
    pub height: i32,
    pub blocked: Vec<Cell>,
    pub items: Vec<Item>,
}

/// Built-in shapes, usable via `type` (rotations and mirrors are always allowed).
pub const SHAPES: [(&str, &[Cell]); 6] = [
    ("U", &[[0, 0], [1, 0], [2, 0], [0, 1], [2, 1]]),
    ("F", &[[0, 0], [1, 0], [2, 0], [1, 1], [2, 1]]),
    ("L", &[[0, 0], [1, 0], [2, 0], [0, 1]]),
    ("T", &[[0, 0], [1, 0], [2, 0], [1, 1]]),
    ("O", &[[0, 0], [1, 0], [0, 1], [1, 1]]),
    ("V", &[[0, 0], [1, 0], [0, 1]]),
];

fn require(ok: bool, msg: impl FnOnce() -> String) -> Result<(), String> {
    if ok {
        Ok(())
    } else {
        Err(msg())
    }
}

/// Parses and validates a JSON instance. Values must be >= 0: the solver's dominance rules assume
/// every bonus is non-negative.
pub fn parse(json: &str) -> Result<Problem, String> {
    let input: ProblemInput =
        serde_json::from_str(json).map_err(|e| format!("malformed JSON: {e}"))?;
    validate(input)
}

pub fn validate(input: ProblemInput) -> Result<Problem, String> {
    let (w, h) = (input.width, input.height);
    require(w > 0 && h > 0 && w <= MAX_SIDE && h <= MAX_SIDE, || {
        format!("width/height must be in 1..={MAX_SIDE}")
    })?;
    for c in &input.blocked {
        require(c[0] >= 0 && c[0] < w && c[1] >= 0 && c[1] < h, || {
            format!(
                "blocked cell {c:?} out of board: cells are [x, y] with x in 0..{}, y in 0..{}",
                w - 1,
                h - 1
            )
        })?;
    }
    require(input.items.len() <= MAX_ITEMS, || {
        format!("at most {MAX_ITEMS} items supported")
    })?;
    let mut items = Vec::with_capacity(input.items.len());
    for it in input.items {
        let name = it.name;
        require(it.value >= 0 && it.value <= 1_000_000, || {
            format!("{name}: value must be an integer in 0..=1000000")
        })?;
        require(!(it.r#type.is_some() && it.shape.is_some()), || {
            format!("{name}: give either type or shape, not both")
        })?;
        let shape = match &it.r#type {
            Some(t) => SHAPES
                .iter()
                .find(|(n, _)| n == t)
                .map(|(_, s)| s.to_vec())
                .ok_or_else(|| {
                    format!("{name}: unknown type '{t}', expected one of U, F, L, T, O, V")
                })?,
            None => it.shape.clone().unwrap_or_default(),
        };
        require(!shape.is_empty(), || {
            format!("{name}: needs a type or a non-empty shape")
        })?;
        let mut seen = HashSet::new();
        for c in &shape {
            require(seen.insert(*c), || format!("{name}: duplicate shape cell"))?;
        }
        items.push(Item {
            name,
            value: it.value as i32,
            r#type: it.r#type,
            shape,
            border: it.border,
            adj_bonus: it.adj_bonus,
        });
    }
    Ok(Problem {
        width: w,
        height: h,
        blocked: input.blocked,
        items,
    })
}

impl Problem {
    pub fn blocked_mask(&self) -> Vec<bool> {
        let mut mask = vec![false; (self.width * self.height) as usize];
        for c in &self.blocked {
            mask[(c[1] * self.width + c[0]) as usize] = true;
        }
        mask
    }
}
