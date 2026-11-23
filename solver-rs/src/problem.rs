//! Input model, validation and the built-in shapes. Coordinates are [x, y], origin top-left.
//!
//! Every item has three attributes (performance, quality, efficiency), each scored with the same
//! rules. The goal maximizes at most one attribute, may require others to reach a target total,
//! and ignores the rest. With no attribute maximized, it leaves as many cells empty as possible
//! (items and 1x2 pieces both occupy). Values may be negative; all percentages round toward zero.

use serde::Deserialize;
use std::collections::{HashMap, HashSet};

pub type Cell = [i32; 2];

/// Items accepted in the input; the solver then drops items that can't matter (see reduce).
pub const MAX_ITEMS: usize = 5000;
/// Items the exact solver can take after that (u64 item sets).
pub const EXACT_MAX_ITEMS: usize = 63;
/// Items the fast mode keeps: more are cut greedily by value per cell.
pub const FAST_MAX_ITEMS: usize = 255;
pub const MAX_SIDE: i32 = 12;
pub const MAX_ABS_VALUE: i64 = 1_000_000;
/// JSON names of the attributes, in index order.
pub const ATTRS: [&str; 3] = ["performance", "quality", "efficiency"];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Deserialize)]
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
    /// Legacy single value: same as `performance`.
    #[serde(default)]
    pub value: Option<i64>,
    #[serde(default)]
    pub performance: Option<i64>,
    #[serde(default)]
    pub quality: Option<i64>,
    #[serde(default)]
    pub efficiency: Option<i64>,
    #[serde(default)]
    pub r#type: Option<String>,
    #[serde(default)]
    pub shape: Option<Vec<Cell>>,
    #[serde(default)]
    pub border: Option<Border>,
    #[serde(default)]
    pub adj_bonus: bool,
    /// The item must be placed.
    #[serde(default)]
    pub required: bool,
}

/// Goal for one attribute: "max", "skip", or a target total (number).
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum GoalSpec {
    Target(i64),
    Mode(String),
}

#[derive(Debug, Deserialize)]
pub struct ProblemInput {
    pub width: i32,
    pub height: i32,
    #[serde(default)]
    pub blocked: Vec<Cell>,
    pub items: Vec<ItemInput>,
    /// Defaults to {"performance": "max"}.
    #[serde(default)]
    pub goal: Option<HashMap<String, GoalSpec>>,
    /// "exact" (default): proven optimum. "fast": beam search, approximate but quick.
    #[serde(default)]
    pub mode: Option<String>,
}

#[derive(Clone, Debug)]
pub struct Item {
    pub name: String,
    /// performance, quality, efficiency
    pub attrs: [i32; 3],
    pub r#type: Option<String>,
    pub shape: Vec<Cell>,
    pub border: Option<Border>,
    pub adj_bonus: bool,
    pub required: bool,
}

/// Maximize one attribute (None: the number of empty cells) subject to `total >= target` for each
/// listed attribute.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Goal {
    pub maximize: Option<usize>,
    pub targets: Vec<(usize, i64)>,
}

#[derive(Clone, Debug)]
pub struct Problem {
    pub width: i32,
    pub height: i32,
    pub blocked: Vec<Cell>,
    pub items: Vec<Item>,
    pub goal: Goal,
    /// Approximate beam search instead of the exact solver.
    pub fast: bool,
}

/// Built-in shapes, usable via `type` (rotations and mirrors are always allowed).
pub const SHAPES: [(&str, &[Cell]); 7] = [
    ("U", &[[0, 0], [1, 0], [2, 0], [0, 1], [2, 1]]),
    ("F", &[[0, 0], [1, 0], [2, 0], [1, 1], [2, 1]]),
    ("L", &[[0, 0], [1, 0], [2, 0], [0, 1]]),
    ("T", &[[0, 0], [1, 0], [2, 0], [1, 1]]),
    ("O", &[[0, 0], [1, 0], [0, 1], [1, 1]]),
    ("V", &[[0, 0], [1, 0], [0, 1]]),
    ("I", &[[0, 0], [1, 0], [2, 0], [3, 0]]),
];

fn require(ok: bool, msg: impl FnOnce() -> String) -> Result<(), String> {
    if ok {
        Ok(())
    } else {
        Err(msg())
    }
}

/// Parses and validates a JSON instance.
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
    let items = input
        .items
        .into_iter()
        .map(validate_item)
        .collect::<Result<Vec<_>, _>>()?;
    let goal = validate_goal(input.goal)?;
    let fast = match input.mode.as_deref() {
        None | Some("exact") => false,
        Some("fast") => true,
        Some(m) => return Err(format!("mode: expected \"exact\" or \"fast\", got \"{m}\"")),
    };
    Ok(Problem {
        width: w,
        height: h,
        blocked: input.blocked,
        items,
        goal,
        fast,
    })
}

fn validate_item(it: ItemInput) -> Result<Item, String> {
    let name = it.name;
    let named = it.performance.is_some() || it.quality.is_some() || it.efficiency.is_some();
    require(!(it.value.is_some() && named), || {
        format!("{name}: give either value or performance/quality/efficiency, not both")
    })?;
    let raw = [
        it.performance.or(it.value).unwrap_or(0),
        it.quality.unwrap_or(0),
        it.efficiency.unwrap_or(0),
    ];
    for (v, attr) in raw.iter().zip(ATTRS) {
        require(v.abs() <= MAX_ABS_VALUE, || {
            format!("{name}: {attr} must be an integer in -{MAX_ABS_VALUE}..={MAX_ABS_VALUE}")
        })?;
    }
    require(!(it.r#type.is_some() && it.shape.is_some()), || {
        format!("{name}: give either type or shape, not both")
    })?;
    let shape = match &it.r#type {
        Some(t) => SHAPES
            .iter()
            .find(|(n, _)| n == t)
            .map(|(_, s)| s.to_vec())
            .ok_or_else(|| {
                format!("{name}: unknown type '{t}', expected one of U, F, L, T, O, V, I")
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
    Ok(Item {
        name,
        attrs: raw.map(|v| v as i32),
        r#type: it.r#type,
        shape,
        border: it.border,
        adj_bonus: it.adj_bonus,
        required: it.required,
    })
}

fn validate_goal(goal: Option<HashMap<String, GoalSpec>>) -> Result<Goal, String> {
    let Some(goal) = goal else {
        return Ok(Goal {
            maximize: Some(0),
            targets: Vec::new(),
        });
    };
    let mut maximize = None;
    let mut targets = Vec::new();
    for (key, spec) in goal {
        let attr = ATTRS.iter().position(|a| *a == key).ok_or_else(|| {
            format!("goal: unknown attribute '{key}', expected performance, quality or efficiency")
        })?;
        match spec {
            GoalSpec::Mode(m) if m == "max" => {
                require(maximize.is_none(), || {
                    "goal: at most one attribute can be \"max\"".to_string()
                })?;
                maximize = Some(attr);
            }
            GoalSpec::Mode(m) if m == "skip" => {}
            GoalSpec::Mode(m) => {
                return Err(format!(
                    "goal.{key}: expected \"max\", \"skip\" or a number, got \"{m}\""
                ))
            }
            GoalSpec::Target(t) => {
                require(t.abs() <= MAX_ABS_VALUE * 1000, || {
                    format!("goal.{key}: target out of range")
                })?;
                targets.push((attr, t));
            }
        }
    }
    targets.sort_unstable();
    Ok(Goal { maximize, targets })
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
