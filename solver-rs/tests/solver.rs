//! Brute-force cross-check and rule cases for the exact solver.

use backpack_solver::exact::{self, Layout, Outcome};
use backpack_solver::geometry::placements;
use backpack_solver::problem::{self, Problem};
use backpack_solver::verifier::evaluate;

/// Optimum of the maximized attribute, or None if no layout meets the targets/required items.
fn solve_value(p: &Problem) -> Option<i64> {
    match exact::solve(p, &mut |_, _, _| {}).expect("solves") {
        Outcome::Solved(sol) => {
            let score = evaluate(p, &sol.layout).expect("solver produced an illegal layout");
            assert!(
                score.feasible(p, &sol.layout),
                "solver layout misses a target or required item"
            );
            assert_eq!(
                score.objective(p),
                sol.value,
                "solver value vs verifier score"
            );
            Some(sol.value)
        }
        Outcome::Infeasible => None,
    }
}

fn parse(json: &str) -> Problem {
    problem::parse(json).expect("valid test instance")
}

#[test]
fn piece_reward_sums_bases_before_flooring() {
    // Row of 4: A(3) [piece] B(4) -> piece gets floor(7/5)=1; per-item flooring would give 0.
    let p = parse(
        r#"{"width":4,"height":1,"items":[
        {"name":"A","value":3,"shape":[[0,0]]},{"name":"B","value":4,"shape":[[0,0]]}]}"#,
    );
    assert_eq!(solve_value(&p), Some(3 + 4 + 1));
}

#[test]
fn item_bonus_merges_before_flooring() {
    // 1-cell item (15, adj) in the centre of 3x3 with 4 pieces around it:
    // item = 15 + floor(15*4/10) = 21 (separate flooring: 19), each piece floor(15/5) = 3.
    let p = parse(
        r#"{"width":3,"height":3,"items":[{"name":"A","value":15,"shape":[[0,0]],"adjBonus":true}]}"#,
    );
    assert_eq!(solve_value(&p), Some(21 + 4 * 3));
}

#[test]
fn border_bonus_merges_with_adjacency_bonus() {
    // 1x3 board, item(5, top, adj) at an end + one piece: 5 + floor(5*(2+1)/10) = 6, piece floor(5/5) = 1.
    let p = parse(
        r#"{"width":3,"height":1,"items":[
        {"name":"A","value":5,"shape":[[0,0]],"border":"top","adjBonus":true}]}"#,
    );
    assert_eq!(solve_value(&p), Some(6 + 1));
}

#[test]
fn each_touched_border_side_adds_its_own_bonus() {
    // 2x2 board, one 1x1 item worth 10: the top-left cell touches both top and left.
    let solve = |border: &str| {
        solve_value(&parse(&format!(
            r#"{{"width":2,"height":2,"blocked":[[1,0],[0,1],[1,1]],"items":[
            {{"name":"A","value":10,"shape":[[0,0]],"border":{border}}}]}}"#
        )))
    };
    assert_eq!(solve(r#""top""#), Some(12));
    assert_eq!(solve(r#"["top"]"#), Some(12));
    assert_eq!(solve(r#"["top","left"]"#), Some(14)); // 10 + 40%
    assert_eq!(solve(r#"["bottom","right"]"#), Some(10)); // touches neither
                                                          // negative: -56 with two sides -> -56 + trunc(-56 * 4 / 10) = -78
    let p = parse(
        r#"{"width":1,"height":1,"items":[
        {"name":"A","performance":-56,"shape":[[0,0]],"border":["top","right"],"required":true}]}"#,
    );
    assert_eq!(solve_value(&p), Some(-78));
}

#[test]
fn negative_bonus_rounds_toward_zero() {
    // -56 with the top bonus: -56 + trunc(-11.2) = -67 (floor would give -68). Required, so placed.
    let p = parse(
        r#"{"width":1,"height":1,"items":[
        {"name":"A","performance":-56,"shape":[[0,0]],"border":"top","required":true}]}"#,
    );
    assert_eq!(solve_value(&p), Some(-67));
}

#[test]
fn negative_piece_sum_rounds_toward_zero() {
    // A [piece] B: the piece is needed for quality 12 (5 + 5 + floor(10/5)); its performance is
    // trunc((-3 - 4) / 5) = -1 (floor: -2), so performance = -3 - 4 - 1 = -8.
    let p = parse(
        r#"{"width":4,"height":1,"goal":{"performance":"max","quality":12},"items":[
        {"name":"A","performance":-3,"quality":5,"shape":[[0,0]],"required":true},
        {"name":"B","performance":-4,"quality":5,"shape":[[0,0]],"required":true}]}"#,
    );
    assert_eq!(solve_value(&p), Some(-8));
}

#[test]
fn target_trades_off_the_maximized_attribute() {
    // A: performance 10, quality 0. B: performance 1, quality 9. One cell only.
    // Without a target A wins; quality >= 5 forces B.
    let base = r#""items":[
        {"name":"A","performance":10,"shape":[[0,0]]},
        {"name":"B","performance":1,"quality":9,"shape":[[0,0]]}]"#;
    let free = parse(&format!(r#"{{"width":1,"height":1,{base}}}"#));
    assert_eq!(solve_value(&free), Some(10));
    let target = parse(&format!(
        r#"{{"width":1,"height":1,"goal":{{"performance":"max","quality":5}},{base}}}"#
    ));
    assert_eq!(solve_value(&target), Some(1));
}

#[test]
fn without_max_the_fewest_cells_are_used() {
    // 3x1 board. Quality >= 10 needs B (one cell, quality 10) or A (two cells, quality 12).
    let p = parse(
        r#"{"width":3,"height":1,"goal":{"quality":10},"items":[
        {"name":"A","quality":12,"shape":[[0,0],[1,0]]},
        {"name":"B","quality":10,"shape":[[0,0]]}]}"#,
    );
    assert_eq!(solve_value(&p), Some(2)); // B placed, two cells stay empty
}

#[test]
fn unreachable_target_or_required_is_infeasible() {
    let target = parse(
        r#"{"width":2,"height":1,"goal":{"performance":"max","quality":1000},"items":[
        {"name":"A","performance":1,"quality":5,"shape":[[0,0]]}]}"#,
    );
    assert_eq!(solve_value(&target), None);
    let required = parse(
        r#"{"width":2,"height":1,"items":[
        {"name":"A","value":1,"shape":[[0,0],[1,0]],"required":true},
        {"name":"B","value":1,"shape":[[0,0]],"required":true}]}"#,
    );
    assert_eq!(solve_value(&required), None);
}

#[test]
fn blocked_cells_are_respected() {
    let p = parse(
        r#"{"width":3,"height":1,"blocked":[[1,0]],"items":[{"name":"A","value":10,"shape":[[0,0],[1,0]]}]}"#,
    );
    assert_eq!(solve_value(&p), Some(0));
}

#[test]
fn built_in_type_matches_explicit_shape() {
    let by_type = parse(r#"{"width":3,"height":2,"items":[{"name":"A","value":9,"type":"U"}]}"#);
    let by_shape = parse(
        r#"{"width":3,"height":2,"items":[{"name":"A","value":9,"shape":[[0,0],[1,0],[2,0],[0,1],[2,1]]}]}"#,
    );
    assert_eq!(solve_value(&by_type), solve_value(&by_shape));
}

#[test]
fn rejects_bad_input() {
    for bad in [
        r#"{"width":3,"height":2,"items":[{"name":"A","value":9,"type":"Q"}]}"#,
        r#"{"width":3,"height":2,"items":[{"name":"A","value":9,"type":"O","shape":[[0,0]]}]}"#,
        r#"{"width":3,"height":2,"blocked":[[0,6]],"items":[]}"#,
        r#"{"width":3,"height":2,"items":[{"name":"A","value":1,"performance":2,"type":"O"}]}"#,
        r#"{"width":3,"height":2,"items":[{"name":"A","value":2000000,"type":"O"}]}"#,
        r#"{"width":3,"height":2,"goal":{"performance":"max","quality":"max"},"items":[]}"#,
        r#"{"width":3,"height":2,"goal":{"speed":"max"},"items":[]}"#,
        r#"{"width":3,"#,
        r#"{"width":2,"height":2,"items":[{"name":"A","value":1,"type":"V","border":["top","bottom"]}]}"#,
        r#"{"width":2,"height":2,"items":[{"name":"A","value":1,"type":"V","border":["left","right"]}]}"#,
        r#"{"width":2,"height":2,"items":[{"name":"A","value":1,"type":"V","border":["top","top"]}]}"#,
    ] {
        assert!(problem::parse(bad).is_err(), "should reject {bad}");
    }
}

#[test]
fn verifier_rejects_overlap_and_wrong_shape() {
    let p =
        parse(r#"{"width":3,"height":1,"items":[{"name":"A","value":1,"shape":[[0,0],[1,0]]}]}"#);
    let overlap = Layout {
        item_cells: vec![Some(vec![0, 1])],
        dominoes: vec![[1, 2]],
    };
    assert!(evaluate(&p, &overlap).is_err());
    let wrong = Layout {
        item_cells: vec![Some(vec![0, 2])],
        dominoes: vec![],
    };
    assert!(evaluate(&p, &wrong).is_err());
}

#[test]
fn matches_brute_force_on_random_single_value_boards() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    for t in 0..150 {
        let json = random_instance(&mut rng, false);
        let p = parse(&json);
        assert_eq!(solve_value(&p), brute_force(&p), "instance {t}: {json}");
    }
}

#[test]
fn matches_brute_force_with_goals_negatives_and_required_items() {
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    for t in 0..300 {
        let json = random_instance(&mut rng, true);
        let p = parse(&json);
        assert_eq!(solve_value(&p), brute_force(&p), "instance {t}: {json}");
    }
}

/// xorshift64*: deterministic, dependency-free.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    /// Uniform in lo..=hi.
    fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + self.below((hi - lo + 1) as u64) as i64
    }
}

/// Random small instance. `full`: three attributes (some negative), a random goal with targets,
/// and required items; otherwise the legacy single positive value.
fn random_instance(rng: &mut Rng, full: bool) -> String {
    let (w, h) = (2 + rng.below(3), 2 + rng.below(3));
    let blocked: Vec<String> = (0..w * h)
        .filter(|_| rng.below(7) == 0)
        .map(|c| format!("[{},{}]", c % w, c / w))
        .collect();
    let borders = [
        "",
        "",
        "\"top\"",
        "\"bottom\"",
        "\"left\"",
        "\"right\"",
        "[\"top\",\"left\"]",
        "[\"bottom\",\"right\"]",
    ];
    let mut items: Vec<String> = Vec::new();
    let mut shapes: Vec<String> = Vec::new();
    for i in 0..1 + rng.below(4) {
        // reuse an earlier shape now and then, sometimes as an exact copy
        if i > 0 && rng.below(4) == 0 {
            let k = rng.below(items.len() as u64) as usize;
            let copy = items[k].replacen(&format!("\"I{k}\""), &format!("\"I{i}\""), 1);
            items.push(copy);
            shapes.push(shapes[k].clone());
            continue;
        }
        let shape = if i > 0 && rng.below(3) == 0 {
            shapes[rng.below(shapes.len() as u64) as usize].clone()
        } else {
            let size = 1 + rng.below(4) as usize;
            random_polyomino(rng, size)
        };
        let values = if full {
            let attr = |rng: &mut Rng| {
                if rng.below(3) == 0 {
                    rng.range(-30, -1)
                } else {
                    rng.range(0, 40)
                }
            };
            let (p, q, e) = (attr(rng), attr(rng), attr(rng));
            let required = rng.below(5) == 0;
            format!(
                "\"performance\":{p},\"quality\":{q},\"efficiency\":{e},\"required\":{required}"
            )
        } else {
            format!("\"value\":{}", 1 + rng.below(40))
        };
        let border = borders[rng.below(borders.len() as u64) as usize];
        let border = if border.is_empty() {
            String::new()
        } else {
            format!(",\"border\":{border}")
        };
        let adj = rng.below(2) == 0;
        items.push(format!(
            "{{\"name\":\"I{i}\",{values},\"shape\":{shape}{border},\"adjBonus\":{adj}}}"
        ));
        shapes.push(shape);
    }
    let goal = if full {
        let attrs = ["performance", "quality", "efficiency"];
        let max = rng.below(4) as usize; // 3: nothing maximized, keep cells empty
        let mut parts = Vec::new();
        if max < 3 {
            parts.push(format!("\"{}\":\"max\"", attrs[max]));
        }
        for (a, name) in attrs.iter().enumerate() {
            if a != max {
                match rng.below(3) {
                    0 => parts.push(format!("\"{name}\":{}", rng.range(-20, 60))),
                    1 => parts.push(format!("\"{name}\":\"skip\"")),
                    _ => {}
                }
            }
        }
        format!(",\"goal\":{{{}}}", parts.join(","))
    } else {
        String::new()
    };
    format!(
        "{{\"width\":{w},\"height\":{h},\"blocked\":[{}],\"items\":[{}]{goal}}}",
        blocked.join(","),
        items.join(",")
    )
}

fn random_polyomino(rng: &mut Rng, size: usize) -> String {
    let mut cells = vec![(0i32, 0i32)];
    while cells.len() < size {
        let (x, y) = cells[rng.below(cells.len() as u64) as usize];
        let (dx, dy) = [(1, 0), (-1, 0), (0, 1), (0, -1)][rng.below(4) as usize];
        if !cells.contains(&(x + dx, y + dy)) {
            cells.push((x + dx, y + dy));
        }
    }
    let parts: Vec<String> = cells.iter().map(|(x, y)| format!("[{x},{y}]")).collect();
    format!("[{}]", parts.join(","))
}

/// Exhaustive: scan cells in order; each free cell is left empty, starts a piece, or anchors an
/// item. Best objective over layouts that meet every target and place every required item.
fn brute_force(p: &Problem) -> Option<i64> {
    let (w, n) = (p.width as usize, (p.width * p.height) as usize);
    let pls: Vec<_> = (0..p.items.len()).map(|i| placements(p, i)).collect();
    let mut occ = p.blocked_mask();
    let mut layout = Layout {
        item_cells: vec![None; p.items.len()],
        dominoes: Vec::new(),
    };
    dfs(p, &pls, w, n, 0, &mut occ, &mut layout)
}

fn dfs(
    p: &Problem,
    pls: &[Vec<backpack_solver::geometry::Placement>],
    w: usize,
    n: usize,
    mut c: usize,
    occ: &mut Vec<bool>,
    layout: &mut Layout,
) -> Option<i64> {
    while c < n && occ[c] {
        c += 1;
    }
    if c == n {
        let score = evaluate(p, layout).expect("brute force builds legal layouts");
        return score.feasible(p, layout).then(|| score.objective(p));
    }
    let mut best = dfs(p, pls, w, n, c + 1, occ, layout); // leave empty
    for other in [c + 1, c + w] {
        let ok = other < n && !occ[other] && (other == c + w || c % w + 1 < w);
        if !ok {
            continue;
        }
        occ[c] = true;
        occ[other] = true;
        layout.dominoes.push([c, other]);
        best = best.max(dfs(p, pls, w, n, c + 1, occ, layout));
        layout.dominoes.pop();
        occ[c] = false;
        occ[other] = false;
    }
    for i in 0..pls.len() {
        if layout.item_cells[i].is_some() {
            continue;
        }
        for pl in &pls[i] {
            if pl.cells[0] != c || pl.cells.iter().any(|&k| occ[k]) {
                continue;
            }
            for &k in &pl.cells {
                occ[k] = true;
            }
            layout.item_cells[i] = Some(pl.cells.clone());
            best = best.max(dfs(p, pls, w, n, c + 1, occ, layout));
            layout.item_cells[i] = None;
            for &k in &pl.cells {
                occ[k] = false;
            }
        }
    }
    best
}

/// Objective the JSON solver reports: the maximized total, or empty cells.
fn reported_value(p: &Problem, out: &backpack_solver::SolutionOut) -> i64 {
    p.goal.maximize.map_or(out.empty_cells, |a| out.totals[a])
}

/// Many items over few shapes, so plenty of them are dominated.
fn crowded_instance(rng: &mut Rng) -> String {
    let base = random_instance(rng, true);
    let mut v: serde_json::Value = serde_json::from_str(&base).unwrap();
    let shapes: Vec<serde_json::Value> = (0..1 + rng.below(2))
        .map(|_| {
            let size = 2 + rng.below(2) as usize;
            serde_json::from_str(&random_polyomino(rng, size)).unwrap()
        })
        .collect();
    let items = (0..3 + rng.below(4))
        .map(|i| {
            let attr = |rng: &mut Rng| rng.range(-10, 30);
            serde_json::json!({
                "name": format!("C{i}"),
                "shape": shapes[rng.below(shapes.len() as u64) as usize],
                "performance": attr(rng), "quality": attr(rng), "efficiency": attr(rng),
                "adjBonus": rng.below(3) == 0,
                "required": rng.below(8) == 0,
            })
        })
        .collect::<Vec<_>>();
    v["items"] = items.into();
    // small boards keep brute force fast; at most 3x3 also means few copies fit
    v["width"] = (2 + rng.below(2)).into();
    v["height"] = (2 + rng.below(2)).into();
    v["blocked"] = serde_json::json!([]);
    v.to_string()
}

#[test]
fn dropping_dominated_items_keeps_the_optimum() {
    let mut rng = Rng(0x1234_5678_9ABC_DEF1);
    let mut dropped = 0;
    for t in 0..200 {
        let json = crowded_instance(&mut rng);
        let p = parse(&json);
        let (reduced, _) = backpack_solver::reduce::reduce(&p, usize::MAX);
        dropped += p.items.len() - reduced.items.len();
        assert_eq!(
            brute_force(&reduced),
            brute_force(&p),
            "instance {t}: {json}"
        );
    }
    assert!(
        dropped > 100,
        "the test should exercise the reduction (dropped {dropped})"
    );
}

#[test]
fn fast_mode_finds_valid_layouts_close_to_the_optimum() {
    let mut rng = Rng(0x0BAD_C0DE_1234_5678);
    let (mut feasible, mut found, mut optimal) = (0, 0, 0);
    for t in 0..300 {
        let json = random_instance(&mut rng, true);
        let p = parse(&json);
        let best = brute_force(&p);
        let fast = json.replacen('{', "{\"mode\":\"fast\",", 1);
        // solve_json checks the layout with the verifier before returning it
        let out = backpack_solver::solve_json(&fast, &mut |_, _, _| {}).expect("solves");
        assert!(out.approximate);
        if out.feasible {
            let v = reported_value(&p, &out);
            let best = best.unwrap_or_else(|| {
                panic!("instance {t}: fast found a layout brute force didn't: {json}")
            });
            assert!(
                v <= best,
                "instance {t}: fast {v} beats the optimum {best}: {json}"
            );
            found += 1;
            optimal += usize::from(v == best);
        }
        feasible += usize::from(best.is_some());
    }
    assert_eq!(found, feasible, "fast mode missed feasible instances");
    assert!(
        optimal * 100 >= found * 95,
        "fast mode optimal on only {optimal} of {found}"
    );
}

#[test]
fn fast_mode_takes_hundreds_of_items() {
    let mut rng = Rng(0x5EED_0000_0000_0001);
    let types = ["U", "F", "L", "T", "O", "V", "I"];
    let items: Vec<serde_json::Value> = (0..400)
        .map(|i| {
            serde_json::json!({
                "name": format!("N{i}"),
                "type": types[rng.below(7) as usize],
                "performance": rng.range(0, 60), "quality": rng.range(-20, 40),
                "efficiency": rng.range(-20, 20), "adjBonus": rng.below(3) == 0,
                "required": i == 399,
            })
        })
        .collect();
    let mut v = serde_json::json!({
        "width": 7, "height": 5, "items": items,
        "goal": {"performance": "max", "quality": 60},
    });
    v["mode"] = "fast".into();
    let out = backpack_solver::solve_json(&v.to_string(), &mut |_, _, _| {}).expect("solves");
    assert!(out.feasible && out.totals[1] >= 60);
    assert!(out.items_considered <= problem::FAST_MAX_ITEMS);
    assert!(
        out.items.iter().any(|it| it.index == 399),
        "the required item is placed"
    );
}

#[test]
fn exact_mode_points_to_fast_mode_when_too_many_items_matter() {
    // performance falls as quality rises: no item dominates another, so none can be dropped
    let items: Vec<serde_json::Value> = (0..70)
        .map(|i| serde_json::json!({"name": format!("V{i}"), "type": "V", "performance": 100 - i, "quality": i}))
        .collect();
    let v = serde_json::json!({
        "width": 7, "height": 5, "items": items,
        "goal": {"performance": "max", "quality": 10},
    });
    let exact = backpack_solver::solve_json(&v.to_string(), &mut |_, _, _| {});
    assert!(
        matches!(&exact, Err(e) if e.contains("快速模式")),
        "exact mode should point to fast mode"
    );
}

/// User report: on a 31-cell board quality can reach only 108, so quality >= 100 plus
/// performance >= 120 is tight; the fast mode must still find a layout (one exists: 145/108).
#[test]
fn fast_mode_meets_tight_targets_on_a_cornered_board() {
    let json = r#"{
        "width": 7, "height": 5, "blocked": [[0,0],[6,0],[0,4],[6,4]],
        "goal": {"performance": 120, "quality": 100, "efficiency": "max"}, "mode": "fast",
        "items": [
            {"name":"U1","type":"U","performance":54,"quality":16,"efficiency":0,"adjBonus":false},
            {"name":"U2","type":"U","performance":32,"quality":18,"efficiency":-16,"adjBonus":false},
            {"name":"F3","type":"F","performance":32,"quality":32,"efficiency":0,"adjBonus":false},
            {"name":"F4","type":"F","performance":32,"quality":16,"efficiency":-16,"adjBonus":false},
            {"name":"L5","type":"L","performance":32,"quality":0,"efficiency":12,"adjBonus":false},
            {"name":"L6","type":"L","performance":0,"quality":16,"efficiency":12,"adjBonus":false},
            {"name":"T7","type":"T","performance":16,"quality":0,"efficiency":12,"adjBonus":false},
            {"name":"T8","type":"T","performance":0,"quality":8,"efficiency":12,"adjBonus":false},
            {"name":"O9","type":"O","performance":16,"quality":0,"efficiency":12,"adjBonus":false},
            {"name":"O10","type":"O","performance":0,"quality":8,"efficiency":12,"adjBonus":false},
            {"name":"V11","type":"V","performance":14,"quality":0,"efficiency":8,"adjBonus":false},
            {"name":"V12","type":"V","performance":0,"quality":6,"efficiency":8,"adjBonus":false},
            {"name":"I13","type":"I","performance":0,"quality":-15,"efficiency":-15,"adjBonus":false,"required":true}
        ]
    }"#;
    let out = backpack_solver::solve_json(json, &mut |_, _, _| {}).expect("solves");
    assert!(out.feasible, "fast mode found no layout");
    assert!(out.totals[0] >= 120 && out.totals[1] >= 100);
}
