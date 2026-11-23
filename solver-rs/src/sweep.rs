//! Cell-by-cell sweep shared by every round of the exact solver. Cells are scanned lane by lane
//! along the board's shorter side.
//!
//! At each step the first undecided cell (pos) is left empty, starts a 1x2 piece, or anchors an item
//! placement (anchor = the placement's first cell in scan order), so each layout is generated once.
//! Identical items are merged into classes, so permutations of them are never explored.
//!
//! The board side of a state is its key: pos, the cells in the window [pos - lane, pos + reach)
//! and the kind of each open entity. All states in one Pareto front share it, so cells are never
//! stored per state; they are decoded when expanding.
//!
//! Scoring, per slot (active attribute: the maximized one, then each target). Rewards are paid as
//! they change: an item is worth v + trunc(v * m / 10) with m = 2*border + adjacent pieces, a piece
//! trunc(S / 5) with S the summed bases of its adjacent items; each new adjacency pays
//! new - old of those values, which telescopes to exactly the final value, toward-zero rounding and
//! negative values included. An open item carries m, an open piece S per slot. In a slot where no
//! value is negative, trunc is floor and only S mod 5 affects the future, so that is all a piece
//! keeps there (more states then compare equal).

use crate::geometry::{neighbours, placements, shape_key};
use crate::problem::Problem;
use smallvec::SmallVec;
use std::collections::HashMap;

const UNDEC: i8 = -1;
const EMPTY: i8 = -2;
const BLOCKED: i8 = -3;
const OCCUPIED: i8 = -4; // taken by an entity that can no longer gain adjacencies
pub const DOMINO: i32 = -1;
const MAX_OPEN: usize = 63; // open entities are tracked in u64 bitsets
/// Attribute slots: [objective, target...]; with nothing maximized all three attributes can be
/// targets after the empty-cell objective. Unused slots stay 0.
pub const SLOTS: usize = 4;

/// Step codes stored per search node (with its pos) to rebuild the layout.
pub const STEP_EMPTY: u32 = 0;
pub const STEP_PIECE_ALONG: u32 = 1; // cells pos, pos + 1
pub const STEP_PIECE_ACROSS: u32 = 2; // cells pos, pos + lane
pub const STEP_ANCHORED: u32 = 3; // + index into the anchored placements at pos

/// Open entity. Item: tag = vclass << 8 | m (sums unused). 1x2 piece: tag = PIECE, sums = summed
/// bases of the adjacent items per slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ent {
    pub tag: u32,
    pub sums: [i32; SLOTS],
}

pub const PIECE: u32 = u32::MAX;

impl Ent {
    pub fn is_piece(&self) -> bool {
        self.tag == PIECE
    }
    pub fn vclass(&self) -> usize {
        (self.tag >> 8) as usize
    }
    pub fn mult(&self) -> i32 {
        (self.tag & 0xFF) as i32
    }
}

// Inline capacities fit the common case (keys are ~20-30 bytes, a handful of open entities);
// longer ones spill to the heap. Sized for memory: fronts dominate the footprint.
pub type Key = SmallVec<[u8; 32]>;
pub type Ents = SmallVec<[Ent; 6]>;
pub type Score = [i32; SLOTS];

/// Distinct (slot values, adjacency flag) of the items: all an open item needs for the future.
pub struct ValueClass {
    pub vals: Score,
    pub adj_bonus: bool,
    /// Some slot value > 0: more adjacency is better there.
    pub any_pos: bool,
    /// Some slot value < 0: more adjacency is worse there.
    pub any_neg: bool,
}

pub struct ItemClass {
    pub vclass: usize,
    pub members: Vec<usize>,
    pub mask: u64,
}

struct Anchored {
    cls: usize,
    cells: Vec<usize>,
    touch: bool,
}

/// A successor state plus the reward collected by the step and the step code.
pub struct Child {
    pub key: Key,
    pub ents: Ents,
    pub used: u64,
    pub gain: Score,
    pub code: u32,
}

/// Per-thread scratch buffers.
pub struct Scratch {
    base: Vec<i8>,
    work: Vec<i8>,
}

pub struct Sweep {
    pub n: usize,
    pub classes: Vec<ItemClass>,
    pub vclasses: Vec<ValueClass>,
    /// Item bits that must be placed.
    pub required: u64,
    pub root_key: Key,
    lane: usize,
    /// Cells at or beyond pos + reach can't be covered yet: they are UNDEC or BLOCKED.
    reach: usize,
    scan_to_cell: Vec<usize>,
    nbr: Vec<Vec<usize>>,
    by_anchor: Vec<Vec<Anchored>>,
    blocked_scan: Vec<bool>,
    /// Non-blocked cells in [s, n), for the required-items capacity check.
    free_from: Vec<u32>,
    /// Cell count of each item, by item bit.
    area: Vec<u32>,
    /// WLOG two adjacent cells are never both left empty: valid only if a 1x2 piece can never
    /// lower any slot, i.e. no item has a negative slot value and empty cells aren't the goal.
    pair_rule: bool,
    /// Slot 0 counts empty cells (no attribute maximized).
    empty_reward: bool,
    /// Per slot: no item value is negative there.
    nonneg: [bool; SLOTS],
    /// Slots in use (maximized + targets); the rest are always 0.
    slots: usize,
    /// Upper bound on the pieces one item can still become adjacent to.
    max_adjacency: i32,
    /// Per slot: items with a negative value there.
    neg_items: [u64; SLOTS],
    /// Per item bit: the most placing it can lower each slot, its later adjacencies included.
    item_loss: Vec<Score>,
    /// Per value class: the most an already placed item can still lower each slot.
    open_loss: Vec<Score>,
}

impl Sweep {
    /// `slots` lists the attribute of each slot (maximized first, then targets); a None first
    /// slot counts empty cells instead.
    pub fn new(p: &Problem, slots: &[Option<usize>]) -> Sweep {
        let (w, h) = (p.width as usize, p.height as usize);
        let n = w * h;
        let along_y = h <= w;
        let lane = if along_y { h } else { w };
        let scan_to_cell: Vec<usize> = (0..n)
            .map(|s| {
                if along_y {
                    (s % lane) * w + s / lane
                } else {
                    (s / lane) * w + s % lane
                }
            })
            .collect();
        let mut cell_to_scan = vec![0; n];
        for (s, &c) in scan_to_cell.iter().enumerate() {
            cell_to_scan[c] = s;
        }
        let nbr = (0..n)
            .map(|s| {
                neighbours(scan_to_cell[s], w, h)
                    .into_iter()
                    .map(|c| cell_to_scan[c])
                    .collect()
            })
            .collect();

        let slot_vals = |i: usize| -> Score {
            std::array::from_fn(|j| {
                slots
                    .get(j)
                    .copied()
                    .flatten()
                    .map_or(0, |a| p.items[i].attrs[a])
            })
        };
        // identical items: same shape (up to rotation/mirror), slot values, border, flags
        type ClassKey = (Vec<[i32; 2]>, Score, String, bool, bool);
        let mut group_of: HashMap<ClassKey, usize> = HashMap::new();
        let mut groups: Vec<Vec<usize>> = Vec::new();
        for (i, it) in p.items.iter().enumerate() {
            let key = (
                shape_key(&it.shape),
                slot_vals(i),
                format!("{:?}", it.border),
                it.adj_bonus,
                it.required,
            );
            let g = *group_of.entry(key).or_insert_with(|| {
                groups.push(Vec::new());
                groups.len() - 1
            });
            groups[g].push(i);
        }
        let mut vclass_of: HashMap<(Score, bool), usize> = HashMap::new();
        let mut vclasses = Vec::new();
        let mut classes = Vec::new();
        let mut by_anchor: Vec<Vec<Anchored>> = (0..n).map(|_| Vec::new()).collect();
        let mut item_loss = vec![[0; SLOTS]; p.items.len()];
        let mut open_k: Vec<i32> = Vec::new(); // per value class: most adjacent pieces
        for members in groups {
            let rep = &p.items[members[0]];
            let vals = slot_vals(members[0]);
            let vclass = *vclass_of.entry((vals, rep.adj_bonus)).or_insert_with(|| {
                vclasses.push(ValueClass {
                    vals,
                    adj_bonus: rep.adj_bonus,
                    any_pos: vals.iter().any(|&v| v > 0),
                    any_neg: vals.iter().any(|&v| v < 0),
                });
                vclasses.len() - 1
            });
            let mask = members.iter().fold(0u64, |m, &i| m | 1 << i);
            // distinct adjacent pieces each cover one of the shape's neighbour cells
            let k = neighbour_cells(&rep.shape) as i32;
            if open_k.len() <= vclass {
                open_k.resize(vclass + 1, 0);
            }
            open_k[vclass] = open_k[vclass].max(k);
            for &i in &members {
                item_loss[i] = std::array::from_fn(|j| {
                    let v = vals[j];
                    if v >= 0 {
                        return 0;
                    }
                    let m = if rep.border.is_some() { 2 } else { 0 };
                    let own = -v
                        + if rep.adj_bonus {
                            -v * (m + k) / 10
                        } else {
                            -v * m / 10
                        };
                    own + k * ((-v + 4) / 5)
                });
            }
            let cls = classes.len();
            for pl in placements(p, members[0]) {
                let mut sc: Vec<usize> = pl.cells.iter().map(|&c| cell_to_scan[c]).collect();
                sc.sort_unstable();
                by_anchor[sc[0]].push(Anchored {
                    cls,
                    cells: sc,
                    touch: pl.touch,
                });
            }
            classes.push(ItemClass {
                vclass,
                members,
                mask,
            });
        }
        assert!(vclasses.len() < 1 << 23, "too many distinct items");
        let mut reach = lane + 1; // 1x2 pieces reach pos + lane
        for (s, list) in by_anchor.iter().enumerate() {
            for a in list {
                reach = reach.max(a.cells[a.cells.len() - 1] - s + 1);
            }
        }
        let blocked = p.blocked_mask();
        let blocked_scan: Vec<bool> = (0..n).map(|s| blocked[scan_to_cell[s]]).collect();
        let mut free_from = vec![0u32; n + 1];
        for s in (0..n).rev() {
            free_from[s] = free_from[s + 1] + u32::from(!blocked_scan[s]);
        }
        let required = p
            .items
            .iter()
            .enumerate()
            .filter(|(_, it)| it.required)
            .fold(0u64, |m, (i, _)| m | 1 << i);
        let area = p.items.iter().map(|it| it.shape.len() as u32).collect();
        let empty_reward = slots.first() == Some(&None);
        let pair_rule = !empty_reward && vclasses.iter().all(|vc| !vc.any_neg);
        let nonneg = std::array::from_fn(|j| vclasses.iter().all(|vc| vc.vals[j] >= 0));
        // each adjacent piece covers a distinct neighbour cell: at most 4 per item cell
        let max_adjacency = 4 * p.items.iter().map(|it| it.shape.len()).max().unwrap_or(0) as i32;
        let neg_items = std::array::from_fn(|j| {
            (0..p.items.len())
                .filter(|&i| slot_vals(i)[j] < 0)
                .fold(0u64, |m, i| m | 1 << i)
        });
        let open_loss = vclasses
            .iter()
            .enumerate()
            .map(|(c, vc)| {
                let k = open_k[c];
                std::array::from_fn(|j| {
                    let v = vc.vals[j];
                    if v >= 0 {
                        return 0;
                    }
                    let bonus = if vc.adj_bonus { -v * k / 10 + 1 } else { 0 };
                    bonus + k * ((-v + 4) / 5)
                })
            })
            .collect();

        let mut sweep = Sweep {
            n,
            classes,
            vclasses,
            required,
            root_key: Key::new(),
            lane,
            reach,
            scan_to_cell,
            nbr,
            by_anchor,
            blocked_scan,
            free_from,
            area,
            pair_rule,
            empty_reward,
            nonneg,
            max_adjacency,
            neg_items,
            item_loss,
            open_loss,
            slots: slots.len(),
        };
        let mut sc = sweep.scratch();
        for s in 0..n {
            sc.work[s] = if sweep.blocked_scan[s] {
                BLOCKED
            } else {
                UNDEC
            };
        }
        sweep.root_key = sweep
            .finish(0, Ents::new(), 0, [0; SLOTS], STEP_EMPTY, &mut sc)
            .0
            .key;
        sweep
    }

    pub fn scratch(&self) -> Scratch {
        Scratch {
            base: vec![0; self.n],
            work: vec![0; self.n],
        }
    }

    pub fn pos_of(key: &[u8]) -> usize {
        key[0] as usize
    }

    pub fn cell_of_scan(&self, s: usize) -> usize {
        self.scan_to_cell[s]
    }

    /// Class (or DOMINO) and scan cells of a step, for rebuilding the layout. None = left empty.
    pub fn step_cells(&self, pos: usize, code: u32) -> Option<(i32, Vec<usize>)> {
        match code {
            STEP_EMPTY => None,
            STEP_PIECE_ALONG => Some((DOMINO, vec![pos, pos + 1])),
            STEP_PIECE_ACROSS => Some((DOMINO, vec![pos, pos + self.lane])),
            k => {
                let a = &self.by_anchor[pos][(k - STEP_ANCHORED) as usize];
                Some((a.cls as i32, a.cells.clone()))
            }
        }
    }

    /// Emits every successor of the state (key, ents, used). Only tracked items are limited to one use;
    /// other classes may be reused (relaxation). Children that can no longer fit every unplaced
    /// required item are dropped.
    pub fn expand(
        &self,
        key: &[u8],
        ents: &[Ent],
        used: u64,
        tracked: u64,
        sc: &mut Scratch,
        emit: &mut impl FnMut(Child),
    ) {
        let pos = self.decode(key, sc);
        let (lane, n) = (self.lane, self.n);
        let base = &sc.base;
        let up_empty = !pos.is_multiple_of(lane) && base[pos - 1] == EMPTY;
        let left_empty = pos >= lane && base[pos - lane] == EMPTY;
        let can_empty = !self.pair_rule || (!up_empty && !left_empty);
        let along = pos % lane + 1 < lane && base[pos + 1] == UNDEC;
        let across = pos + lane < n && base[pos + lane] == UNDEC;
        let mut out = |(child, free): (Child, u32)| {
            if self.required_area(child.used) <= free {
                emit(child);
            }
        };
        if can_empty {
            out(self.place(
                pos, ents, used, DOMINO, None, false, tracked, STEP_EMPTY, sc,
            ));
        }
        if along {
            let cells = [pos, pos + 1];
            out(self.place(
                pos,
                ents,
                used,
                DOMINO,
                Some(&cells),
                false,
                tracked,
                STEP_PIECE_ALONG,
                sc,
            ));
        }
        if across {
            let cells = [pos, pos + lane];
            out(self.place(
                pos,
                ents,
                used,
                DOMINO,
                Some(&cells),
                false,
                tracked,
                STEP_PIECE_ACROSS,
                sc,
            ));
        }
        for (k, a) in self.by_anchor[pos].iter().enumerate() {
            let limited = self.classes[a.cls].mask & tracked;
            if limited != 0 && limited & !used == 0 {
                continue;
            }
            if a.cells.iter().all(|&c| sc.base[c] == UNDEC) {
                let code = STEP_ANCHORED + k as u32;
                out(self.place(
                    pos,
                    ents,
                    used,
                    a.cls as i32,
                    Some(&a.cells),
                    a.touch,
                    tracked,
                    code,
                    sc,
                ));
            }
        }
    }

    /// The most slot j can still drop from this state, or None if unbounded (a negative class that
    /// isn't tracked may be placed any number of times). Only negative values ever lower a slot:
    /// unplaced negative items, and the bonuses and pieces of placed ones still open.
    pub fn future_loss(&self, j: usize, ents: &[Ent], used: u64, tracked: u64) -> Option<i64> {
        let neg = self.neg_items[j];
        if neg & !tracked != 0 {
            return None;
        }
        let mut loss = 0i64;
        let mut todo = neg & !used;
        while todo != 0 {
            loss += self.item_loss[todo.trailing_zeros() as usize][j] as i64;
            todo &= todo - 1;
        }
        for e in ents.iter().filter(|e| !e.is_piece()) {
            loss += self.open_loss[e.vclass()][j] as i64;
        }
        Some(loss)
    }

    /// Cells still needed by required items not placed yet.
    fn required_area(&self, used: u64) -> u32 {
        let mut missing = self.required & !used;
        let mut total = 0;
        while missing != 0 {
            total += self.area[missing.trailing_zeros() as usize];
            missing &= missing - 1;
        }
        total
    }

    /// Fills sc.base with the cells encoded in key; cells before the window become OCCUPIED.
    fn decode(&self, key: &[u8], sc: &mut Scratch) -> usize {
        let pos = key[0] as usize;
        let from = pos.saturating_sub(self.lane);
        let to = self.n.min(pos + self.reach);
        let base = &mut sc.base;
        base[..from].fill(OCCUPIED);
        for c in from..to {
            base[c] = key[1 + c - from] as i8 + OCCUPIED;
        }
        for c in to..self.n {
            base[c] = if self.blocked_scan[c] { BLOCKED } else { UNDEC };
        }
        pos
    }

    #[allow(clippy::too_many_arguments)]
    fn place(
        &self,
        pos: usize,
        ents_in: &[Ent],
        mut used: u64,
        cls: i32,
        cells: Option<&[usize]>,
        touch: bool,
        tracked: u64,
        code: u32,
        sc: &mut Scratch,
    ) -> (Child, u32) {
        sc.work.copy_from_slice(&sc.base);
        let owner = &mut sc.work;
        let mut ents: Ents = Ents::from_slice(ents_in);
        let mut gain = [0; SLOTS];
        match cells {
            None => {
                owner[pos] = EMPTY;
                gain[0] += i32::from(self.empty_reward);
            }
            Some(cells) => {
                let id = ents.len();
                for &c in cells {
                    owner[c] = id as i8;
                }
                let me_is_piece = cls == DOMINO;
                let mut me = if me_is_piece {
                    Ent {
                        tag: PIECE,
                        sums: [0; SLOTS],
                    }
                } else {
                    let ic = &self.classes[cls as usize];
                    let vc = &self.vclasses[ic.vclass];
                    let m = if touch { 2 } else { 0 };
                    for j in 0..SLOTS {
                        let v = vc.vals[j];
                        gain[j] += v + v * m / 10;
                    }
                    let free = ic.mask & tracked & !used;
                    used |= free.isolate_lowest_one(); // lowest free tracked member
                    Ent {
                        tag: (ic.vclass as u32) << 8 | if vc.adj_bonus { m as u32 } else { 0 },
                        sums: [0; SLOTS],
                    }
                };
                ents.push(me);
                let mut seen: u64 = 0;
                for &c in cells {
                    for &nb in &self.nbr[c] {
                        let o = owner[nb];
                        if o < 0
                            || o as usize == id
                            || seen & (1 << o) != 0
                            || ents[o as usize].is_piece() == me_is_piece
                        {
                            continue;
                        }
                        seen |= 1 << o;
                        let other = &mut ents[o as usize];
                        let (item, piece) = if me_is_piece {
                            (other, &mut me)
                        } else {
                            (&mut me, other)
                        };
                        let vc = &self.vclasses[item.vclass()];
                        for j in 0..SLOTS {
                            let (old, new) = (piece.sums[j], piece.sums[j] + vc.vals[j]);
                            gain[j] += new / 5 - old / 5;
                            piece.sums[j] = if self.nonneg[j] { new % 5 } else { new };
                        }
                        if vc.adj_bonus {
                            assert!(item.mult() < 0xFF, "too many adjacent pieces");
                            let m = item.mult();
                            for j in 0..SLOTS {
                                let v = vc.vals[j];
                                gain[j] += v * (m + 1) / 10 - v * m / 10;
                            }
                            item.tag += 1;
                        }
                    }
                }
                ents[id] = me;
            }
        }
        self.finish(pos, ents, used, gain, code, sc)
    }

    /// Per slot, the most B's entity can still gain over A's in the same position, whatever comes
    /// next (both see the same future adjacencies). None: A's can end up behind by any amount.
    pub fn penalty(&self, a: &Ent, b: &Ent) -> Option<Score> {
        if a.is_piece() {
            // future gain trunc((S + X) / 5) - trunc(S / 5): only the remainder matters where
            // nothing is negative (floor); otherwise see piece_penalty
            return Some(std::array::from_fn(|j| match (a.sums[j], b.sums[j]) {
                (x, y) if x == y => 0,
                (x, y) if self.nonneg[j] => i32::from(x < y),
                (x, y) => piece_penalty(x, y),
            }));
        }
        let (va, vb) = (&self.vclasses[a.vclass()], &self.vclasses[b.vclass()]);
        // pieces placed next to the item later gain trunc(.. + v ..): monotone in v
        if (0..SLOTS).any(|j| va.vals[j] < vb.vals[j]) {
            return None;
        }
        if !vb.adj_bonus && !va.adj_bonus {
            return Some([0; SLOTS]);
        }
        // the item's own future bonus for k more adjacent pieces
        let (ma, mb) = (a.mult(), b.mult());
        let mut pen = [0; SLOTS];
        for (j, p) in pen.iter_mut().enumerate().take(self.slots) {
            let (xa, xb) = (va.vals[j], vb.vals[j]);
            *p = if self.nonneg[j] {
                // floor((r + v k) / 10) with r = v m mod 10, and xa >= xb >= 0
                match (va.adj_bonus, vb.adj_bonus) {
                    (_, false) => 0,
                    (true, true) => i32::from(xb * mb % 10 > xa * ma % 10),
                    (false, true) => (xb * mb % 10 + xb * self.max_adjacency) / 10,
                }
            } else {
                let gain = |v: i32, adj: bool, m: i32, k: i32| {
                    if adj {
                        v * (m + k) / 10 - v * m / 10
                    } else {
                        0
                    }
                };
                (0..=self.max_adjacency)
                    .map(|k| gain(xb, vb.adj_bonus, mb, k) - gain(xa, va.adj_bonus, ma, k))
                    .max()
                    .unwrap_or(0)
                    .max(0)
            };
        }
        Some(pen)
    }

    /// Advances sc.work to the next undecided cell, drops entities that can change no more,
    /// renumbers the rest canonically and builds the child's key. Also returns the number of
    /// undecided cells left.
    fn finish(
        &self,
        from: usize,
        ents: Ents,
        used: u64,
        gain: Score,
        code: u32,
        sc: &mut Scratch,
    ) -> (Child, u32) {
        assert!(
            ents.len() <= MAX_OPEN,
            "too many open entities: board too large for this solver"
        );
        let (n, nbr) = (self.n, &self.nbr);
        let owner = &sc.work;
        let mut pos = from;
        while pos < n && owner[pos] != UNDEC {
            pos += 1;
        }
        let lo = if pos == n {
            n
        } else {
            pos.saturating_sub(self.lane)
        };
        let hi = n.min(pos + self.reach);
        let mut live: u64 = 0;
        let mut free = self.free_from[hi];
        for c in lo..hi {
            let o = owner[c];
            free += u32::from(c >= pos && o == UNDEC);
            if o >= 0 && live & (1 << o) == 0 && nbr[c].iter().any(|&nb| owner[nb] == UNDEC) {
                live |= 1 << o;
            }
        }
        let mut remap = [-1i8; MAX_OPEN + 1];
        let mut kept = Ents::new();
        for &o in &owner[lo..hi] {
            if o >= 0 && live & (1 << o) != 0 && remap[o as usize] < 0 {
                remap[o as usize] = kept.len() as i8;
                kept.push(ents[o as usize]);
            }
        }
        let mut key = Key::new();
        key.push(pos as u8);
        for c in lo..hi {
            let mut o = owner[c];
            if o >= 0 {
                o = if live & (1 << o) != 0 {
                    remap[o as usize]
                } else {
                    OCCUPIED
                };
            } else if o == EMPTY && c < pos && !nbr[c].iter().any(|&nb| owner[nb] == UNDEC) {
                o = OCCUPIED; // an empty cell only matters while a neighbour is undecided
            }
            key.push((o - OCCUPIED) as u8);
        }
        for e in &kept {
            key.push(if e.is_piece() { 0xFF } else { 0xFE });
        }
        (
            Child {
                key,
                ents: kept,
                used,
                gain,
                code,
            },
            free,
        )
    }
}

/// Cells outside the shape that touch it.
fn neighbour_cells(shape: &[[i32; 2]]) -> usize {
    let mut out: Vec<[i32; 2]> = Vec::new();
    for &[x, y] in shape {
        for n in [[x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1]] {
            if !shape.contains(&n) && !out.contains(&n) {
                out.push(n);
            }
        }
    }
    out.len()
}

/// Exact max over any future sum X of [t(b + X) - t(b)] - [t(a + X) - t(a)], t = trunc(/5):
/// the most a piece with sum b can still gain over one with sum a. With d = b - a, the first
/// bracket difference t(y + d) - t(y) peaks at ceil(d / 5) for d >= 0; for d < 0 it peaks at
/// -max(0, (|d| + 1) / 5 - 1), reached where y and y + d straddle zero (trunc's wide zero bucket).
fn piece_penalty(a: i32, b: i32) -> i32 {
    let (a, b) = (a as i64, b as i64);
    let d = b - a;
    let peak = if d >= 0 {
        (d + 4) / 5
    } else {
        -(((-d + 1) / 5 - 1).max(0))
    };
    (peak - (b / 5 - a / 5)) as i32
}

#[cfg(test)]
mod tests {
    use super::piece_penalty;

    #[test]
    fn piece_penalty_matches_brute_force() {
        let t = |x: i32| x / 5;
        for a in -60..=60 {
            for b in -60..=60 {
                let brute = (-400..=400)
                    .map(|x| (t(b + x) - t(b)) - (t(a + x) - t(a)))
                    .max()
                    .unwrap();
                assert_eq!(piece_penalty(a, b), brute, "a={a} b={b}");
            }
        }
    }
}
