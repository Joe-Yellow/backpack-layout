//! Cell-by-cell sweep shared by every round of the exact solver. Cells are scanned lane by lane
//! along the board's shorter side.
//!
//! At each step the first undecided cell (pos) is left empty, starts a 1x2 piece, or anchors an item
//! placement (anchor = the placement's first cell in scan order), so each layout is generated once.
//! Item-piece adjacency is recorded when the later of the two is placed and its rewards are paid at
//! once. An entity leaves the state once it has no undecided neighbour: it can gain nothing more.
//! Identical items are merged into classes, so permutations of them are never explored.
//! WLOG two adjacent cells are never both left empty (a 1x2 piece there never lowers the score).
//!
//! The board side of a state is its key: pos, the cells in the window [pos - lane, pos + reach)
//! and the kind of each open entity. All states in one Pareto front share it, so cells are never
//! stored per state; they are decoded when expanding.
//!
//! Rewards are paid incrementally: floor((a + b) / q) = floor(a / q) + floor((a % q + b) / q), so an
//! entity only carries its remainder: item rem = (value * multiplier) % 10, piece rem = (sum of
//! adjacent item bases) % 5.

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

/// Step codes stored per search node (with its pos) to rebuild the layout.
pub const STEP_EMPTY: u32 = 0;
pub const STEP_PIECE_ALONG: u32 = 1; // cells pos, pos + 1
pub const STEP_PIECE_ACROSS: u32 = 2; // cells pos, pos + lane
pub const STEP_ANCHORED: u32 = 3; // + index into the anchored placements at pos

// Inline capacities fit the common case (keys are ~20-30 bytes, a handful of open entities);
// longer ones spill to the heap. Sized for memory: fronts dominate the footprint.
pub type Key = SmallVec<[u8; 32]>;
pub type Ents = SmallVec<[i32; 8]>;

/// Item entity: value << 5 | rem << 1 | adj_bonus (>= 0). 1x2 piece entity: -(rem + 1) (< 0).
#[inline]
fn pack_item(value: i32, adj: bool, rem: i32) -> i32 {
    (value << 5) | (rem << 1) | adj as i32
}
#[inline]
fn pack_piece(rem: i32) -> i32 {
    -(rem + 1)
}
#[inline]
pub fn is_piece(e: i32) -> bool {
    e < 0
}
#[inline]
pub fn item_value(e: i32) -> i32 {
    e >> 5
}
#[inline]
pub fn item_adj(e: i32) -> bool {
    e & 1 == 1
}
#[inline]
pub fn ent_rem(e: i32) -> i32 {
    if e < 0 {
        -e - 1
    } else {
        (e >> 1) & 15
    }
}

pub struct ItemClass {
    pub value: i32,
    pub adj_bonus: bool,
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
    pub gain: i32,
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
    pub root_key: Key,
    lane: usize,
    /// Cells at or beyond pos + reach can't be covered yet: they are UNDEC or BLOCKED.
    reach: usize,
    scan_to_cell: Vec<usize>,
    nbr: Vec<Vec<usize>>,
    by_anchor: Vec<Vec<Anchored>>,
    blocked_scan: Vec<bool>,
}

impl Sweep {
    pub fn new(p: &Problem) -> Sweep {
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

        let mut groups: Vec<Vec<usize>> = Vec::new();
        // identical items: same shape (up to rotation/mirror), value, border and bonus flag
        type ClassKey = (Vec<[i32; 2]>, i32, String, bool);
        let mut group_of: HashMap<ClassKey, usize> = HashMap::new();
        for (i, it) in p.items.iter().enumerate() {
            let key = (
                shape_key(&it.shape),
                it.value,
                format!("{:?}", it.border),
                it.adj_bonus,
            );
            let g = *group_of.entry(key).or_insert_with(|| {
                groups.push(Vec::new());
                groups.len() - 1
            });
            groups[g].push(i);
        }
        let mut classes = Vec::new();
        let mut by_anchor: Vec<Vec<Anchored>> = (0..n).map(|_| Vec::new()).collect();
        for members in groups {
            let rep = &p.items[members[0]];
            let mask = members.iter().fold(0u64, |m, &i| m | 1 << i);
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
                value: rep.value,
                adj_bonus: rep.adj_bonus,
                members,
                mask,
            });
        }
        let mut reach = lane + 1; // 1x2 pieces reach pos + lane
        for (s, list) in by_anchor.iter().enumerate() {
            for a in list {
                reach = reach.max(a.cells[a.cells.len() - 1] - s + 1);
            }
        }
        let blocked = p.blocked_mask();
        let blocked_scan: Vec<bool> = (0..n).map(|s| blocked[scan_to_cell[s]]).collect();

        let mut sweep = Sweep {
            n,
            classes,
            root_key: Key::new(),
            lane,
            reach,
            scan_to_cell,
            nbr,
            by_anchor,
            blocked_scan,
        };
        let mut sc = sweep.scratch();
        for s in 0..n {
            sc.work[s] = if sweep.blocked_scan[s] {
                BLOCKED
            } else {
                UNDEC
            };
        }
        sweep.root_key = sweep.finish(0, Ents::new(), 0, 0, STEP_EMPTY, &mut sc).key;
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
    /// other classes may be reused (relaxation).
    pub fn expand(
        &self,
        key: &[u8],
        ents: &[i32],
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
        let can_empty = !up_empty && !left_empty;
        let along = pos % lane + 1 < lane && base[pos + 1] == UNDEC;
        let across = pos + lane < n && base[pos + lane] == UNDEC;
        if can_empty {
            emit(self.place(
                pos, ents, used, DOMINO, None, false, tracked, STEP_EMPTY, sc,
            ));
        }
        if along {
            emit(self.place(
                pos,
                ents,
                used,
                DOMINO,
                Some(&[pos, pos + 1]),
                false,
                tracked,
                STEP_PIECE_ALONG,
                sc,
            ));
        }
        if across {
            emit(self.place(
                pos,
                ents,
                used,
                DOMINO,
                Some(&[pos, pos + lane]),
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
                emit(self.place(
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
        ents_in: &[i32],
        mut used: u64,
        cls: i32,
        cells: Option<&[usize]>,
        touch: bool,
        tracked: u64,
        code: u32,
        sc: &mut Scratch,
    ) -> Child {
        sc.work.copy_from_slice(&sc.base);
        let owner = &mut sc.work;
        let mut ents: Ents = Ents::from_slice(ents_in);
        let mut gain = 0;
        match cells {
            None => owner[pos] = EMPTY,
            Some(cells) => {
                let id = ents.len();
                for &c in cells {
                    owner[c] = id as i8;
                }
                let me_is_piece = cls == DOMINO;
                let mut me = if me_is_piece {
                    pack_piece(0)
                } else {
                    let ic = &self.classes[cls as usize];
                    let bonus = if touch { 2 * ic.value } else { 0 };
                    gain += ic.value + bonus / 10;
                    let free = ic.mask & tracked & !used;
                    used |= free.isolate_lowest_one(); // lowest free tracked member
                    pack_item(
                        ic.value,
                        ic.adj_bonus,
                        if ic.adj_bonus { bonus % 10 } else { 0 },
                    )
                };
                ents.push(me);
                let mut seen: u64 = 0;
                for &c in cells {
                    for &nb in &self.nbr[c] {
                        let o = owner[nb];
                        if o < 0
                            || o as usize == id
                            || seen & (1 << o) != 0
                            || is_piece(ents[o as usize]) == me_is_piece
                        {
                            continue;
                        }
                        seen |= 1 << o;
                        let other = ents[o as usize];
                        let (item, piece) = if me_is_piece {
                            (other, me)
                        } else {
                            (me, other)
                        };
                        let value = item_value(item);
                        let pr = ent_rem(piece) + value;
                        gain += pr / 5;
                        let new_piece = pack_piece(pr % 5);
                        let mut new_item = item;
                        if item_adj(item) {
                            let ir = ent_rem(item) + value;
                            gain += ir / 10;
                            new_item = pack_item(value, true, ir % 10);
                        }
                        if me_is_piece {
                            me = new_piece;
                            ents[o as usize] = new_item;
                        } else {
                            me = new_item;
                            ents[o as usize] = new_piece;
                        }
                    }
                }
                ents[id] = me;
            }
        }
        self.finish(pos, ents, used, gain, code, sc)
    }

    /// Advances sc.work to the next undecided cell, drops entities that can gain nothing more,
    /// renumbers the rest canonically and builds the child's key.
    fn finish(
        &self,
        from: usize,
        ents: Ents,
        used: u64,
        gain: i32,
        code: u32,
        sc: &mut Scratch,
    ) -> Child {
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
        for c in lo..hi {
            let o = owner[c];
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
        for &e in &kept {
            key.push(if is_piece(e) { 0xFE } else { 0xFF });
        }
        Child {
            key,
            ents: kept,
            used,
            gain,
            code,
        }
    }
}
