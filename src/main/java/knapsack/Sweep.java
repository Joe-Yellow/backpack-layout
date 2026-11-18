package knapsack;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

/**
 * Cell-by-cell sweep shared by the exact and relaxed DPs. Cells are scanned lane by lane along the
 * board's shorter side.
 *
 * <p>At each step the first undecided cell (pos) is left empty, starts a 1x2 piece, or anchors an
 * item placement (anchor = the placement's first cell in scan order), so each layout is generated once.
 * Item-piece adjacency is recorded when the later of the two is placed, and its rewards are paid at
 * once (see Ent). An entity leaves the state once it has no undecided neighbour: it can gain nothing more.
 * Identical items are merged into classes, so permutations of them are never explored.
 * WLOG two adjacent cells are never both left empty (a 1x2 piece there never lowers the score).
 */
final class Sweep {

    static final int UNDEC = -1;
    static final int EMPTY = -2;
    static final int BLOCKED = -3;
    static final int OCCUPIED = -4; // taken by an entity that can no longer gain adjacencies
    static final int DOMINO = -1;

    record ItemClass(int value, boolean adjBonus, int[] members, long mask) {}

    private record Anchored(int cls, int[] cells, boolean touch) {}

    /**
     * Open entity. Rewards are paid incrementally: floor((a + b) / q) = floor(a / q) + floor((a % q + b) / q),
     * so only the remainder is carried. Item: value, adjBonus, rem = (value * multiplier) % 10.
     * 1x2 piece: value = DOMINO, rem = (sum of adjacent bases) % 5.
     */
    record Ent(int value, boolean adjBonus, int rem) {
        boolean isDomino() {
            return value == DOMINO;
        }
    }

    /** used = bitmask of item indices; within a class, members are always used in order. */
    record State(int pos, int[] owner, Ent[] ents, long used) {}

    /** cls = item class or DOMINO; cells == null means "leave pos empty". */
    record Step(long gain, State next, int cls, int[] cells) {}

    final int n;
    final List<ItemClass> classes = new ArrayList<>();
    private final int lane;
    private final int[] scanToCell;
    private final int[][] nbr;
    private final List<List<Anchored>> byAnchor = new ArrayList<>();
    private final State root;

    Sweep(Problem p) {
        if (p.items().size() > 63) throw new IllegalArgumentException("at most 63 items supported");
        int w = p.width();
        int h = p.height();
        n = w * h;
        boolean alongY = h <= w;
        lane = alongY ? h : w;
        scanToCell = new int[n];
        int[] cellToScan = new int[n];
        for (int s = 0; s < n; s++) {
            int c = alongY ? (s % lane) * w + s / lane : (s / lane) * w + s % lane;
            scanToCell[s] = c;
            cellToScan[c] = s;
        }
        nbr = new int[n][];
        for (int s = 0; s < n; s++) {
            nbr[s] = Geometry.neighbours(scanToCell[s], w, h).stream().mapToInt(c -> cellToScan[c]).toArray();
        }

        Map<String, List<Integer>> groups = new LinkedHashMap<>();
        for (int i = 0; i < p.items().size(); i++) {
            Problem.Item it = p.items().get(i);
            String key = Geometry.shapeKey(it.shape()) + "|" + it.value() + "|" + it.border() + "|" + it.adjBonus();
            groups.computeIfAbsent(key, k -> new ArrayList<>()).add(i);
        }
        for (int s = 0; s < n; s++) byAnchor.add(new ArrayList<>());
        for (List<Integer> members : groups.values()) {
            Problem.Item rep = p.items().get(members.get(0));
            int cls = classes.size();
            long mask = 0;
            for (int m : members) mask |= 1L << m;
            classes.add(new ItemClass(rep.value(), rep.adjBonus(), members.stream().mapToInt(Integer::intValue).toArray(), mask));
            for (Geometry.Placement pl : Geometry.placements(p, members.get(0))) {
                int[] sc = Arrays.stream(pl.cells()).map(c -> cellToScan[c]).sorted().toArray();
                byAnchor.get(sc[0]).add(new Anchored(cls, sc, pl.touch()));
            }
        }

        boolean[] blocked = p.blockedMask();
        int[] owner = new int[n];
        Arrays.fill(owner, UNDEC);
        IntStream.range(0, n).filter(s -> blocked[scanToCell[s]]).forEach(s -> owner[s] = BLOCKED);
        root = advance(0, owner, new Ent[0], 0L).next();
    }

    State root() {
        return root;
    }

    /**
     * Successor steps. Only items in the tracked mask are limited to one use each; other classes may be
     * placed any number of times (a relaxation) and are not recorded in State.used.
     */
    List<Step> steps(State s, long tracked) {
        int pos = s.pos();
        int[] owner = s.owner();
        List<Step> out = new ArrayList<>();
        boolean upEmpty = pos % lane > 0 && owner[pos - 1] == EMPTY;
        boolean leftEmpty = pos >= lane && owner[pos - lane] == EMPTY;
        if (!upEmpty && !leftEmpty) out.add(place(s, DOMINO, null, false, 0));
        if (pos % lane + 1 < lane && owner[pos + 1] == UNDEC) out.add(place(s, DOMINO, new int[] {pos, pos + 1}, false, 0));
        if (pos + lane < n && owner[pos + lane] == UNDEC) out.add(place(s, DOMINO, new int[] {pos, pos + lane}, false, 0));
        for (Anchored a : byAnchor.get(pos)) {
            long limited = classes.get(a.cls()).mask() & tracked;
            if (limited != 0 && (limited & ~s.used()) == 0) continue;
            boolean free = true;
            for (int c : a.cells()) free &= owner[c] == UNDEC;
            if (free) out.add(place(s, a.cls(), a.cells(), a.touch(), tracked));
        }
        return out;
    }

    private Step place(State s, int cls, int[] cells, boolean touch, long tracked) {
        int[] owner = s.owner().clone();
        Ent[] ents = Arrays.copyOf(s.ents(), s.ents().length + (cells == null ? 0 : 1));
        long used = s.used();
        long gain = 0;
        if (cells == null) {
            owner[s.pos()] = EMPTY;
        } else {
            int id = ents.length - 1;
            for (int c : cells) owner[c] = id;
            Ent me;
            if (cls == DOMINO) {
                me = new Ent(DOMINO, false, 0);
            } else {
                ItemClass ic = classes.get(cls);
                long bonus = touch ? 2L * ic.value() : 0;
                gain += ic.value() + bonus / 10;
                me = new Ent(ic.value(), ic.adjBonus(), ic.adjBonus() ? (int) (bonus % 10) : 0);
                used |= Long.lowestOneBit(ic.mask() & tracked & ~used);
            }
            boolean[] seen = new boolean[id];
            for (int c : cells) {
                for (int nb : nbr[c]) {
                    int o = owner[nb];
                    if (o < 0 || o == id || seen[o] || ents[o].isDomino() == me.isDomino()) continue;
                    seen[o] = true;
                    Ent item = me.isDomino() ? ents[o] : me;
                    Ent piece = me.isDomino() ? me : ents[o];
                    long pr = piece.rem() + item.value();
                    gain += pr / 5;
                    piece = new Ent(DOMINO, false, (int) (pr % 5));
                    if (item.adjBonus()) {
                        long ir = item.rem() + item.value();
                        gain += ir / 10;
                        item = new Ent(item.value(), true, (int) (ir % 10));
                    }
                    if (me.isDomino()) {
                        me = piece;
                        ents[o] = item;
                    } else {
                        me = item;
                        ents[o] = piece;
                    }
                }
            }
            ents[id] = me;
        }
        Step adv = advance(s.pos(), owner, ents, used);
        return new Step(gain, adv.next(), cls, cells);
    }

    /** Moves pos to the next undecided cell, drops entities that can gain nothing more, renumbers canonically. */
    private Step advance(int from, int[] owner, Ent[] ents, long used) {
        int pos = from;
        while (pos < n && owner[pos] != UNDEC) pos++;
        int lo = pos == n ? n : Math.max(0, pos - lane);
        boolean[] live = new boolean[ents.length];
        for (int c = lo; c < n; c++) {
            if (owner[c] < 0) continue;
            for (int nb : nbr[c]) live[owner[c]] |= owner[nb] == UNDEC;
        }
        int[] remap = new int[ents.length];
        Arrays.fill(remap, -1);
        List<Ent> kept = new ArrayList<>();
        for (int c = lo; c < n; c++) {
            int o = owner[c];
            if (o >= 0 && live[o] && remap[o] < 0) {
                remap[o] = kept.size();
                kept.add(ents[o]);
            }
        }
        for (int c = 0; c < n; c++) {
            if (owner[c] >= 0) owner[c] = c < lo ? EMPTY : live[owner[c]] ? remap[owner[c]] : OCCUPIED;
        }
        for (int c = lo; c < pos; c++) { // an empty cell only matters while a neighbour is undecided
            if (owner[c] != EMPTY) continue;
            boolean matters = false;
            for (int nb : nbr[c]) matters |= owner[nb] == UNDEC;
            if (!matters) owner[c] = OCCUPIED;
        }
        return new Step(0, new State(pos, owner, kept.toArray(new Ent[0]), used), 0, null);
    }

    /** Board-side key: window cells + kind of each open entity (values/remainders compared separately). */
    Key shapeKey(State s) {
        int from = Math.max(0, s.pos() - lane);
        int fields = (n - from) + s.ents().length;
        long[] packed = new long[(fields + FIELDS_PER_WORD - 1) / FIELDS_PER_WORD + 1];
        packed[packed.length - 1] = s.pos();
        int field = 0;
        for (int c = from; c < n; c++) pack(packed, field++, s.owner()[c] - OCCUPIED);
        for (Ent e : s.ents()) pack(packed, field++, e.isDomino() ? 1 : 2);
        return new Key(packed);
    }

    private static final int FIELDS_PER_WORD = 10; // 10 x 6 bits per long, no field straddles two words

    private static void pack(long[] packed, int field, int code) {
        if (code < 0 || code >= 64) throw new IllegalStateException("key field out of range: " + code);
        packed[field / FIELDS_PER_WORD] |= (long) code << (field % FIELDS_PER_WORD * 6);
    }

    /** Packed key with value equality. */
    record Key(long[] packed) {
        @Override
        public boolean equals(Object o) {
            return o instanceof Key k && Arrays.equals(packed, k.packed);
        }

        @Override
        public int hashCode() {
            return Arrays.hashCode(packed);
        }
    }

    /** Converts the chosen steps of one root-to-end path into a layout (board cell ids). */
    Verifier.Layout layout(List<Step> path, int itemCount) {
        int[][] itemCells = new int[itemCount][];
        int[] nextMember = new int[classes.size()];
        List<int[]> dominoes = new ArrayList<>();
        for (Step st : path) {
            if (st.cells() == null) continue;
            int[] cells = Arrays.stream(st.cells()).map(sc -> scanToCell[sc]).sorted().toArray();
            if (st.cls() == DOMINO) {
                dominoes.add(cells);
            } else {
                itemCells[classes.get(st.cls()).members()[nextMember[st.cls()]++]] = cells;
            }
        }
        return new Verifier.Layout(itemCells, dominoes);
    }
}
