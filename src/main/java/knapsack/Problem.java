package knapsack;

import com.google.gson.Gson;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** Input instance. Coordinates are [x, y], origin top-left. */
public record Problem(int width, int height, List<int[]> blocked, List<Item> items) {

    /**
     * Give either type (a name from SHAPES) or shape (cell list). border: null | top | bottom | left | right.
     */
    public record Item(String name, int value, List<int[]> shape, String border, boolean adjBonus, String type) {
        public Item(String name, int value, List<int[]> shape, String border, boolean adjBonus) {
            this(name, value, shape, border, adjBonus, null);
        }
    }

    static final Set<String> BORDERS = Set.of("top", "bottom", "left", "right");

    /** Built-in shapes, usable via "type" (insertion-ordered for the UI palette). */
    static final Map<String, List<int[]>> SHAPES;

    static {
        Map<String, List<int[]>> m = new LinkedHashMap<>();
        m.put("U", cells(0, 0, 1, 0, 2, 0, 0, 1, 2, 1));
        m.put("F", cells(0, 0, 1, 0, 2, 0, 1, 1, 2, 1)); // drawn lying flat; rotations are free anyway
        m.put("L", cells(0, 0, 1, 0, 2, 0, 0, 1));
        m.put("T", cells(0, 0, 1, 0, 2, 0, 1, 1));
        m.put("O", cells(0, 0, 1, 0, 0, 1, 1, 1));
        m.put("V", cells(0, 0, 1, 0, 0, 1));
        SHAPES = Collections.unmodifiableMap(m);
    }

    public static Problem load(Path path) throws IOException {
        return parse(Files.readString(path));
    }

    /** Parses and validates a JSON instance; throws IllegalArgumentException on bad input. */
    public static Problem parse(String json) {
        Problem p;
        try {
            p = new Gson().fromJson(json, Problem.class);
        } catch (com.google.gson.JsonParseException e) {
            throw new IllegalArgumentException("malformed JSON: " + e.getMessage());
        }
        if (p == null) throw new IllegalArgumentException("empty input");
        return p.validated();
    }

    /** Validates and fills defaults. Values must be >= 0: the solver's dominance rules assume every bonus is non-negative. */
    Problem validated() {
        require(width > 0 && height > 0, "width/height must be positive");
        List<int[]> bl = blocked == null ? List.of() : blocked;
        for (int[] c : bl) {
            require(c != null && c.length == 2 && c[0] >= 0 && c[0] < width && c[1] >= 0 && c[1] < height,
                    "blocked cell " + Arrays.toString(c) + " out of board: cells are [x, y] with x in 0.."
                            + (width - 1) + ", y in 0.." + (height - 1));
        }
        require(items != null, "items missing");
        List<Item> resolved = new ArrayList<>();
        for (Item raw : items) {
            require(raw != null && raw.name() != null, "item name missing");
            Item it = withShape(raw);
            resolved.add(it);
            require(it.value() >= 0, it.name() + ": value must be >= 0");
            require(it.shape() != null && !it.shape().isEmpty(), it.name() + ": needs a type or a non-empty shape");
            Set<List<Integer>> seen = new HashSet<>();
            for (int[] c : it.shape()) {
                require(c != null && c.length == 2 && seen.add(List.of(c[0], c[1])),
                        it.name() + ": malformed or duplicate shape cell");
            }
            require(it.border() == null || BORDERS.contains(it.border()),
                    it.name() + ": border must be one of " + BORDERS);
        }
        return new Problem(width, height, List.copyOf(bl), List.copyOf(resolved));
    }

    private static Item withShape(Item it) {
        require(it.type() == null || it.shape() == null, it.name() + ": give either type or shape, not both");
        if (it.type() == null) return it;
        List<int[]> shape = SHAPES.get(it.type());
        require(shape != null, it.name() + ": unknown type '" + it.type() + "', expected one of " + SHAPES.keySet());
        return new Item(it.name(), it.value(), shape, it.border(), it.adjBonus(), it.type());
    }

    private static List<int[]> cells(int... xy) {
        List<int[]> out = new ArrayList<>();
        for (int i = 0; i < xy.length; i += 2) out.add(new int[] {xy[i], xy[i + 1]});
        return List.copyOf(out);
    }

    boolean[] blockedMask() {
        boolean[] mask = new boolean[width * height];
        for (int[] c : blocked) mask[c[1] * width + c[0]] = true;
        return mask;
    }

    private static void require(boolean ok, String msg) {
        if (!ok) throw new IllegalArgumentException(msg);
    }
}
