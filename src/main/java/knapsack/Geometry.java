package knapsack;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Shape orientations and placement enumeration. Cells are indexed y * width + x. */
final class Geometry {

    /** cells sorted ascending, so cells[0] is the anchor (top-left-most cell). */
    record Placement(int item, int[] cells, boolean touch) {}

    private Geometry() {}

    /** Distinct normalized orientations among the 8 rotations/mirrors. Points are {x, y}, sorted by (y, x). */
    static List<int[][]> orientations(List<int[]> shape) {
        Map<String, int[][]> unique = new LinkedHashMap<>();
        for (int t = 0; t < 8; t++) {
            int[][] pts = new int[shape.size()][];
            for (int k = 0; k < pts.length; k++) {
                int x = shape.get(k)[0];
                int y = shape.get(k)[1];
                for (int r = 0; r < t % 4; r++) {
                    int tmp = x;
                    x = -y;
                    y = tmp;
                }
                if (t >= 4) x = -x;
                pts[k] = new int[] {x, y};
            }
            int minX = Arrays.stream(pts).mapToInt(p -> p[0]).min().orElseThrow();
            int minY = Arrays.stream(pts).mapToInt(p -> p[1]).min().orElseThrow();
            for (int[] p : pts) {
                p[0] -= minX;
                p[1] -= minY;
            }
            Arrays.sort(pts, Comparator.<int[]>comparingInt(p -> p[1]).thenComparingInt(p -> p[0]));
            unique.putIfAbsent(Arrays.deepToString(pts), pts);
        }
        return List.copyOf(unique.values());
    }

    /** All legal placements of one item: inside the board, not on blocked cells. */
    static List<Placement> placements(Problem p, int item) {
        int w = p.width();
        int h = p.height();
        boolean[] blocked = p.blockedMask();
        Problem.Item it = p.items().get(item);
        List<Placement> out = new ArrayList<>();
        for (int[][] o : orientations(it.shape())) {
            int ow = Arrays.stream(o).mapToInt(q -> q[0]).max().orElseThrow() + 1;
            int oh = Arrays.stream(o).mapToInt(q -> q[1]).max().orElseThrow() + 1;
            for (int dy = 0; dy + oh <= h; dy++) {
                for (int dx = 0; dx + ow <= w; dx++) {
                    int[] cells = new int[o.length];
                    boolean ok = true;
                    for (int k = 0; k < o.length && ok; k++) {
                        cells[k] = (o[k][1] + dy) * w + o[k][0] + dx;
                        ok = !blocked[cells[k]];
                    }
                    if (ok) out.add(new Placement(item, cells, touches(it.border(), cells, w, h)));
                }
            }
        }
        out.sort((a, b) -> Arrays.compare(a.cells(), b.cells())); // canonical order: same shape -> same list
        return out;
    }

    /** Orientation-independent shape identity. */
    static String shapeKey(List<int[]> shape) {
        return orientations(shape).stream().map(Arrays::deepToString).sorted().findFirst().orElseThrow();
    }

    static boolean touches(String border, int[] cells, int w, int h) {
        if (border == null) return false;
        for (int c : cells) {
            int x = c % w;
            int y = c / w;
            boolean hit = switch (border) {
                case "top" -> y == 0;
                case "bottom" -> y == h - 1;
                case "left" -> x == 0;
                case "right" -> x == w - 1;
                default -> throw new IllegalArgumentException("bad border " + border);
            };
            if (hit) return true;
        }
        return false;
    }

    /** 4-neighbours of a cell inside the board. */
    static List<Integer> neighbours(int c, int w, int h) {
        List<Integer> out = new ArrayList<>(4);
        int x = c % w;
        int y = c / w;
        if (x > 0) out.add(c - 1);
        if (x + 1 < w) out.add(c + 1);
        if (y > 0) out.add(c - w);
        if (y + 1 < h) out.add(c + w);
        return out;
    }
}
