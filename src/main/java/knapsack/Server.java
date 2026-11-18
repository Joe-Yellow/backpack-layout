package knapsack;

import com.google.gson.Gson;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;

/** Local web UI: serves the page, the built-in shapes and a solve endpoint. Listens on loopback only. */
final class Server {

    private static final int MAX_BODY_BYTES = 1 << 20;
    private static final Gson GSON = new Gson();
    private static final Object SOLVE_LOCK = new Object(); // the solver already uses every core

    record PlacedItem(int index, String name, String type, int value, long finalValue, List<int[]> cells) {}

    record PlacedPiece(List<int[]> cells, long value) {}

    record Solution(long total, double seconds, int rounds, List<PlacedItem> items, List<PlacedPiece> pieces) {}

    private Server() {}

    static void start(int port) throws IOException {
        HttpServer http = HttpServer.create(new InetSocketAddress(InetAddress.getLoopbackAddress(), port), 0);
        http.createContext("/", Server::page);
        http.createContext("/shapes", ex -> send(ex, 200, GSON.toJson(Problem.SHAPES)));
        http.createContext("/solve", Server::solve);
        http.setExecutor(Executors.newFixedThreadPool(4));
        http.start();
        System.out.println("UI ready: http://localhost:" + port + "  (Ctrl+C to stop)");
    }

    private static void page(HttpExchange ex) throws IOException {
        if (!ex.getRequestURI().getPath().equals("/")) {
            send(ex, 404, error("not found"));
            return;
        }
        try (InputStream in = Server.class.getResourceAsStream("/web/index.html")) {
            if (in == null) throw new IOException("index.html missing from classpath");
            byte[] html = in.readAllBytes();
            ex.getResponseHeaders().set("Content-Type", "text/html; charset=utf-8");
            ex.sendResponseHeaders(200, html.length);
            try (OutputStream out = ex.getResponseBody()) {
                out.write(html);
            }
        }
    }

    private static void solve(HttpExchange ex) throws IOException {
        if (!ex.getRequestMethod().equals("POST")) {
            send(ex, 405, error("POST only"));
            return;
        }
        byte[] body = ex.getRequestBody().readNBytes(MAX_BODY_BYTES + 1);
        if (body.length > MAX_BODY_BYTES) {
            send(ex, 413, error("request too large"));
            return;
        }
        try {
            Problem p = Problem.parse(new String(body, StandardCharsets.UTF_8));
            Solution sol;
            synchronized (SOLVE_LOCK) {
                sol = solve(p);
            }
            send(ex, 200, GSON.toJson(sol));
        } catch (IllegalArgumentException e) {
            send(ex, 400, error(e.getMessage()));
        } catch (OutOfMemoryError e) {
            send(ex, 500, error("out of memory: raise the heap, e.g. MAVEN_OPTS=-Xmx8g"));
        } catch (RuntimeException e) {
            e.printStackTrace();
            send(ex, 500, error("internal error, see server console"));
        }
    }

    private static Solution solve(Problem p) {
        long t0 = System.nanoTime();
        ExactSolver.Result r = ExactSolver.solve(p);
        double seconds = (System.nanoTime() - t0) / 1e9;
        Verifier.Score s = Verifier.evaluate(p, r.layout());
        if (s.total() != r.value()) throw new IllegalStateException("solver/verifier mismatch");
        int w = p.width();
        List<PlacedItem> items = new ArrayList<>();
        for (int i = 0; i < p.items().size(); i++) {
            int[] cells = r.layout().itemCells()[i];
            if (cells == null) continue;
            Problem.Item it = p.items().get(i);
            items.add(new PlacedItem(i, it.name(), it.type(), it.value(), s.itemValues()[i], xy(cells, w)));
        }
        List<PlacedPiece> pieces = new ArrayList<>();
        for (int j = 0; j < r.layout().dominoes().size(); j++) {
            pieces.add(new PlacedPiece(xy(r.layout().dominoes().get(j), w), s.dominoValues()[j]));
        }
        return new Solution(s.total(), seconds, r.rounds(), items, pieces);
    }

    private static List<int[]> xy(int[] cells, int w) {
        return Arrays.stream(cells).mapToObj(c -> new int[] {c % w, c / w}).toList();
    }

    private static String error(String msg) {
        return GSON.toJson(Map.of("error", msg == null ? "error" : msg));
    }

    private static void send(HttpExchange ex, int status, String json) throws IOException {
        byte[] bytes = json.getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().set("Content-Type", "application/json; charset=utf-8");
        ex.sendResponseHeaders(status, bytes.length);
        try (OutputStream out = ex.getResponseBody()) {
            out.write(bytes);
        }
    }
}
