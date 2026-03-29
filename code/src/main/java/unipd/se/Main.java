package unipd.se;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import unipd.se.model.ExpandedQueryDoc;
import unipd.se.model.Paper;
import unipd.se.model.QueryDoc;
import org.apache.lucene.store.Directory;

import java.io.File;
import java.util.List;
import java.util.Map;
import java.util.concurrent.*;

/**
 * Entry point for the Information Retrieval pipeline.
 *
 * Modalità 1 — BM25 only (default):
 *   java Main
 *   → esegue retrieval BM25, salva results/bm25_results.json, valuta e salva metriche
 *
 * Modalità 2 — BM25 + re-ranking neurale:
 *   java Main [papersPath] [queriesPath] [rerankedResultsPath]
 *   → se rerankedResultsPath è fornito, valuta QUELLO invece dei risultati BM25
 *
 * Flusso completo consigliato:
 *   1. java Main                          → produce results/bm25_results.json
 *   2. python Reranker.py                 → produce results/reranked_results.json
 *   3. java Main _ _ results/reranked_results.json   → valuta il re-ranking
 */
public class Main {

    public static void main(String[] args) {
        int cores = Runtime.getRuntime().availableProcessors();
        System.out.println("Starting retrieval [cores=" + cores + "]");

        String papersPath  = args.length > 0 && !args[0].equals("_") ? args[0] : "code/data/collection_data.json";
        String queriesPath = args.length > 1 && !args[1].equals("_") ? args[1] : "code/data/expanded_queries_bge_large.json";

        // Se fornito il rerankedResultPath salta BM25 e valuta direttamente i risultati re-ranked
        String rerankedPath = args.length > 2 ? args[2] : null;

        ObjectMapper mapper = new ObjectMapper();

        try {
            // ── 1. Carica dati in parallelo ──────────────────────────────────────
            // Papers e queries vengono deserializzati contemporaneamente su due thread I/O
            ExecutorService ioPool = Executors.newFixedThreadPool(2);
            Future<List<Paper>> papersFuture =
                    ioPool.submit(() -> DataLoader.loadPapers(papersPath));
            Future<List<ExpandedQueryDoc>> queriesFuture =
                    ioPool.submit(() -> DataLoader.loadQueries(queriesPath, ExpandedQueryDoc[].class));
            ioPool.shutdown();

            List<Paper> papers             = papersFuture.get();
            List<ExpandedQueryDoc> queries = queriesFuture.get();

            Map<String, List<String>> results;

            if (rerankedPath != null) {
                // ── Modalità 2: leggi i risultati re-ranked da file ──────────────
                System.out.println("Loading re-ranked results from: " + rerankedPath);
                results = mapper.readValue(
                        new File(rerankedPath),
                        new TypeReference<Map<String, List<String>>>() {}
                );
                System.out.println("Loaded results for " + results.size() + " queries.");

            } else {
                // ── Modalità 1: esegui BM25 e salva ─────────────────────────────
                Directory index = Indexer.buildIndex(papers);

                // Ricerca parallela (Searcher gestisce internamente il parallelismo)
                results = Searcher.search(index, queries, 1.0f, 100);

                // Salva i risultati BM25 in background mentre il main prepara la config
                new File("results").mkdirs();
                File bm25File = new File("results/bm25_results.json");
                CompletableFuture<Void> saveFuture = CompletableFuture.runAsync(() -> {
                    try {
                        mapper.writerWithDefaultPrettyPrinter().writeValue(bm25File, results);
                        System.out.println("BM25 results saved to: " + bm25File.getPath());
                        System.out.println("→ Now run: python Reranker.py");
                        System.out.println("→ Then re-run Main with: java Main _ _ results/reranked_results.json");
                    } catch (Exception e) {
                        System.err.println("Failed to save BM25 results: " + e.getMessage());
                    }
                });

                // Config costruita mentre il file viene scritto in background
                ObjectNode config = mapper.createObjectNode();
                config.put("analyzer",     "MyCustomAnalyzer");
                config.put("query_parser", "SBERT");
                config.put("top_n",        100);
                config.put("title_boost",  1.0);
                config.put("similarity",   "BM25");
                config.put("reranker",     "none");
                config.putPOJO("fields",   new String[]{"title", "abstract"});

                // Assicura che il file sia stato scritto prima di procedere
                saveFuture.join();

                String basePath = "results/evaluation_results";
                File file = new File(basePath + ".json");
                int counter = 1;
                while (file.exists()) {
                    file = new File(basePath + "_" + counter++ + ".json");
                }
                Evaluator.evaluate(results, queries, config, file.getPath());
                return;
            }

            // ── 3. Configurazione (modalità reranked) ────────────────────────────
            ObjectNode config = mapper.createObjectNode();
            config.put("analyzer",     "MyCustomAnalyzer");
            config.put("query_parser", "SBERT");
            config.put("top_n",        100);
            config.put("title_boost",  1.0);
            config.put("similarity",   "BM25");
            config.put("reranker",     "cross-encoder/ms-marco-MiniLM-L-6-v2");
            config.putPOJO("fields",   new String[]{"title", "abstract"});

            // ── 4. Salva le metriche in un file con nome progressivo ─────────────
            String basePath = "results/evaluation_results_reranked";
            File file = new File(basePath + ".json");
            int counter = 1;
            while (file.exists()) {
                file = new File(basePath + "_" + counter++ + ".json");
            }
            Evaluator.evaluate(results, queries, config, file.getPath());

        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            System.err.println("Pipeline interrupted: " + e.getMessage());
        } catch (Exception e) {
            System.err.println("Error running IR pipeline: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
