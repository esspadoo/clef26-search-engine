package unipd.se;

import org.apache.lucene.store.Directory;
import unipd.se.model.QueryDoc;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.*;

/**
 * Integration guide and examples for using PRF with your existing pipeline.
 * This class shows how to add PRF to Main.java and other components.
 */
public class PRFIntegration {

    /**
     * Example 1: Simple replacement of BM25 search with PRF two-pass search
     */
    public static Map<String, List<String>> integrateSimplePRF(
            Directory index,
            List<? extends QueryDoc> queries,
            float titleBoost,
            int topK
    ) throws IOException {
        System.out.println("Running PRF-based search (two-pass)...");

        Map<String, List<String>> results = new LinkedHashMap<>();
        Map<String, Float> fields = new HashMap<>();
        fields.put("title", titleBoost);
        fields.put("abstract", 1.0f);

        for (QueryDoc q : queries) {
            try {
                String queryText = q.getSearchText();
                if (queryText == null || queryText.isEmpty()) {
                    results.put(q.index, Collections.emptyList());
                    continue;
                }

                PRFConfig.Config config = PRFConfig.BALANCED;

                Map<String, Object> prfResult = PseudoRelevanceFeedback.twoPassSearchWithPRF(
                        index, queryText, fields,
                        config.topKRelevant, config.topNTerms, topK
                );

                @SuppressWarnings("unchecked")
                List<String> docs = (List<String>) prfResult.get("pass2_docs");
                results.put(q.index, docs);

                @SuppressWarnings("unchecked")
                List<String> expandedTerms = (List<String>) prfResult.get("expanded_terms");
                System.out.println("Query " + q.index + ": expanded with " + expandedTerms.size() + " terms");

            } catch (IOException e) {
                System.err.println("PRF search failed for query " + q.index + ": " + e.getMessage());
                results.put(q.index, Collections.emptyList());
            }
        }

        return results;
    }

    /**
     * Example 2: Hybrid approach - combine BM25 and PRF results
     */
    public static Map<String, List<String>> hybridBM25andPRF(
            Directory index,
            List<? extends QueryDoc> queries,
            float titleBoost,
            int topK
    ) throws IOException {
        System.out.println("Running hybrid BM25 + PRF search...");

        Map<String, List<String>> results = new LinkedHashMap<>();
        Map<String, Float> fields = new HashMap<>();
        fields.put("title", titleBoost);
        fields.put("abstract", 1.0f);

        Map<String, List<String>> bm25Results = SearcherV2.search(index, queries, titleBoost, topK);

        for (QueryDoc q : queries) {
            try {
                String queryText = q.getSearchText();
                if (queryText == null || queryText.isEmpty()) {
                    results.put(q.index, bm25Results.getOrDefault(q.index, Collections.emptyList()));
                    continue;
                }

                PRFConfig.Config config = PRFConfig.BALANCED;
                Map<String, Object> prfResult = PseudoRelevanceFeedback.twoPassSearchWithPRF(
                        index, queryText, fields,
                        config.topKRelevant, config.topNTerms, topK
                );

                @SuppressWarnings("unchecked")
                List<String> prfDocs = (List<String>) prfResult.get("pass2_docs");
                List<String> bm25Docs = bm25Results.getOrDefault(q.index, Collections.emptyList());

                LinkedHashSet<String> merged = new LinkedHashSet<>(bm25Docs);
                merged.addAll(prfDocs);

                results.put(q.index, new ArrayList<>(merged));

                System.out.println("Query " + q.index + ": BM25=" + bm25Docs.size() +
                                 ", PRF=" + prfDocs.size() + ", merged=" + merged.size());

            } catch (IOException e) {
                System.err.println("Error for query " + q.index + ": " + e.getMessage());
                results.put(q.index, bm25Results.getOrDefault(q.index, Collections.emptyList()));
            }
        }

        return results;
    }

    /**
     * Example 3: Configurable PRF with different strategies for different queries
     */
    public static Map<String, List<String>> adaptivePRF(
            Directory index,
            List<? extends QueryDoc> queries,
            float titleBoost,
            int topK
    ) throws IOException {
        System.out.println("Running adaptive PRF search...");

        Map<String, List<String>> results = new LinkedHashMap<>();
        Map<String, Float> fields = new HashMap<>();
        fields.put("title", titleBoost);
        fields.put("abstract", 1.0f);

        for (QueryDoc q : queries) {
            try {
                String queryText = q.getSearchText();
                if (queryText == null || queryText.isEmpty()) {
                    results.put(q.index, Collections.emptyList());
                    continue;
                }

                PRFConfig.Config config = selectConfig(queryText);

                Map<String, Object> prfResult = PseudoRelevanceFeedback.twoPassSearchWithPRF(
                        index, queryText, fields,
                        config.topKRelevant, config.topNTerms, topK
                );

                @SuppressWarnings("unchecked")
                List<String> docs = (List<String>) prfResult.get("pass2_docs");
                results.put(q.index, docs);

                System.out.println("Query " + q.index + " (" + config.name + "): " + docs.size() + " results");

            } catch (IOException e) {
                System.err.println("Error for query " + q.index + ": " + e.getMessage());
                results.put(q.index, Collections.emptyList());
            }
        }

        return results;
    }

    /**
     * Helper: Select PRF configuration based on query characteristics
     */
    private static PRFConfig.Config selectConfig(String query) {
        int wordCount = query.trim().split("\\s+").length;

        if (wordCount <= 2) {
            return PRFConfig.SHORT_QUERY;
        } else if (wordCount >= 5) {
            return PRFConfig.LONG_QUERY;
        } else if (query.toLowerCase().contains("vaccine") ||
                   query.toLowerCase().contains("covid") ||
                   query.toLowerCase().contains("medical") ||
                   query.toLowerCase().contains("disease")) {
            return PRFConfig.SCIENTIFIC_MEDICAL;
        } else {
            return PRFConfig.BALANCED;
        }
    }

    /**
     * Example 4: Parallel PRF execution for multiple queries
     */
    public static Map<String, List<String>> parallelPRF(
            Directory index,
            List<? extends QueryDoc> queries,
            float titleBoost,
            int topK
    ) throws IOException {
        System.out.println("Running parallel PRF search...");

        Map<String, Float> fields = new HashMap<>();
        fields.put("title", titleBoost);
        fields.put("abstract", 1.0f);

        PRFConfig.Config config = PRFConfig.BALANCED;

        int cores = Runtime.getRuntime().availableProcessors();
        ForkJoinPool pool = new ForkJoinPool(cores);

        try {
            return pool.submit(() ->
                    queries.parallelStream().collect(
                            () -> new LinkedHashMap<String, List<String>>(),
                            (map, q) -> {
                                try {
                                    String queryText = q.getSearchText();
                                    if (queryText == null || queryText.isEmpty()) {
                                        map.put(q.index, Collections.emptyList());
                                        return;
                                    }

                                    Map<String, Object> prfResult =
                                            PseudoRelevanceFeedback.twoPassSearchWithPRF(
                                                    index, queryText, fields,
                                                    config.topKRelevant, config.topNTerms, topK
                                            );

                                    @SuppressWarnings("unchecked")
                                    List<String> docs = (List<String>) prfResult.get("pass2_docs");
                                    map.put(q.index, docs);
                                } catch (IOException e) {
                                    System.err.println("Error for query " + q.index + ": " + e.getMessage());
                                    map.put(q.index, Collections.emptyList());
                                }
                            },
                            (map1, map2) -> map1.putAll(map2)
                    )
            ).get();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("Parallel search interrupted", e);
        } catch (ExecutionException e) {
            throw new IOException("Parallel search failed", e.getCause());
        } finally {
            pool.shutdown();
        }
    }

    /**
     * Example 5: How to modify Main.java to use PRF
     *
     * Option A - Use PRF instead of BM25:
     *   results = PRFIntegration.integrateSimplePRF(index, queries, 1.0f, 100);
     *
     * Option B - Hybrid approach:
     *   results = PRFIntegration.hybridBM25andPRF(index, queries, 1.0f, 100);
     *
     * Option C - Adaptive approach:
     *   results = PRFIntegration.adaptivePRF(index, queries, 1.0f, 100);
     *
     * Option D - Parallel approach:
     *   results = PRFIntegration.parallelPRF(index, queries, 1.0f, 100);
     */
    public static void main(String[] args) {
        System.out.println("PRF Integration Examples");
        System.out.println("========================");
        System.out.println();
        System.out.println("This class shows how to integrate PRF into your search pipeline.");
        System.out.println();
        System.out.println("Usage patterns:");
        System.out.println("  1. Simple PRF: integrateSimplePRF(index, queries, boost, topK)");
        System.out.println("  2. Hybrid: hybridBM25andPRF(index, queries, boost, topK)");
        System.out.println("  3. Adaptive: adaptivePRF(index, queries, boost, topK)");
        System.out.println("  4. Parallel: parallelPRF(index, queries, boost, topK)");
        System.out.println();
        System.out.println("See javadoc for detailed examples and integration steps.");
    }
}
