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

/**
 * Entry point for the information retrieval pipeline with optional re-ranking.
 *
 * Default mode:
 * java MainRerank
 * Runs BM25 retrieval, saves the results to {@code results/bm25_results.json},
 * and evaluates them.
 *
 * Re-ranking evaluation mode:
 * java MainRerank [papersPath] [queriesPath] [rerankedResultsPath]
 * If {@code rerankedResultsPath} is provided, the program loads and evaluates
 * those re-ranked results instead of running BM25 retrieval again.
 *
 * Recommended workflow:
 * 1. Run {@code java MainRerank} to generate BM25 results.
 * 2. Run the Python reranker to produce re-ranked results.
 * 3. Run {@code java MainRerank _ _ results/reranked_results.json}
 *    to evaluate the re-ranked output.
 *
 * @author RETRIX
 * @version 1.0
 * @since 1.0
 */
public class MainRerank {

    /**
     * Runs the retrieval and evaluation pipeline.
     * The method loads the paper collection and the expanded queries, then either
     * executes BM25 retrieval or loads previously re-ranked results from a file.
     * Finally, it builds the evaluation configuration and saves the computed
     * metrics to a JSON file.
     *
     * @param args command-line arguments:
     *             args[0] = path to the paper collection JSON file,
     *             args[1] = path to the expanded queries JSON file,
     *             args[2] = optional path to a JSON file containing
     *             re-ranked results
     */
    public static void main(String[] args) {
        System.out.println("Starting retrieval");

        // Ternary operator: "?" acts as an inline if-else
        String papersPath = args.length > 0 && !args[0].equals("_") ? args[0] : "code/data/collection_data.json";
        String queriesPath = args.length > 1 && !args[1].equals("_") ? args[1] : "code/data/expanded_queries_4.json";

        // If rerankedResultsPath is provided, skip BM25 and directly evaluate the re-ranked results
        String rerankedPath = args.length > 2 ? args[2] : null;

        ObjectMapper mapper = new ObjectMapper();

        try {
            // 1. Load data
            List<Paper> papers = DataLoader.loadPapers(papersPath);
            List<ExpandedQueryDoc> queries = DataLoader.loadQueries(queriesPath, ExpandedQueryDoc[].class);

            Map<String, List<String>> results;

            if (rerankedPath != null) {
                // Mode 2: load re-ranked results from file
                System.out.println("Loading re-ranked results from: " + rerankedPath);
                results = mapper.readValue(
                        new File(rerankedPath),
                        new TypeReference<Map<String, List<String>>>() {}
                );
                System.out.println("Loaded results for " + results.size() + " queries.");

            } else {
                // Mode 1: run BM25 and save the results
                Directory index = Indexer.buildIndex(papers);
                results = Searcher.search(index, queries, 1.0f, 100);

                // Save BM25 results to JSON (input for Reranker.py)
                new File("results").mkdirs();
                File bm25File = new File("results/bm25_results.json");
                mapper.writerWithDefaultPrettyPrinter().writeValue(bm25File, results);
                System.out.println("BM25 results saved to: " + bm25File.getPath());
                System.out.println("→ Now run: python Reranker.py");
                System.out.println("→ Then re-run Main with: java Main _ _ results/reranked_results.json");
            }


            // 2. Print top results (optional, useful for debugging)
            /*
            int printed = 0;
            for (Map.Entry<String, List<String>> entry : results.entrySet()) {
                System.out.println("Query: " + entry.getKey());
                System.out.println("Top docs: " + entry.getValue());
                System.out.println();
                if (++printed >= 3) break; // stampa solo le prime 3 per brevità
            }
            */

            // 3. Configuration, updated to reflect re-ranking
            ObjectNode config = mapper.createObjectNode();
            config.put("analyzer",      "MyCustomAnalyzer");
            config.put("query_parser",  "SBERT");
            config.put("top_n",         100);
            config.put("title_boost",   1.0);
            config.put("similarity",    "BM25");
            config.put("reranker",      rerankedPath != null ? "cross-encoder/ms-marco-MiniLM-L-6-v2" : "none");
            config.putPOJO("fields",    new String[]{"title", "abstract"});

            // 4. Save evaluation metrics to a progressively named file
            String basePath = "results/evaluation_results" + (rerankedPath != null ? "_reranked" : "");
            String extension = ".json";
            File file = new File(basePath + extension);
            int counter = 1;
            while (file.exists()) {
                file = new File(basePath + "_" + counter + extension);
                counter++;
            }

            Evaluator.evaluate(results, queries, config, file.getPath());

        } catch (Exception e) {
            System.err.println("Error running IR pipeline: " + e.getMessage());
            e.printStackTrace();
        }
    }
}