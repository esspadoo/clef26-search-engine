package unipd.se;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import unipd.se.model.ExpandedQueryDoc;
import unipd.se.model.Paper;

import java.io.File;
import java.util.List;
import java.util.Map;
import java.util.concurrent.*;

/**
 * Main entry point of the information retrieval pipeline.
 * This class loads the scientific paper collection and the query set,
 * evaluates previously re-ranked results
 * and stores the evaluation metrics in JSON format.
 *
 * @author RETRIX
 * @version 1.0
 * @since 1.0
 */
public class Main {

    /**
     * Runs the retrieval and evaluation pipeline.
     * If no arguments are provided, it returns error.
     * If a third argument is provided, it is interpreted as the path to a
     * file containing re-ranked results, which are loaded and evaluated
     * directly. Arguments to pass to evaluate a reranked results:
     * _ _ results/reranked_results.json
     *
     * @param args command-line arguments:
     *             args[0] = path to the paper collection JSON file,
     *             args[1] = path to the query JSON file,
     *             args[2] = optional path to a JSON file containing
     *             re-ranked results
     */
    static void main(String[] args) {
        int cores = Runtime.getRuntime().availableProcessors();
        System.out.println("Starting retrieval [cores=" + cores + "]");

        String papersPath  = args.length > 0 && !args[0].equals("_") ? args[0] : "code/data/collection_data.json";

        //per run dev_set EN
        String queriesPath = args.length > 1 && !args[1].equals("_") ? args[1] : "code/data/Dev_set/ENexpanded_queries_bge_largeDEV.json";
        //String queriesPath = args.length > 1 && !args[1].equals("_") ? args[1] : "code/data/Dev_set/expanded_queries_bge_large_frDEV_en.json";

        // If rerankedResultPath is provided, evaluate re-ranked results
        String rerankedPath = args.length > 2 ? args[2] : null;

        ObjectMapper mapper = new ObjectMapper();

        try (ExecutorService ioPool = Executors.newFixedThreadPool(2)) {
            // 1. Load data in parallel
            // Papers and queries are deserialized simultaneously using two I/O threads
            Future<List<Paper>> papersFuture =
                    ioPool.submit(() -> DataLoader.loadPapers(papersPath));
            Future<List<ExpandedQueryDoc>> queriesFuture =
                    ioPool.submit(() -> DataLoader.loadQueries(queriesPath, ExpandedQueryDoc[].class));

            List<Paper> papers             = papersFuture.get();
            List<ExpandedQueryDoc> queries = queriesFuture.get();

            Map<String, List<String>> results = null;

            if (rerankedPath != null && !rerankedPath.trim().isEmpty()) {
                // Mode 2: load re-ranked results from file
                File rerankedFile = new File(rerankedPath);

                // Check if file exists and is not empty
                if (!rerankedFile.exists()) {
                    System.err.println("Error: Reranked results file does not exist: " + rerankedPath);
                    return;
                }

                if (rerankedFile.length() == 0) {
                    System.err.println("Error: Reranked results file is empty: " + rerankedPath);
                    return;
                }

                System.out.println("Loading re-ranked results from: " + rerankedPath);
                try {
                    results = mapper.readValue(
                            rerankedFile,
                            new TypeReference<>() {}
                    );
                    System.out.println("Loaded results for " + results.size() + " queries.");
                } catch (Exception e) {
                    System.err.println("Error reading reranked results file: " + e.getMessage());
                    return;
                }
            } else {
                System.out.println("No reranked results file provided or path is empty. BM25 search will not be performed.");
                return; // Exit without performing BM25 search
            }

            // Ensure we have valid results before proceeding
            if (results == null || results.isEmpty()) {
                System.err.println("Error: No valid results to evaluate.");
                return;
            }

            // 3. Save evaluation metrics to a progressively named file
            String basePath = "results/evaluation_results_reranked";
            File file = new File(basePath + ".json");
            int counter = 1;
            while (file.exists()) {
                file = new File(basePath + "_" + counter++ + ".json");
            }
            Evaluator.evaluate(results, queries, file.getPath());

        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            System.err.println("Pipeline interrupted: " + e.getMessage());
        } catch (Exception e) {
            System.err.println("Error running IR pipeline: " + e.getMessage());
            e.printStackTrace();
        }
    }
}