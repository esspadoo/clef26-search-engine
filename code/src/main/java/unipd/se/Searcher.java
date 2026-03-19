package unipd.se;

import org.apache.lucene.document.Document;
import org.apache.lucene.queryparser.simple.SimpleQueryParser;
import org.apache.lucene.search.similarities.BM25Similarity;
import unipd.se.model.QueryDoc;
import org.apache.lucene.index.*;
import org.apache.lucene.search.*;
import org.apache.lucene.store.Directory;

import java.io.IOException;
import java.util.*;

/**
 * Utility class for performing searches on a Lucene index of papers.
 * <p>
 * Supports both QueryDoc and subclasses (e.g., ExpandedQueryDoc).
 * Uses BM25 similarity and a weighted multi-field query (title + abstract).
 * </p>
 */
public class Searcher {

    /** Shared custom analyzer for parsing queries. */
    private static final MyEnglishAnalyzer ANALYZER = new MyEnglishAnalyzer();

    /**
     * Search the index with a configurable title boost.
     *
     * @param dir the Lucene index directory
     * @param queries list of queries (QueryDoc or subclasses)
     * @param titleBoost boost applied to the title field
     * @param topK number of top documents to retrieve
     * @return map from query index → ranked list of pubkeys
     */
    public static Map<String, List<String>> search(
            Directory dir,
            List<? extends QueryDoc> queries,
            float titleBoost,
            int topK
    ) throws IOException {

        Map<String, List<String>> results = new HashMap<>();

        try (IndexReader reader = DirectoryReader.open(dir)) {
            IndexSearcher searcher = new IndexSearcher(reader);

            // BM25 (default but explicit = good practice)
            searcher.setSimilarity(new BM25Similarity());

            // Field weights
            Map<String, Float> fields = new HashMap<>();
            fields.put("title", titleBoost);
            fields.put("abstract", 1.0f);

            SimpleQueryParser parser = new SimpleQueryParser(ANALYZER, fields);

            for (QueryDoc q : queries) {

                String text = q.getSearchText();

                if (text == null || text.isEmpty()) {
                    results.put(q.index, Collections.emptyList());
                    continue;
                }

                Query query = parser.parse(text);
                TopDocs topDocs = searcher.search(query, topK);

                List<String> topIds = new ArrayList<>(topDocs.scoreDocs.length);

                for (ScoreDoc sd : topDocs.scoreDocs) {
                    Document doc = searcher.doc(sd.doc);
                    topIds.add(doc.get("pubkey"));
                }

                results.put(q.index, topIds);
            }
        }

        return results;
    }

    /**
     * Extracts the correct query text depending on type.
     * Prefers expanded query if available.
     */
    private static String getQueryText(QueryDoc q) {
        try {
            // If ExpandedQueryDoc has "expanded" field
            return (String) q.getClass().getField("expanded").get(q);
        } catch (Exception e) {
            // fallback to original text
            return q.text;
        }
    }
}