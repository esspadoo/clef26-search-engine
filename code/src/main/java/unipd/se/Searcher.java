package unipd.se;

import org.apache.lucene.document.Document;
import org.apache.lucene.queryparser.simple.SimpleQueryParser;
import org.apache.lucene.search.similarities.BM25Similarity;
import unipd.se.model.QueryDoc;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.index.*;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.search.*;
import org.apache.lucene.store.Directory;

import java.io.IOException;
import java.util.*;

/**
 * Utility class for performing searches on a Lucene index of papers.
 * <p>
 * Each {@link QueryDoc} is searched against the index, and the top N matching
 * papers are returned for each query.
 * </p>
 */
public class Searcher {

    /** Shared analyzer for parsing queries. */
    private static final StandardAnalyzer ANALYZER = new StandardAnalyzer();

    /**
     * Searches the Lucene index for each query and returns the top results.
     * <p>
     * Currently, queries are matched against the "abstract" field of each paper.
     * </p>
     *
     * @param dir the Lucene {@link Directory} containing the index
     * @param queries the list of {@link QueryDoc} objects to search for
     * @return a map from query pubkey to a list of top-matching paper pubkeys
     * @throws IOException if an I/O error occurs reading the index
     * @throws ParseException if a query cannot be parsed
     */
    public static Map<String, List<String>> search(
            Directory dir,
            List<QueryDoc> queries
    ) throws IOException, ParseException {

        Map<String, List<String>> results = new HashMap<>();

        // Ensure the reader is closed properly
        try (IndexReader reader = DirectoryReader.open(dir)) {
            IndexSearcher searcher = new IndexSearcher(reader);

            // Use BM25 similarity
            searcher.setSimilarity(new BM25Similarity());

            // QueryParser with fields
            Map<String, Float> fields = new HashMap<>();
            fields.put("title", 1.0f);
            fields.put("abstract", 1.0f);
            SimpleQueryParser parser = new SimpleQueryParser(ANALYZER, fields);

            for (QueryDoc q : queries) {
                Query query = parser.parse(QueryParser.escape(q.text));
                TopDocs topDocs = searcher.search(query, 20);

                List<String> topIds = new ArrayList<>();
                for (ScoreDoc sd : topDocs.scoreDocs) {
                    Document doc = searcher.doc(sd.doc);
                    topIds.add(doc.get("pubkey")); // store retrieved paper pubkeys
                }

                results.put(q.index, topIds); // <- key is query.index
            }
        }

        return results;
    }
}