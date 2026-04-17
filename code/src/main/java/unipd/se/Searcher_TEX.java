package unipd.se;

import org.apache.lucene.document.Document;
import org.apache.lucene.queryparser.simple.SimpleQueryParser;
import org.apache.lucene.search.similarities.BM25Similarity;
import unipd.se.model.ExpandedQueryDoc_TEX;
import unipd.se.model.QueryDoc;
import org.apache.lucene.index.*;
import org.apache.lucene.search.*;
import org.apache.lucene.store.Directory;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.*;

public class Searcher_TEX {

    private static final MyEnglishAnalyzer_TEX ANALYZER = new MyEnglishAnalyzer_TEX();
    private static final Pattern VENUE_PATTERN = Pattern.compile("\\bvenue:\\s*([^\\s]+)");
    private static final Pattern AUTHOR_PATTERN = Pattern.compile("\\bauthor:\\s*([^\\s]+(?:\\s+[^\\s]+)*)");

    private static final float ORIGINAL_TEXT_BOOST = 13.8f;
    private static final float EXPANDED_TEXT_BOOST = 19.8f;
    private static final float EXPANSION_TERMS_BOOST = 4.2f;

    public static Map<String, List<String>> search(
            Directory dir,
            List<? extends QueryDoc> queries,
            float titleBoost,
            int topK
    ) throws IOException {

        Map<String, Float> fields = new HashMap<>();
        fields.put("title", titleBoost);
        fields.put("abstract", 1.0f);

        try (IndexReader reader = DirectoryReader.open(dir)) {
            IndexSearcher searcher = new IndexSearcher(reader);
            searcher.setSimilarity(new BM25Similarity());

            int cores = Runtime.getRuntime().availableProcessors();

            try (ForkJoinPool pool = new ForkJoinPool(cores)) {
                return pool.submit(() ->
                        queries.parallelStream().collect(Collectors.toMap(
                                q -> q.index,
                                q -> searchSingle(q, searcher, fields, topK),
                                (a, b) -> a,
                                LinkedHashMap::new
                        ))
                ).get();

            } catch (InterruptedException | ExecutionException e) {
                throw new IOException("Search execution failed", e);
            }
        }
    }

    private static List<String> searchSingle(
            QueryDoc q,
            IndexSearcher searcher,
            Map<String, Float> fields,
            int topK
    ) {
        try {
            SimpleQueryParser parser = new SimpleQueryParser(ANALYZER, fields);
            Map<String, String> filters = new HashMap<>();
            Query finalQuery;

            if (q instanceof ExpandedQueryDoc_TEX eq) {
                // Costruisce la query pesata (Originale vs Espansione)
                finalQuery = buildWeightedExpandedQuery(eq, parser, filters);
            } else {
                String text = q.getSearchText();
                if (text == null || text.isBlank()) return Collections.emptyList();
                finalQuery = buildSimpleQuery(text, parser, filters);
            }

            // Applica filtri venue/author se presenti
            finalQuery = applyFilters(finalQuery, filters);

            TopDocs topDocs = searcher.search(finalQuery, topK);
            List<String> topIds = new ArrayList<>(topDocs.scoreDocs.length);
            for (ScoreDoc sd : topDocs.scoreDocs) {
                Document doc = searcher.storedFields().document(sd.doc);
                topIds.add(doc.get("pubkey"));
            }
            return topIds;

        } catch (IOException e) {
            System.err.println("Search failed for query " + q.index + ": " + e.getMessage());
            return Collections.emptyList();
        }
    }

    /**
     * Build a BooleanQuery where the original text and the expansion have different weights.
     */
    private static Query buildWeightedExpandedQuery(
            ExpandedQueryDoc_TEX eq,
            SimpleQueryParser parser,
            Map<String, String> filters
    ) {
        String original = eq.getOriginal();
        String expanded = eq.getExpanded();
        String expansion = eq.getExpTerms();

        BooleanQuery.Builder mainBuilder = new BooleanQuery.Builder();

        // 1. Parte Originale (MUST o SHOULD con alto Boost)
        if (original != null && !original.isBlank()) {
            String cleanOriginal = extractFilters(original, filters);
            Query originalQuery = parser.parse(cleanOriginal);
            // Boost 5.0: Le parole dell'utente sono il segnale principale
            mainBuilder.add(new BoostQuery(originalQuery, ORIGINAL_TEXT_BOOST), BooleanClause.Occur.SHOULD);
        }

        if (expanded != null && !expanded.isBlank()) {
            String cleanExpanded = extractFilters(expanded, filters);
            Query expandedQuery = parser.parse(cleanExpanded);
            mainBuilder.add(new BoostQuery(expandedQuery, EXPANDED_TEXT_BOOST), BooleanClause.Occur.SHOULD);
        }

        // 2. Parte Espansione (SHOULD con basso Boost)
        if (expansion != null && !expansion.isBlank()) {
            Query expansionQuery = parser.parse(expansion);
            // Boost 1.0: L'espansione aiuta solo se l'originale non è sufficiente
            mainBuilder.add(new BoostQuery(expansionQuery, EXPANSION_TERMS_BOOST), BooleanClause.Occur.SHOULD);
        }

        BooleanQuery bq = mainBuilder.build();
        // Fallback se entrambe le parti sono vuote
        return bq.clauses().isEmpty() ? parser.parse("") : bq;
    }

    private static Query buildSimpleQuery(String text, SimpleQueryParser parser, Map<String, String> filters) {
        String searchText = extractFilters(text, filters);
        return parser.parse(searchText);
    }

    private static String extractFilters(String text, Map<String, String> filters) {
        Matcher venueM = VENUE_PATTERN.matcher(text);
        if (venueM.find()) {
            filters.put("venue", venueM.group(1));
            text = text.replaceFirst("\\bvenue:\\s*\\S+", "").trim();
        }

        Matcher authorM = AUTHOR_PATTERN.matcher(text);
        if (authorM.find()) {
            filters.put("authors", authorM.group(1));
            text = text.replaceFirst("\\bauthor:\\s*([^\\s]+(?:\\s+[^\\s]+)*)", "").trim();
        }
        return text;
    }

    private static Query applyFilters(Query query, Map<String, String> filters) {
        if (filters.isEmpty()) return query;
        BooleanQuery.Builder bq = new BooleanQuery.Builder();
        bq.add(query, BooleanClause.Occur.MUST);
        for (Map.Entry<String, String> f : filters.entrySet()) {
            bq.add(new TermQuery(new Term(f.getKey(), f.getValue().toLowerCase())), BooleanClause.Occur.MUST);
        }
        return bq.build();
    }
}