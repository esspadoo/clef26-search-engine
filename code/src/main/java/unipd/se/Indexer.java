package unipd.se;

import org.apache.lucene.store.FSDirectory;
import unipd.se.model.Paper;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.*;
import org.apache.lucene.index.*;
import org.apache.lucene.store.Directory;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.List;

/**
 * Utility class for building a Lucene index from a collection of papers.
 * <p>
 * This class creates a persistent, on-disk index using {@link FSDirectory}.
 * Each paper is stored as a {@link Document} with the following fields:
 * <ul>
 *     <li>{@code pubkey} – unique identifier, stored but not tokenized</li>
 *     <li>{@code title} – paper title, tokenized for full-text search</li>
 *     <li>{@code abstract} – paper abstract, tokenized for full-text search</li>
 * </ul>
 * <p>
 * The index is stored in the {@code index/} folder under the project root.
 * A shared {@link StandardAnalyzer} is used, and documents are buffered in memory
 * to improve indexing performance.
 * </p>
 */
public class Indexer {

    /** Shared analyzer for tokenizing text fields. */
    private static final StandardAnalyzer ANALYZER = new StandardAnalyzer();

    /**
     * Builds a persistent Lucene index from the given list of {@link Paper} objects.
     * <p>
     * Each paper is converted into a Lucene {@link Document} with its {@code pubkey},
     * {@code title}, and {@code abstract} fields.
     * </p>
     *
     * @param papers the list of {@link Paper} objects to index
     * @return a {@link Directory} representing the persistent on-disk index
     * @throws IOException if an error occurs while writing the index to disk
     */
    public static Directory buildIndex(List<Paper> papers) throws IOException {
        Directory dir = FSDirectory.open(Paths.get("index"));

        IndexWriterConfig config = new IndexWriterConfig(ANALYZER);
        config.setOpenMode(IndexWriterConfig.OpenMode.CREATE);
        config.setRAMBufferSizeMB(256.0);

        try (IndexWriter writer = new IndexWriter(dir, config)) {
            for (Paper p : papers) {
                Document doc = new Document();
                doc.add(new StringField("pubkey", p.pubkey, Field.Store.YES));
                doc.add(new TextField("title", p.title, Field.Store.YES));
                doc.add(new TextField("abstract", p.abstractText, Field.Store.YES));

                writer.addDocument(doc);
            }
        }

        return dir;
    }
}