package unipd.se;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.en.*;
import org.apache.lucene.analysis.standard.StandardTokenizer;
import org.apache.lucene.analysis.core.LowerCaseFilter;
import org.apache.lucene.analysis.core.StopFilter;
import org.apache.lucene.analysis.miscellaneous.ASCIIFoldingFilter;
import org.apache.lucene.analysis.pattern.PatternReplaceFilter;
import org.apache.lucene.analysis.miscellaneous.TrimFilter;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.regex.Pattern;

//Added to read custom stoplist instead of the ENGLISH_STOP_WORDS_SET default one
import org.apache.lucene.analysis.CharArraySet;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * Custom analyzer for social-post-like queries and scientific documents.
 * <p>
 * Features:
 * - Removes '#' from hashtags (#anxiety → anxiety)
 * - Lowercases text
 * - Removes stopwords
 * - Applies stemming (Snowball stemmer)
 * - Normalizes accents
 */
public class MyEnglishAnalyzer_V2 extends Analyzer {

    // Remove '#' only at the beginning of tokens
    private static final Pattern HASHTAG_PATTERN = Pattern.compile("^#");

    // regex per rimuovere menzioni (@)
    private static final Pattern MENTION_PATTERN = Pattern.compile("@\\w+");

    @Override
    protected TokenStreamComponents createComponents(String fieldName) {

        Tokenizer tokenizer = new StandardTokenizer();
        TokenStream stream = tokenizer;

        //included custom stopwords list
        CharArraySet stopWords = new CharArraySet(16, true);

        /*
        try {
            List<String> lines = Files.readAllLines(Path.of("../../../../../../code/data/stoplist_en_ranksnl_large.txt"));
            stopWords.addAll(lines);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }*/

        try {
            Path path = Paths.get("code", "data", "stoplist_en_ranksnl_large.txt");
            List<String> lines = Files.readAllLines(path);
            stopWords.addAll(lines);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        // Trim tokens
        stream = new TrimFilter(stream);

        // Remove '#' from hashtags
        stream = new PatternReplaceFilter(stream, HASHTAG_PATTERN, "", true);

        //DA TESTARE
        //Rimuovo menzioni
        //stream = new PatternReplaceFilter(stream, MENTION_PATTERN, "", true);

        // Normalize accents
        stream = new ASCIIFoldingFilter(stream);

        // Lowercase
        stream = new LowerCaseFilter(stream);

        // Remove possessives (e.g., doctor's → doctor)
        stream = new EnglishPossessiveFilter(stream);

        // Remove stopwords
        stream = new StopFilter(stream, stopWords);

        // Apply stemming
        stream = new KStemFilter(stream);
        return new TokenStreamComponents(tokenizer, stream);
    }
}