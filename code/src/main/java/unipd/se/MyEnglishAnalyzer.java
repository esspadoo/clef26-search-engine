package unipd.se;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardTokenizer;
import org.apache.lucene.analysis.core.LowerCaseFilter;
import org.apache.lucene.analysis.core.StopFilter;
import org.tartarus.snowball.ext.EnglishStemmer; // Snowball stemmer
import org.apache.lucene.analysis.snowball.SnowballFilter;
import org.apache.lucene.analysis.en.EnglishPossessiveFilter;
import org.apache.lucene.analysis.miscellaneous.ASCIIFoldingFilter;
import org.apache.lucene.analysis.pattern.PatternReplaceFilter;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.analysis.miscellaneous.TrimFilter;

import java.util.regex.Pattern;

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
public class MyEnglishAnalyzer extends Analyzer {

    // Remove '#' only at the beginning of tokens
    private static final Pattern HASHTAG_PATTERN = Pattern.compile("^#");

    @Override
    protected TokenStreamComponents createComponents(String fieldName) {

        Tokenizer tokenizer = new StandardTokenizer();
        TokenStream stream = tokenizer;

        // Trim tokens
        stream = new TrimFilter(stream);

        // Remove '#' from hashtags
        stream = new PatternReplaceFilter(stream, HASHTAG_PATTERN, "", true);

        // Normalize accents
        stream = new ASCIIFoldingFilter(stream);

        // Lowercase
        stream = new LowerCaseFilter(stream);

        // Remove possessives (e.g., doctor's → doctor)
        stream = new EnglishPossessiveFilter(stream);

        // Remove stopwords
        stream = new StopFilter(stream, EnglishAnalyzer.ENGLISH_STOP_WORDS_SET);

        // Apply stemming
        stream = new SnowballFilter(stream, new EnglishStemmer());

        return new TokenStreamComponents(tokenizer, stream);
    }
}