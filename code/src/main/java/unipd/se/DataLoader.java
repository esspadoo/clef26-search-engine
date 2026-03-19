package unipd.se;

import unipd.se.model.Paper;
import unipd.se.model.QueryDoc;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

/**
 * Utility class responsible for loading data from JSON files.
 * <p>
 * Uses Jackson {@link ObjectMapper} to deserialize JSON arrays
 * into Java objects (papers and queries).
 */
public final class DataLoader {

    /** Shared ObjectMapper instance (thread-safe after configuration). */
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private DataLoader() {}

    /**
     * Loads a list of {@link Paper} objects from a JSON file.
     *
     * @param path path to the JSON file
     * @return list of papers
     * @throws IOException if reading/parsing fails
     */
    public static List<Paper> loadPapers(String path) throws IOException {
        return Arrays.asList(MAPPER.readValue(new File(path), Paper[].class));
    }

    /**
     * Loads a list of queries from a JSON file.
     * <p>
     * Works for both {@link QueryDoc} and subclasses (e.g., ExpandedQueryDoc).
     *
     * @param path path to the JSON file
     * @param clazz class type of the query (QueryDoc or subclass)
     * @param <T> type extending QueryDoc
     * @return list of queries
     * @throws IOException if reading/parsing fails
     */
    public static <T extends QueryDoc> List<T> loadQueries(String path, Class<T[]> clazz)
            throws IOException {
        return Arrays.asList(MAPPER.readValue(new File(path), clazz));
    }
}