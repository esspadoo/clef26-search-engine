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
 * This class uses Jackson's {@link ObjectMapper} to deserialize JSON arrays
 * into Java objects representing papers and queries.
 */
public class DataLoader {

    /**
     * Shared ObjectMapper instance to avoid repeated instantiation overhead.
     */
    private static final ObjectMapper MAPPER = new ObjectMapper();

    /**
     * Loads a list of {@link Paper} objects from a JSON file.
     * <p>
     * The input file is expected to contain a JSON array where each element
     * represents a paper.
     *
     * @param path the file system path to the JSON file containing papers
     * @return a list of {@link Paper} objects
     * @throws IOException if an error occurs while reading or parsing the file
     */
    public static List<Paper> loadPapers(String path) throws IOException {
        Paper[] papers = MAPPER.readValue(new File(path), Paper[].class);
        return Arrays.asList(papers);
    }


    /**
     * Loads a list of {@link QueryDoc} objects from a JSON file.
     * <p>
     * The input file is expected to contain a JSON array where each element
     * represents a query document.
     *
     * @param path the file system path to the JSON file containing queries
     * @return a list of {@link QueryDoc} objects
     * @throws IOException if an error occurs while reading or parsing the file
     */
    public static List<QueryDoc> loadQueries(String path) throws IOException {
        QueryDoc[] queries = MAPPER.readValue(new File(path), QueryDoc[].class);
        return Arrays.asList(queries);
    }
}