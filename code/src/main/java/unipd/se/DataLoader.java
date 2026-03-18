package unipd.se;

import unipd.se.model.Paper;
import unipd.se.model.QueryDoc;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.util.Arrays;
import java.util.List;


public class DataLoader {

    public static List<Paper> loadPapers(String path) throws Exception {
        ObjectMapper mapper = new ObjectMapper();
        Paper[] papers = mapper.readValue(new File(path), Paper[].class);
        return Arrays.asList(papers);
    }

    public static List<QueryDoc> loadQueries(String path) throws Exception {
        ObjectMapper mapper = new ObjectMapper();
        QueryDoc[] queries = mapper.readValue(new File(path), QueryDoc[].class);
        return Arrays.asList(queries);
    }
}