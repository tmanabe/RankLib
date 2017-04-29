package ciir.umass.edu.metric;

import ciir.umass.edu.learning.DataPoint;
import ciir.umass.edu.learning.RankList;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class TestAPIAScorer{
    @Test
    public void testLabels() {
        float[] expected0 = new float[]{0.0f};
        DataPoint dp0 = new DataPoint("0 qid:0 1:1");
        assertEquals(expected0.length, dp0.getLabels().length);
        assertEquals(expected0[0], dp0.getLabel(), 1.0e-6);
        assertEquals(expected0[0], dp0.getLabels()[0], 1.0e-6);
        float[] expected1 = new float[]{3.0f, 0.0f, 1.0f, 2.0f};
        DataPoint dp1 = new DataPoint("3,0,1,2 qid:0 1:1");
        assertEquals(expected1.length, dp1.getLabels().length);
        assertEquals(expected1[0], dp1.getLabel(), 1.0e-6);
        for(int i = 0; i < expected1.length; i++)
            assertEquals(expected1[i], dp1.getLabels()[i], 1.0e-6);
    }
    @Test
    public void testSetLabel() {
        float[] expected1 = new float[]{4.0f, 0.0f, 1.0f, 2.0f};
        DataPoint dp1 = new DataPoint("3,0,1,2 qid:0 1:1");
        dp1.setLabel(4.0f);
        assertEquals(expected1.length, dp1.getLabels().length);
        assertEquals(expected1[0], dp1.getLabel(), 1.0e-6);
        for(int i = 0; i < expected1.length; i++)
            assertEquals(expected1[i], dp1.getLabels()[i], 1.0e-6);
    }
    @Test
    public void testAPIAScorer() {
        APIAScorer apias = new APIAScorer();
        assertEquals(0, apias.k);
    }
    @Test
    public void testScore_single() {
        APIAScorer apias = new APIAScorer();
        RankList rl0 = new RankList();
        rl0.add(new DataPoint("2 qid:0 1:1"));
        rl0.add(new DataPoint("0 qid:0 1:1"));
        rl0.add(new DataPoint("1 qid:0 1:1"));
        rl0.add(new DataPoint("3 qid:0 1:1"));
        assertEquals(0.805556, apias.score(rl0), 1.0e-6);
    }
    @Test
    public void testScore() {
        APIAScorer apias = new APIAScorer();
        RankList rl0 = new RankList();
        rl0.add(new DataPoint("2,0,2 qid:0 1:1"));
        rl0.add(new DataPoint("0,0,1 qid:0 1:1"));
        rl0.add(new DataPoint("1,1,0 qid:0 1:1"));
        rl0.add(new DataPoint("3,2,0 qid:0 1:1"));
        float expect = (0.805556f + 0.416667f + 1.000000f) / 3;
        assertEquals(expect, apias.score(rl0), 1.0e-6);
    }
    @Test
    public void testScore_mean() {
        APIAScorer apias = new APIAScorer();
        List<RankList> rls = new ArrayList<>();
        RankList rl0 = new RankList();
        rl0.add(new DataPoint("2 qid:0 1:1"));
        rl0.add(new DataPoint("0 qid:0 1:1"));
        rl0.add(new DataPoint("1 qid:0 1:1"));
        rl0.add(new DataPoint("3 qid:0 1:1"));
        rls.add(rl0);
        RankList rl1 = new RankList();
        rl1.add(new DataPoint("2,0,2 qid:0 1:1"));
        rl1.add(new DataPoint("0,0,1 qid:0 1:1"));
        rl1.add(new DataPoint("1,1,0 qid:0 1:1"));
        rl1.add(new DataPoint("3,2,0 qid:0 1:1"));
        rls.add(rl1);
        float expect = 0.805556f / 2 + (0.805556f + 0.416667f + 1.000000f) / 6;
        assertEquals(expect, apias.score(rls), 1.0e-6);
    }
    @Test
    public void testName() {
        APIAScorer apias = new APIAScorer();
        assertEquals("MAPIA", apias.name());
    }
}
