package ciir.umass.edu.learning;

import ciir.umass.edu.metric.APIAScorer;
import org.junit.Test;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.*;

public class TestSpanF {
    @Test
    public void testGetLastFeature() {
        DataPoint dp0 = new DataPoint("0 qid:0 1:0 #did:0");
        assertEquals(1, dp0.getLastFeature());
        DataPoint dp1 = new DataPoint("0 qid:0 1:0 2:0 #did:0");
        assertEquals(2, dp1.getLastFeature());
    }
    @Test
    public void testGetK() {
        DataPoint dp0 = new DataPoint("0 qid:0 1:7 #did:0");
        assertEquals(7, (int)dp0.getFeatureValue(1));
        DataPoint dp1 = new DataPoint("0 qid:0 1:7.0 #did:0");
        assertEquals(7, (int)dp0.getFeatureValue(1));
    }
    private void _testInit(double[] expected_ws, int expected_f, SpanF spanf) {
        assertEquals(expected_f, spanf.f);
        assertEquals(expected_ws.length, spanf.weight.length);
        for(int i = 0; i < expected_ws.length; i++)
            assertEquals(expected_ws[i], spanf.weight[i], 1.0e-6);
    }
    @Test
    public void testInitWithoutSample() {
        List<RankList> samples = new ArrayList<>();
        SpanF spanf = new SpanF(samples, null);
        spanf.init();
        double[] expect = {1.2};
        _testInit(expect, 0, spanf);
    }
    @Test
    public void testInit_k1f1() {
        List<RankList> samples = new ArrayList<>();
        DataPoint dp0 = new DataPoint("0 qid:0 1:1 2:-1.0 3:1.0 4:7 #did=0");
        RankList rl0 = new RankList();
        rl0.add(dp0);
        samples.add(rl0);
        SpanF spanf = new SpanF(samples, null);
        spanf.init();
        double[] expect = {1.2, 0.75, 1.0};
        _testInit(expect, 1, spanf);
    }
    @Test
    public void testInit_k2f2() {
        List<RankList> samples = new ArrayList<>();
        StringBuilder sb = new StringBuilder();
        sb.append("0 qid:0 ");
        sb.append("1:2 2:-1.0 3:0.0 ");
        sb.append("4:1.0 5:1 6:2 ");
        sb.append("7:2.0 8:3 9:4 ");
        sb.append("#did=0");
        DataPoint dp0 = new DataPoint(sb.toString());
        RankList rl0 = new RankList();
        rl0.add(dp0);
        samples.add(rl0);
        SpanF spanf = new SpanF(samples, null);
        spanf.init();
        double[] expect = {1.2, 0.75, 0.75, 1.0, 1.0};
        _testInit(expect, 2, spanf);
    }
    @Test
    public void testWeightNameOf() {
        List<RankList> samples = new ArrayList<>();
        DataPoint dp0 = new DataPoint("0 qid:0 1:1 2:-1.0 3:1.0 4:7 #did=0");
        RankList rl0 = new RankList();
        rl0.add(dp0);
        samples.add(rl0);
        SpanF spanf = new SpanF(samples, null);
        spanf.init();
        assertEquals("k1", spanf.weightNameOf(0));
        assertEquals("b(1)", spanf.weightNameOf(1));
        assertEquals("boost(1)", spanf.weightNameOf(2));
    }
    @Test
    public void testLearn_k1() {
        List<RankList> samples = new ArrayList<>();
        StringBuilder sb = new StringBuilder();
        sb.append("1 qid:0 ");
        sb.append("1:2 2:1.0 3:1.0 ");
        sb.append("4:1.0 5:1 6:1 ");
        sb.append("#did=0");
        DataPoint dp0 = new DataPoint(sb.toString());
        sb = new StringBuilder();
        sb.append("0 qid:0 ");
        sb.append("1:2 2:1.0 3:1.0 ");
        sb.append("4:1.0 5:0 6:100 ");
        sb.append("#did=1");
        DataPoint dp1 = new DataPoint(sb.toString());
        RankList rl0 = new RankList();
        rl0.add(dp0);
        rl0.add(dp1);
        samples.add(rl0);
        SpanF.verbose = false;
        SpanF spanf = new SpanF(samples, null) {
            @Override
            public int[] getShuffledWeights() {
                return new int[]{0, 1, 2};
            }
        };
        spanf.set(new APIAScorer());
        spanf.init();
        spanf.learn();
        assertTrue(spanf.weight[0] < 1.2);
    }
    @Test
    public void testLearn_boost() {
        List<RankList> samples = new ArrayList<>();
        StringBuilder sb = new StringBuilder();
        sb.append("1 qid:0 ");
        sb.append("1:2 2:1.0 3:1.0 ");
        sb.append("4:1.0 5:1 6:1 ");
        sb.append("#did=0");
        DataPoint dp0 = new DataPoint(sb.toString());
        sb = new StringBuilder();
        sb.append("0 qid:0 ");
        sb.append("1:2 2:1.0 3:1.0 ");
        sb.append("4:1.0 5:0 6:100 ");
        sb.append("#did=1");
        DataPoint dp1 = new DataPoint(sb.toString());
        RankList rl0 = new RankList();
        rl0.add(dp0);
        rl0.add(dp1);
        samples.add(rl0);
        SpanF.verbose = false;
        SpanF spanf = new SpanF(samples, null) {
            @Override
            public int[] getShuffledWeights() {
                return new int[]{2, 0, 1};
            }
        };
        spanf.set(new APIAScorer());
        spanf.init();
        spanf.learn();
        assertTrue(1.0 < spanf.weight[2]);
    }
    @Test
    public void testLearn_b() {
        List<RankList> samples = new ArrayList<>();
        StringBuilder sb = new StringBuilder();
        sb.append("1 qid:0 ");
        sb.append("1:2 2:1.0 3:1.0 ");
        sb.append("4:1.0 5:1 6:1 ");
        sb.append("#did=0");
        DataPoint dp0 = new DataPoint(sb.toString());
        sb = new StringBuilder();
        sb.append("0 qid:0 ");
        sb.append("1:2 2:1.0 3:1.0 ");
        sb.append("4:10.0 5:0 6:100 ");
        sb.append("#did=1");
        DataPoint dp1 = new DataPoint(sb.toString());
        RankList rl0 = new RankList();
        rl0.add(dp0);
        rl0.add(dp1);
        samples.add(rl0);
        SpanF.verbose = false;
        SpanF spanf = new SpanF(samples, null) {
            @Override
            public int[] getShuffledWeights() {
                return new int[]{1, 2, 0};
            }
        };
        spanf.set(new APIAScorer());
        spanf.init();
        spanf.learn();
        assertTrue(0.75 < spanf.weight[1]);
    }
    @Test
    public void testRank_withoutDedup() {
        List<RankList> samples = new ArrayList<>();
        DataPoint dp0 = new DataPoint("0 qid:0 1:1 2:0.5 3:2.0 4:3 #did=0");
        DataPoint dp1 = new DataPoint("0 qid:0 1:1 2:0.5 3:2.0 4:5 #did=1");
        DataPoint dp2 = new DataPoint("0 qid:0 1:1 2:0.5 3:2.0 4:7 #did=2");
        RankList rl0 = new RankList();
        rl0.add(dp0);
        rl0.add(dp1);
        rl0.add(dp2);
        samples.add(rl0);
        SpanF spanf = new SpanF(samples, null);
        spanf.init();
        RankList ranking = spanf.rank(rl0);
        assertEquals(dp2, ranking.get(0));
        assertEquals(dp1, ranking.get(1));
        assertEquals(dp0, ranking.get(2));
    }
    @Test
    public void testRank_withDedup() {
        List<RankList> samples = new ArrayList<>();
        DataPoint dp0 = new DataPoint("0 qid:0 1:1 2:0.5 3:2.0 4:3 #did=0");
        DataPoint dp1 = new DataPoint("0 qid:0 1:1 2:0.5 3:2.0 4:5 #did=0");
        DataPoint dp2 = new DataPoint("0 qid:0 1:1 2:0.5 3:2.0 4:7 #did=0");
        RankList rl0 = new RankList();
        rl0.add(dp0);
        rl0.add(dp1);
        rl0.add(dp2);
        samples.add(rl0);
        SpanF spanf = new SpanF(samples, null);
        spanf.init();
        RankList ranking = spanf.rank(rl0);
        assertEquals(1, ranking.size());
        assertEquals(dp2, ranking.get(0));
    }
    @Test
    public void testEval_k1f1() {
        List<RankList> samples = new ArrayList<>();
        DataPoint dp0 = new DataPoint("0 qid:0 1:1 2:0.5 3:2.0 4:7 #did=0");
        RankList rl0 = new RankList();
        rl0.add(dp0);
        samples.add(rl0);
        SpanF spanf = new SpanF(samples, null);
        spanf.init();
        double w = 7.0 * 1.0 / (1.0 - 0.75 + 0.75 * 2.0);
        double expect = w / (w + 1.2) * 0.5;
        assertEquals(expect, spanf.eval(dp0), 1.0e-6);
    }
    @Test
    public void testEval_k2f2() {
        List<RankList> samples = new ArrayList<>();
        StringBuilder sb = new StringBuilder();
        sb.append("1 qid:0 ");
        sb.append("1:2 2:-0.5 3:0.5 ");
        sb.append("4:2.0 5:4 6:5 ");
        sb.append("7:3.0 8:6 9:7 ");
        sb.append("#did=0");
        DataPoint dp0 = new DataPoint(sb.toString());
        RankList rl0 = new RankList();
        rl0.add(dp0);
        samples.add(rl0);
        SpanF spanf = new SpanF(samples, null);
        spanf.init();
        double w = 0.0, expect = 0.0;
        w += 4.0 * 1.0 / (1.0 - 0.75 + 0.75 * 2.0);
        w += 6.0 * 1.0 / (1.0 - 0.75 + 0.75 * 3.0);
        expect += w / (w + 1.2) * -0.5;
        w = 0.0;
        w += 5.0 * 1.0 / (1.0 - 0.75 + 0.75 * 2.0);
        w += 7.0 * 1.0 / (1.0 - 0.75 + 0.75 * 3.0);
        expect += w / (w + 1.2) * 0.5;
        assertEquals(expect, spanf.eval(dp0), 1.0e-6);
    }
    @Test
    public void testName() {
        SpanF spanf = new SpanF();
        assertEquals("SpanF", spanf.name());
    }
    @Test
    public void testGetShuffledWeights() {
        List<RankList> samples = new ArrayList<>();
        DataPoint dp0 = new DataPoint("0 qid:0 1:1 2:-1.0 3:1.0 4:7 #did=0");
        RankList rl0 = new RankList();
        rl0.add(dp0);
        samples.add(rl0);
        SpanF spanf = new SpanF(samples, null);
        spanf.init();
        try {
            Method gsw = SpanF.class.getDeclaredMethod("getShuffledWeights");
            gsw.setAccessible(true);
            int[] actual = (int[])gsw.invoke(spanf);
            Arrays.sort(actual);
            assertEquals(3, actual.length);
            for(int i = 0; i < 3; i++)
                assertEquals(i, actual[i], 1.0e-6);
        }
        catch (Exception e) {
            fail();
        }
    }
}
