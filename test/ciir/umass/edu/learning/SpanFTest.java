package ciir.umass.edu.learning;

import ciir.umass.edu.metric.APIAScorer;
import org.junit.Test;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.*;

public class SpanFTest {
    @Test
    public void testGetK() {
        DataPoint dp0 = new DenseDataPoint("0 qid:0 1:7 #did:0");
        assertEquals(7, (int)dp0.getFeatureValue(1));
        DataPoint dp1 = new DenseDataPoint("0 qid:0 1:7.0 #did:0");
        assertEquals(7, (int)dp1.getFeatureValue(1));
    }
    @Test
    public void testGetF() {
        DataPoint dp0 = new DenseDataPoint("0 qid:0 1:0 2:7 #did:0");
        assertEquals(7, (int)dp0.getFeatureValue(2));
        DataPoint dp1 = new DenseDataPoint("0 qid:0 1:0 2:7.0 #did:0");
        assertEquals(7, (int)dp1.getFeatureValue(2));
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
        SpanF spanf = new SpanF(samples, null, new APIAScorer());
        spanf.init();
        double[] expect = {0.4};
        _testInit(expect, 0, spanf);
    }
    @Test
    public void testInit_k1f1s1() {
        List<RankList> samples = new ArrayList<>();
        DataPoint dp0 = new DenseDataPoint(
            "0 qid:0 1:1 2:1 3:1.0 4:1.0 5:1 6:1 7:10 #did=0"
        );
        RankList rl0 = new RankList(Collections.singletonList(dp0));
        samples.add(rl0);
        SpanF spanf = new SpanF(samples, null, new APIAScorer());
        spanf.init();
        double[] expect = {0.4, 0.55, 0.25, 0.3, 1.0};
        _testInit(expect, 1, spanf);
    }
    @Test
    public void testInit_k2f2s2() {
        List<RankList> samples = new ArrayList<>();
        StringBuilder sb = new StringBuilder();
        sb.append("0 qid:0 1:2 2:2 3:0.5 4:2.0 5:0.5 6:2.0 ");
        sb.append("7:2 8:2 9:20 10:1 9:11 ");
        sb.append("12:2 13:1 14:5 15:2 16:40 ");
        sb.append("#did=0");
        DataPoint dp0 = new DenseDataPoint(sb.toString());
        RankList rl0 = new RankList(Collections.singletonList(dp0));
        samples.add(rl0);
        SpanF spanf = new SpanF(samples, null, new APIAScorer());
        spanf.init();
        double[] expect = {0.4, 0.55, 0.55, 0.25, 0.25, 0.3, 0.3, 1.0, 1.0};
        _testInit(expect, 2, spanf);
    }
    @Test
    public void testWeightNameOf() {
        List<RankList> samples = new ArrayList<>();
        StringBuilder sb = new StringBuilder();
        sb.append("0 qid:0 1:2 2:2 3:0.5 4:2.0 5:0.5 6:2.0 ");
        sb.append("7:2 8:2 9:20 10:1 9:11 ");
        sb.append("12:2 13:1 14:5 15:2 16:40 ");
        sb.append("#did=0");
        DataPoint dp0 = new DenseDataPoint(sb.toString());
        RankList rl0 = new RankList(Collections.singletonList(dp0));
        samples.add(rl0);
        SpanF spanf = new SpanF(samples, null, new APIAScorer());
        spanf.init();
        assertEquals("k1", spanf.weightNameOf(0));
        assertEquals("z(1)", spanf.weightNameOf(1));
        assertEquals("z(2)", spanf.weightNameOf(2));
        assertEquals("x(1)", spanf.weightNameOf(3));
        assertEquals("x(2)", spanf.weightNameOf(4));
        assertEquals("b(1)", spanf.weightNameOf(5));
        assertEquals("b(2)", spanf.weightNameOf(6));
        assertEquals("boost(1)", spanf.weightNameOf(7));
        assertEquals("boost(2)", spanf.weightNameOf(8));
    }
    @Test
    public void testLearn_k1() {
        List<RankList> samples = new ArrayList<>();
        StringBuilder sb = new StringBuilder();
        sb.append("1 qid:0 ");
        sb.append("1:2 2:1 3:1.0 4:1.0 5:1.0 ");
        sb.append("6:1 7:2 8:10 ");
        sb.append("9:1 10:2 11:10 ");
        sb.append("#did=0");
        DataPoint dp0 = new DenseDataPoint(sb.toString());
        sb = new StringBuilder();
        sb.append("0 qid:0 ");
        sb.append("1:2 2:1 3:1.0 4:1.0 5:1.0 ");
        sb.append("6:1 7:1 8:10 ");
        sb.append("9:1 10:5 11:10 ");
        sb.append("#did=1");
        DataPoint dp1 = new DenseDataPoint(sb.toString());
        DataPoint[] dps = new DataPoint[]{dp0, dp1};
        RankList rl0 = new RankList(Arrays.asList(dps));
        samples.add(rl0);
        SpanF.verbose = true;
        SpanF spanf = new SpanF(samples, null, new APIAScorer()) {
            @Override
            public int[] getShuffledWeights() {
                return new int[]{0};
            }
        };
        spanf.init();
        spanf.learn();
        System.out.println(spanf.weight[0]);
        assertTrue(spanf.weight[0] < 0.4);
    }
    @Test
    public void testLearn_boost() {
        List<RankList> samples = new ArrayList<>();
        StringBuilder sb = new StringBuilder();
        sb.append("1 qid:0 ");
        sb.append("1:1 2:2 3:1.0 4:1.0 5:1.0 ");
        sb.append("6:1 7:1 8:10 9:1 10:1 11:10 ");
        sb.append("#did=0");
        DataPoint dp0 = new DenseDataPoint(sb.toString());
        sb = new StringBuilder();
        sb.append("0 qid:0 ");
        sb.append("1:1 2:2 3:1.0 4:1.0 5:1.0 ");
        sb.append("6:1 7:10 8:10 9:0 ");
        sb.append("#did=1");
        DataPoint dp1 = new DenseDataPoint(sb.toString());
        DataPoint[] dps = new DataPoint[]{dp0, dp1};
        RankList rl0 = new RankList(Arrays.asList(dps));
        samples.add(rl0);
        SpanF.verbose = false;
        SpanF spanf = new SpanF(samples, null, new APIAScorer()) {
            @Override
            public int[] getShuffledWeights() {
                return new int[]{7, 8};
            }
        };
        spanf.init();
        spanf.learn();
        assertTrue(spanf.weight[7] < spanf.weight[8]);
    }
    @Test
    public void testLearn_b() {
        List<RankList> samples = new ArrayList<>();
        StringBuilder sb = new StringBuilder();
        sb.append("1 qid:0 ");
        sb.append("1:1 2:1 3:1.0 4:10.0 ");
        sb.append("5:1 6:2 7:10 ");
        sb.append("#did=0");
        DataPoint dp0 = new DenseDataPoint(sb.toString());
        sb = new StringBuilder();
        sb.append("0 qid:0 ");
        sb.append("1:1 2:1 3:1.0 4:1.0 ");
        sb.append("5:1 6:1 7:10 ");
        sb.append("#did=1");
        DataPoint dp1 = new DenseDataPoint(sb.toString());
        DataPoint[] dps = new DataPoint[]{dp0, dp1};
        RankList rl0 = new RankList(Arrays.asList(dps));
        samples.add(rl0);
        SpanF.verbose = false;
        SpanF spanf = new SpanF(samples, null, new APIAScorer()) {
            @Override
            public int[] getShuffledWeights() {
                return new int[]{3};
            }
        };
        spanf.init();
        spanf.learn();
        assertTrue(spanf.weight[3] < 0.3);

    }
    @Test
    public void testRank_withoutDedup() {
        List<RankList> samples = new ArrayList<>();
        DataPoint dp0 = new DenseDataPoint("0 qid:0 1:1 2:1 3:1.0 4:1.0 5:1 6:1 7:10 #did=0");
        DataPoint dp1 = new DenseDataPoint("0 qid:0 1:1 2:1 3:1.0 4:1.0 5:1 6:2 7:10 #did=1");
        DataPoint dp2 = new DenseDataPoint("0 qid:0 1:1 2:1 3:1.0 4:1.0 5:1 6:3 7:10 #did=2");
        DataPoint[] dps = new DataPoint[]{dp0, dp1, dp2};
        RankList rl0 = new RankList(Arrays.asList(dps));
        samples.add(rl0);
        SpanF spanf = new SpanF(samples, null, new APIAScorer());
        spanf.init();
        RankList ranking = spanf.rank(rl0);
        assertEquals(dp2, ranking.get(0));
        assertEquals(dp1, ranking.get(1));
        assertEquals(dp0, ranking.get(2));
    }
    @Test
    public void testRank_withDedup() {
        List<RankList> samples = new ArrayList<>();
        DataPoint dp0 = new DenseDataPoint("0 qid:0 1:1 2:1 3:1.0 4:1.0 5:1 6:1 7:10 #did=0");
        DataPoint dp1 = new DenseDataPoint("0 qid:0 1:1 2:1 3:1.0 4:1.0 5:1 6:2 7:10 #did=0");
        DataPoint dp2 = new DenseDataPoint("0 qid:0 1:1 2:1 3:1.0 4:1.0 5:1 6:3 7:10 #did=0");
        DataPoint[] dps = new DataPoint[]{dp0, dp1, dp2};
        RankList rl0 = new RankList(Arrays.asList(dps));
        samples.add(rl0);
        SpanF spanf = new SpanF(samples, null, new APIAScorer());
        spanf.init();
        RankList ranking = spanf.rank(rl0);
        assertEquals(3, ranking.size());
        assertEquals(dp2, ranking.get(0));
        assertEquals(null, ranking.get(1));
        assertEquals(null, ranking.get(2));
    }
    @Test
    public void testEval_k1f1s1() {
        List<RankList> samples = new ArrayList<>();
        DataPoint dp0 = new DenseDataPoint(
                "0 qid:0 1:1 2:1 3:0.7 4:0.3 5:1 6:1 7:10 #did=0"
        );
        RankList rl0 = new RankList(Collections.singletonList(dp0));
        samples.add(rl0);
        SpanF spanf = new SpanF(samples, null, new APIAScorer());
        spanf.init();
        double w = 1.0 / (1 - 0.3 + 0.3 * 0.3) *
                Math.pow(1, 0.55) / Math.pow(10, 0.25);
        double expect = w / (w + 0.4) * 0.7;
        assertEquals(expect, spanf.eval(dp0), 1.0e-6);
    }
    @Test
    public void testEval_k2f2s2() {
        List<RankList> samples = new ArrayList<>();
        StringBuilder sb = new StringBuilder();
        sb.append("1 qid:0 1:2 2:2 3:0.1 4:0.2 5:0.3 6:0.7 ");
        sb.append("7:2 8:1 9:5 10:2 11:10 ");
        sb.append("12:2 13:3 14:10 15:4 16:5 ");
        sb.append("17:2 18:5 19:5 20:6 21:10 ");
        sb.append("22:2 23:7 24:10 25:8 26:5 ");
        sb.append("#did=0");
        DataPoint dp0 = new DenseDataPoint(sb.toString());
        RankList rl0 = new RankList(Collections.singletonList(dp0));
        samples.add(rl0);
        SpanF spanf = new SpanF(samples, null, new APIAScorer());
        spanf.init();
        double w = 0.0, expect = 0.0;
        w += 1.0 / (1 - 0.3 + 0.3 * 0.3) *
                (Math.pow(1, 0.55) / Math.pow(5, 0.25) +
                        Math.pow(2, 0.55) / Math.pow(10, 0.25));
        w += 1.0 / (1 - 0.3 + 0.3 * 0.7) *
                (Math.pow(3, 0.55) / Math.pow(10, 0.25) +
                        Math.pow(4, 0.55) / Math.pow(5, 0.25));
        expect += w / (w + 0.4) * 0.1;
        w = 0.0;
        w += 1.0 / (1 - 0.3 + 0.3 * 0.3) *
                (Math.pow(5, 0.55) / Math.pow(5, 0.25) +
                        Math.pow(6, 0.55) / Math.pow(10, 0.25));
        w += 1.0 / (1 - 0.3 + 0.3 * 0.7) *
                (Math.pow(7, 0.55) / Math.pow(10, 0.25) +
                        Math.pow(8, 0.55) / Math.pow(5, 0.25));
        expect += w / (w + 0.4) * 0.2;
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
        StringBuilder sb = new StringBuilder();
        sb.append("0 qid:0 1:2 2:2 3:0.5 4:2.0 5:0.5 6:2.0 ");
        sb.append("7:2 8:2 9:20 10:1 9:11 ");
        sb.append("12:2 13:1 14:5 15:2 16:40 ");
        sb.append("#did=0");
        DataPoint dp0 = new DenseDataPoint(sb.toString());
        RankList rl0 = new RankList(Collections.singletonList(dp0));
        samples.add(rl0);
        SpanF spanf = new SpanF(samples, null, new APIAScorer());
        spanf.init();
        try {
            Method gsw = SpanF.class.getDeclaredMethod("getShuffledWeights");
            gsw.setAccessible(true);
            int[] actual = (int[])gsw.invoke(spanf);
            Arrays.sort(actual);
            assertEquals(9, actual.length);
            for(int i = 0; i < 9; i++)
                assertEquals(i, actual[i], 1.0e-6);
        }
        catch (Exception e) {
            fail();
        }
    }
    @Test
    public void testGetDistance() {
        SpanF spanF = new SpanF();
        spanF.f = 2;
        double[] regVector = {0.4, 0.55, 0.55, 0.25, 0.25, 0.3, 0.3, 1.0, 1.0};
        double[] weights = {0.8, 0.45, 0.65, 0.15, 0.35, 0.2, 0.4, 0.5, 2.0};
        assertEquals(Math.sqrt(0.18), spanF.getDistance(regVector, weights), 1.0e-6);
    }
}
