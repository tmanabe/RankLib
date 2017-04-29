package ciir.umass.edu.metric;

import ciir.umass.edu.learning.DataPoint;
import ciir.umass.edu.learning.RankList;

public class APIAScorer extends MetricScorer {
    public APIAScorer()
    {
        this.k = 0;//consider the whole list
    }
    public double score(RankList rl)
    {
        double cap = 0.0;
        int cc;  // |Intents|
        if(0 < rl.size()) {
            DataPoint sampleDP = rl.get(0);
            cc = sampleDP.getLabels().length;
        } else {
            return 0.0;
        }
        for(int i = 0; i < cc; i++) {
            double ap = 0.0;
            int c = 0;
            for (int r = 1; r <= rl.size(); r++) {
                if (0.0 < rl.get(r - 1).getLabels()[i])//relevant
                {
                    c++;
                    ap += ((double) c) / r;
                }
            }
            if(0 < c)
                cap += ap / c;
        }
        if(0 < cc)
            return cap / cc;
        return 0.0;
    }
    public MetricScorer clone()
    {
        super.clone();
        return new APIAScorer();
    }
    public String name()
    {
        return "MAPIA";
    }
}
