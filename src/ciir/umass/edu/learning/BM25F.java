package ciir.umass.edu.learning;

import java.util.*;

import ciir.umass.edu.metric.MetricScorer;
import ciir.umass.edu.utilities.MergeSorter;
import ciir.umass.edu.utilities.SimpleMath;

import javax.xml.crypto.Data;

/**
 * @author tmanabe
 *
 * This class implements the BM25F model and its optimization with the method known as Coordinate Ascent.
 *
 * You must sort kf + 1 features in the following order where
 * k means |keywords|, f |fields|, tf(i, j) TF of i-th keyword in j-th field,
 * H(i) entropy of i-th keyword, and lp(j) length penalty of j-th field.
 * [k,     H(1),     H(2),     H(3),     ..., H(k),
 *  lp(1), tf(1, 1), tf(2, 1), tf(3, 1), ..., tf(k, 1),
 *  lp(2), tf(1, 2), tf(2, 2), tf(3, 2), ..., tf(k, 2),
 *  lp(3), tf(1, 3), tf(2, 3), tf(3, 3), ..., tf(k, 3),
 *  ...
 *  lp(f), tf(1, f), tf(2, f), tf(3, f), ..., tf(k, f)]
 * Note that H(i) is the (i+1)-th feature,
 * lp(j) the ((k+1)j+1)-th, and tf(i, j) the ((k+1)j+i+1)-th.
 */
public class BM25F extends CoorAscent {

	//Local variables
    protected int f;

    BM25F()
    {

    }
    BM25F(List<RankList> samples, int[] features, MetricScorer scorer)
    {
        super(samples, features, scorer);
    }
	public void init()
    {
		PRINT("Initializing... ");
        if(0 < samples.size()) {
            RankList sampleRL = samples.get(0);
            DataPoint sampleDP = sampleRL.get(0);
            int k = (int)sampleDP.getFeatureValue(1);
            f = sampleDP.getLastFeature() / (k + 1) - 1;
        } else {
            f = 0;
        }
		weight = new double[1 + 2 * f];
		// [k1, b(1), b(2), b(3), ..., b(f), boost(1), boost(2), boost(3), ..., boost(f)]
        // Note that b(j) is the j-th element and also that boost(j) is the (f+j)-th.
        weight[0] = 1.2;
        for(int i=1;i<=f;i++)
            weight[i] = 0.75;
        for(int i=f+1;i<weight.length;i++)
            weight[i] = 1.0;
		PRINTLN("[Done]");
	}
	String weightNameOf(int weight_index)
    {
        if(weight_index <= 0) {
            return "k1";
        } else if(weight_index <= f) {
            return "b(" + weight_index +")";
        } else {
            return "boost(" + (weight_index - f) + ")";
        }
    }
    public void learn()
    {
        double[] regVector = new double[weight.length];
        copy(weight, regVector);

        //this holds the final best model/score
        double[] bestModel = new double[weight.length];
        double bestModelScore = Double.NEGATIVE_INFINITY;

        int[] sign = new int[]{1, -1};

        PRINTLN("---------------------------");
        PRINTLN("Training starts...");
        PRINTLN("---------------------------");

        for(int r=0;r<nRestart;r++)
        {
            PRINTLN("[+] Random restart #" + (r+1) + "/" + nRestart + "...");
            int consecutive_fails = 0;

            //initialize weight vector
            weight[0] = 1.2;
            for(int i=1;i<=f;i++)
                weight[i] = 0.75f;
            for(int i=f+1;i<weight.length;i++)
                weight[i] = 1.0f;
            double startScore = scorer.score(rank(samples));//compute all the scores (in whatever metric specified) and store them as cache

            //local best (within the current restart cycle)
            double bestScore = startScore;
            double[] bestWeight = new double[weight.length];
            copy(weight, bestWeight);

            //There must be at least one weight increasing it helps
            while((weight.length>1&&consecutive_fails < weight.length - 1) || (weight.length==1&&consecutive_fails==0))
            {
                PRINT("Shuffling weights' order... ");
                int[] wids = getShuffledWeights();//contain index of weights
                PRINTLN("[Done.]");
                PRINTLN("Optimizing weight vector... ");
                PRINTLN("------------------------------");
                PRINTLN(new int[]{11, 8, 7}, new String[]{"weight name", "value", scorer.name()});
                PRINTLN("------------------------------");
                //Try maximizing each feature individually
                for(int current_weight : wids)
                {
                    String weightName = weightNameOf(current_weight);

                    double origWeight = weight[current_weight];
                    double bestWeightValue = origWeight;//0.0;
                    boolean succeeds = false;//whether or not we succeed in finding a better weight value for the current feature
                    for(int s : sign)
                    {
                        double step = 0.001;
                        if(origWeight != 0.0 && step > 0.5 * Math.abs(origWeight))
                            step = stepBase * Math.abs(origWeight);
                        double totalStep = step;
                        for(int j=0;j<nMaxIteration;j++)
                        {
                            double w = origWeight + totalStep * s;

                            if(current_weight <= 0 && w <= 0) {  // k1
                                break;
                            } else if(0 < current_weight && current_weight <= f && (w < 0 || 1.0d < w)) {  // b
                                break;
                            }

                            weight[current_weight] = w;
                            double score = scorer.score(rank(samples));
                            if(regularized)
                            {
                                double penalty = slack * getDistance(weight, regVector);
                                score -= penalty;
                                //PRINTLN("Penalty: " + penalty);
                            }
                            if(score > bestScore)//better than the local best, replace the local best with this model
                            {
                                bestScore = score;
                                bestWeightValue = weight[current_weight];
                                succeeds = true;

                                String bw = ((bestWeightValue>0.0)?"+":"") + SimpleMath.round(bestWeightValue, 4);
                                PRINTLN(new int[]{11, 8, 7}, new String[]{weightName, bw+"", SimpleMath.round(bestScore, 4)+""});
                            }
                            step *= stepScale;
                            totalStep += step;
                        }
                        if(succeeds)
                            break;//no need to search the other direction (e.g. sign = '-')
                        else
                        {
                            //so that we can start searching in the other direction (since the optimization in the first direction failed)
                            weight[current_weight] = origWeight;//restore the weight to its initial value
                        }
                    }
                    if(succeeds)
                    {
                        weight[current_weight] = bestWeightValue;
                        consecutive_fails = 0;//since we found a better weight value
                        //then normalize the new weight vector
                        copy(weight, bestWeight);
                    }
                    else
                    {
                        consecutive_fails++;
                        //Restore the orig. weight value
                        weight[current_weight] = origWeight;
                    }
                }
                PRINTLN("------------------------------");
                //if we haven't made much progress then quit
                if(bestScore - startScore < tolerance)
                    break;
            }
            //update the (global) best model with the best model found in this round
            if(bestScore > bestModelScore)
            {
                bestModelScore = bestScore;
                copy(bestWeight, bestModel);
            }
        }

        copy(bestModel, weight);
        current_feature = -1;//turn off the cache mode
        scoreOnTrainingData = SimpleMath.round(scorer.score(rank(samples)), 4);
        PRINTLN("---------------------------------");
        PRINTLN("Finished successfully.");
        PRINTLN(scorer.name() + " on training data: " + scoreOnTrainingData);
        if(validationSamples != null)
        {
            bestScoreOnValidationData = scorer.score(rank(validationSamples));
            PRINTLN(scorer.name() + " on validation data: " + SimpleMath.round(bestScoreOnValidationData, 4));
        }
        PRINTLN("---------------------------------");
    }
	public RankList rank(RankList rl)
	{
		double[] score = new double[rl.size()];
        Map<String, Double> descToBestScore = new HashMap<>();
        Map<String, Integer> descToBestIndex = new HashMap<>();
        for(int index=0;index<rl.size();index++) {
            DataPoint dp = rl.get(index);
            String desc = dp.description;
            Double current = eval(dp);
            Double best = descToBestScore.get(desc);
            if(best == null) {
                descToBestScore.put(desc, current);
                descToBestIndex.put(desc, index);
                score[index] = current;
            } else {
                if(best < current) {
                    score[descToBestIndex.get(desc)] = Double.NEGATIVE_INFINITY;
                    descToBestScore.put(desc, current);
                    descToBestIndex.put(desc, index);
                    score[index] = current;
                } else {
                    score[index] = Double.NEGATIVE_INFINITY;
                }
            }
        }
		int[] idx = MergeSorter.sort(score, false);
        int effectiveCount = 0;
        for (; effectiveCount < idx.length; ++effectiveCount) {
            if (Double.isInfinite(score[idx[effectiveCount]])) {
                break;
            }
        }
        int[] effectiveIdx = new int[effectiveCount];
        System.arraycopy(idx, 0, effectiveIdx, 0, effectiveIdx.length);
		return new RankList(rl, effectiveIdx);
	}
	public double eval(DataPoint p)
    {
        int k = (int)p.getFeatureValue(1);
        if (f <= 0) {  // For scoring test data points
            f = p.getLastFeature() / (k + 1) - 1;
        }
        double score = 0.0;
        for(int i=1; i<=k; i++) {
            double w = 0.0;
            for(int j=1; j<=f; j++) {
                double numer = p.getFeatureValue((k+1)*j+i+1) * weight[f+j];
                double denom = 1.0 - weight[j] + weight[j] * p.getFeatureValue((k+1)*j+1);
                if(0.0 < denom) w += numer / denom;
            }
            score += w / (w + weight[0]) * p.getFeatureValue(i+1);
        }
 		return score;
	}
    public Ranker createNew()
    {
        return new BM25F();
    }
	public Ranker clone() throws CloneNotSupportedException
	{
	    super.clone();
	    return new BM25F();
	}
	public String name()
	{
		return "BM25F";
	}
    public int[] getShuffledWeights()
    {
        int[] wids = new int[weight.length];
        List<Integer> l = new ArrayList<>();
        for(int i=0;i<weight.length;i++)
            l.add(i);
        Collections.shuffle(l);
        for(int i=0;i<l.size();i++)
            wids[i] = l.get(i);
        return wids;
    }
    double getDistance(double[] w1, double[] w2)
    {
        assert(w1.length == w2.length);
        double s1 = 0.0;
        double s2 = 0.0;
        for(int i=f+1;i<w1.length;i++)
        {
            s1 += Math.abs(w1[i]);
            s2 += Math.abs(w2[i]);
        }
        double dist = 0.0;
        for(int i=f+1;i<w1.length;i++)
        {
            double t = w1[i]/s1 - w2[i]/s2;
            dist += t*t;
        }
        return (double)Math.sqrt(dist);
    }
}
