package ciir.umass.edu.learning;

import ciir.umass.edu.utilities.MergeSorter;
import ciir.umass.edu.utilities.SimpleMath;

import java.util.*;

/**
 * @author tmanabe
 *
 * This class implements the Span model proposed by Song et al. [1] then
 * naturally extended for multiple fields by tmanabe, and
 * its optimization with the method known as Coordinate Ascent.
 *
 * [1] R. Song, M. J. Taylor, J.-R. Wen, H.-W. Hon, and Y. Yu. Viewing term
 * proximity from a different perspective. In ECIR, pages 346–357, 2008.
 *
 * You must sort features in the following order where
 * k means |keywords|, f |fields|,
 * H(j) the entoropy of j-th keyword,
 * lp(i) the length penalty of i-th field,
 * n(i,j) |spans of i-th field in which j-th keyword occurs|,
 * s(h,i) |keywords occur in the h-th span of i-th field|, and
 * w(h,i,j) the width of h-th span of i-th field in which j-th keyword occurs.
 *
 * [k, f, H(1), H(2), H(3), ..., H(k), lp(1), lp(2), lp(3), ..., lp(f),
 *
 *  n(1,1), s(1,1), w(1,1,1), s(2,1), w(2,1,1), ..., s(n(1,1),1), w(n(1,1),1,1),
 *  n(2,1), s(1,2), w(1,2,1), s(2,2), w(2,2,1), ..., s(n(2,1),2), w(n(2,1),2,1),
 *  n(3,1), s(1,3), w(1,3,1), s(2,3), w(2,3,1), ..., s(n(3,1),3), w(n(3,1),3,1),
 *  ...
 *  n(f,1), s(1,f), w(1,f,1), s(2,f), w(2,f,1), ..., s(n(f,1),f), w(n(f,1),f,1),
 *
 *  n(1,2), s(1,1), w(1,1,2), s(2,1), w(2,1,2), ..., s(n(1,2),1), w(n(1,2),1,2),
 *  n(2,2), s(1,2), w(1,2,2), s(2,2), w(2,2,2), ..., s(n(2,2),2), w(n(2,2),2,2),
 *  n(3,2), s(1,3), w(1,3,2), s(2,3), w(2,3,2), ..., s(n(3,2),3), w(n(3,2),3,2),
 *  ...
 *  n(f,2), s(1,f), w(1,f,2), s(2,f), w(2,f,2), ..., s(n(f,2),f), w(n(f,2),f,2),
 *
 *  n(1,3), s(1,1), w(1,1,3), s(2,1), w(2,1,3), ..., s(n(1,3),1), w(n(1,3),1,3),
 *  n(2,3), s(1,2), w(1,2,3), s(2,2), w(2,2,3), ..., s(n(2,3),2), w(n(2,3),2,3),
 *  n(3,3), s(1,3), w(1,3,3), s(2,3), w(2,3,3), ..., s(n(3,3),3), w(n(3,3),3,3),
 *  ...
 *  n(f,3), s(1,f), w(1,f,3), s(2,f), w(2,f,3), ..., s(n(f,3),f), w(n(f,3),f,3),
 *
 *  ...
 *
 *  n(1,k), s(1,1), w(1,1,k), s(2,1), w(2,1,k), ..., s(n(1,k),1), w(n(1,k),1,k),
 *  n(2,k), s(1,2), w(1,2,k), s(2,2), w(2,2,k), ..., s(n(2,k),2), w(n(2,k),2,k),
 *  n(3,k), s(1,3), w(1,3,k), s(2,3), w(2,3,k), ..., s(n(3,k),3), w(n(3,k),3,k),
 *  ...
 *  n(f,k), s(1,f), w(1,f,k), s(2,f), w(2,f,k), ..., s(n(f,k),f), w(n(f,k),f,k)]
 */
public class SpanF extends CoorAscent {

	//Local variables
    protected int f;

    SpanF()
    {

    }
    SpanF(List<RankList> samples, int[] features)
    {
        super(samples, features);
    }
	public void init()
    {
		PRINT("Initializing... ");
        if(0 < samples.size()) {
            RankList sampleRL = samples.get(0);
            DataPoint sampleDP = sampleRL.get(0);
            f = (int)sampleDP.getFeatureValue(2);
        } else {
            f = 0;
        }
		weight = new double[1 + 4 * f];
		/**
         * [k1,
         *  z(1), z(2), z(3), ..., z(f),
         *  x(1), x(2), x(3), ..., x(f),
         *  b(1), b(2), b(3), ..., b(f),
         *  boost(1), boost(2), boost(3), ..., boost(f)]
         *  Note that z(i) is the i-th element,
         *  that x(i) is the (f+i)-th,
         *  that b(i) is the (2f+i)-th, and
         *  that boost(i) is the (3f+i)-th.
         */
        weight[0] = 0.4;
        for(int i=0*f+1;i<=1*f;i++)
            weight[i] = 0.55;
        for(int i=1*f+1;i<=2*f;i++)
            weight[i] = 0.25;
        for(int i=2*f+1;i<=3*f;i++)
            weight[i] = 0.3;
        for(int i=3*f+1;i<=4*f;i++)
            weight[i] = 1.0;
		PRINTLN("[Done]");
	}
	String weightNameOf(int weight_index)
    {
        if(weight_index <= 0) {
            return "k1";
        } else if(weight_index <= 1*f) {
            return "z(" + (weight_index - 0*f)+")";
        } else if(weight_index <= 2*f) {
            return "x(" + (weight_index - 1*f) +")";
        } else if(weight_index <= 3*f) {
            return "b(" + (weight_index - 2*f) +")";
        } else {
            return "boost(" + (weight_index - 3*f) + ")";
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
            weight[0] = 0.4;
            for(int i=0*f+1;i<=1*f;i++)
                weight[i] = 0.55;
            for(int i=1*f+1;i<=2*f;i++)
                weight[i] = 0.25;
            for(int i=2*f+1;i<=3*f;i++)
                weight[i] = 0.3;
            for(int i=3*f+1;i<=4*f;i++)
                weight[i] = 1.0;

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
                            } else if(2*f < current_weight && current_weight <= 3*f && (w < 0 || 1.0 < w)) {  // b
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
		return new RankList(rl, idx);
	}
	public double eval(DataPoint p)
    {
        int k = (int)p.getFeatureValue(1);
        assert f == (int) p.getFeatureValue(2);
        double[] hs = new double[k+1];
        double[] lps = new double[f+1];
        int idx = 3;
        for (int j=1; j<=k; ++j) {
            hs[j] = p.getFeatureValue(idx++);
        }
        for (int i=1; i<=f; ++i) {
            lps[i] = p.getFeatureValue(idx++);
        }
        double score = 0.0;
        for (int j=1; j<=k; ++j) {
            double keyword_score = 0.0;
            for (int i=1; i<=f; ++i) {
                double field_score = 0.0;
                double b = weight[2*f+i];
                int n = (int) p.getFeatureValue(idx++);
                for (int h=1; h<=n; ++h) {
                    field_score +=
                            Math.pow(p.getFeatureValue(idx++), weight[i]) /
                                    Math.pow(p.getFeatureValue(idx++), weight[f+i]);
                }
                keyword_score += weight[3*f+i] /
                        (1 - b + b * lps[i]) * field_score;
            }
            score += keyword_score / (weight[0] + keyword_score) * hs[j];
        }
 		return score;
	}
	public Ranker clone()
	{
	    super.clone();
	    return new SpanF();
	}
	public String name()
	{
		return "SpanF";
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
}
