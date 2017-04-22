package ciir.umass.edu.learning;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import ciir.umass.edu.utilities.MergeSorter;
import ciir.umass.edu.utilities.SimpleMath;

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
 *  lp(2), tf(1, 2), tf(2, 2), tf(3, 2), ..., tf(k, 1),
 *  lp(3), tf(1, 3), tf(2, 3), tf(3, 3), ..., tf(k, 1),
 *  ...
 *  lp(f), tf(1, f), tf(2, f), tf(3, f), ..., tf(k, f)]
 * Note that lp(j) is the (k*j)-th element, and
 * also that tf(i, j) is the (k*j+i)-th.
 */
public class BM25F extends CoorAscent {

	//Local variables
    protected int f;

    protected int current_weight = -1;//used only during learning

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
            weight[i] = 0.75f;
        for(int i=f+1;i<weight.length;i++)
            weight[i] = 1.0f;
		PRINTLN("[Done]");
	}
    public void learn()
    {
        double[] regVector = new double[weight.length];
        copy(weight, regVector);

        //this holds the final best model/score
        double[] bestModel = null;
        double bestModelScore = 0.0;

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
            current_weight = -1;
            double startScore = scorer.score(rank(samples));//compute all the scores (in whatever metric specified) and store them as cache

            //local best (within the current restart cycle)
            double bestScore = startScore;
            double[] bestWeight = new double[weight.length];
            copy(weight, bestWeight);

            //There must be at least one feature increasing whose weight helps
            while((weight.length>1&&consecutive_fails < weight.length - 1) || (weight.length==1&&consecutive_fails==0))
            {
                PRINT("Shuffling weights' order... ");
                int[] wids = getShuffledWeights();//contain index of weights
                PRINTLN("[Done.]");
                PRINTLN("Optimizing weight vector... ");
                PRINTLN("------------------------------");
                PRINTLN(new int[]{9, 8, 7}, new String[]{"weight ID", "value", scorer.name()});
                PRINTLN("------------------------------");
                //Try maximizing each feature individually
                for(int i=0;i<wids.length;i++)
                {
                    current_weight = wids[i];//this will trigger the "else" branch in the procedure rank()

                    double origWeight = weight[wids[i]];
                    double bestWeightValue = origWeight;//0.0;
                    boolean succeeds = false;//whether or not we succeed in finding a better weight value for the current feature
                    for(int s=0;s<sign.length;s++)//search by both increasing and decreasing
                    {
                        double step = 0.001;
                        if(origWeight != 0.0 && step > 0.5 * Math.abs(origWeight))
                            step = stepBase * Math.abs(origWeight);
                        double totalStep = step;
                        for(int j=0;j<nMaxIteration;j++)
                        {
                            double w = origWeight + totalStep * sign[s];
                            weight_change = w - weight[wids[i]];//weight_change is used in the "else" branch in the procedure rank()
                            weight[wids[i]] = w;
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
                                bestWeightValue = weight[wids[i]];
                                succeeds = true;

                                String bw = ((bestWeightValue>0.0)?"+":"") + SimpleMath.round(bestWeightValue, 4);
                                PRINTLN(new int[]{9, 8, 7}, new String[]{wids[i]+"", bw+"", SimpleMath.round(bestScore, 4)+""});
                            }
                            step *= stepScale;
                            totalStep += step;
                        }
                        if(succeeds)
                            break;//no need to search the other direction (e.g. sign = '-')
                        else
                        {
                            weight_change = origWeight - weight[wids[i]];
                            //so that we can start searching in the other direction (since the optimization in the first direction failed)
                            weight[wids[i]] = origWeight;//restore the weight to its initial value
                        }
                    }
                    if(succeeds)
                    {
                        weight_change = bestWeightValue - weight[wids[i]];
                        weight[wids[i]] = bestWeightValue;
                        consecutive_fails = 0;//since we found a better weight value
                        //then normalize the new weight vector
                        double sum = normalize(weight);
                        scaleCached(sum);
                        copy(weight, bestWeight);
                    }
                    else
                    {
                        consecutive_fails++;
                        weight_change = origWeight - weight[wids[i]];
                        //Restore the orig. weight value
                        weight[wids[i]] = origWeight;
                    }
                }
                PRINTLN("------------------------------");
                //if we haven't made much progress then quit
                if(bestScore - startScore < tolerance)
                    break;
            }
            //update the (global) best model with the best model found in this round
            if(bestModel == null || bestScore > bestModelScore)
            {
                bestModelScore = bestScore;
                bestModel = bestWeight;
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
        for(int i=0;i<rl.size();i++)
            score[i] = eval(rl.get(i));
		int[] idx = MergeSorter.sort(score, false);
		return new RankList(rl, idx);
	}
	public double eval(DataPoint p)
    {
        int k = (int)p.getFeatureValue(features[0]);
		double score = 0.0;
        for(int i=1; i<=k; i++) {
            double w = 0;
            for(int j=1; j<=f; j++) {
                double numer = p.getFeatureValue(k*j+i) * weight[f+j];
                double denom = 1 - weight[j] + weight[j] * p.getFeatureValue(k*j);
                if(0 < denom) w += numer / denom;
            }
            score += w / (w + weight[0]) * p.getFeatureValue(k);
        }
		return score;
	}
	public Ranker clone()
	{
		return new BM25F();
	}
	public String name()
	{
		return "BM25F";
	}
    protected int[] getShuffledWeights()
    {
        int[] wids = new int[weight.length];
        List<Integer> l = new ArrayList<Integer>();
        for(int i=0;i<weight.length;i++)
            l.add(i);
        Collections.shuffle(l);
        for(int i=0;i<l.size();i++)
            wids[i] = l.get(i);
        return wids;
    }
	public double distance(BM25F ca)
	{
		return getDistance(weight, ca.weight);
	}
}