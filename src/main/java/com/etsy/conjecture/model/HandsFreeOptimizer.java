package com.etsy.conjecture.model;

import com.etsy.conjecture.data.*;

import java.util.*;

/**
 *  *  * Hyperparamer free optimizer, uses NoPesky Learning rate and Adaptive Regularization.
 *   *   * http://www8.gsb.columbia.edu/cbs-directory/sites/cbs-directory/files/publications/indg0711-dalessandro.pdf
 *    *    */
/**Todo  How to handle the raw type and override method*/
public class HandsFreeOptimizer extends SGDOptimizer {

    /**TODO Setting to hard coded for now based on paper above, relook at in the future*/
    private double regularizationLearningRate = 0.001;

    @Override
    public StringKeyedVector getUpdates(Collection minibatch) {
        StringKeyedVector updateVec = new StringKeyedVector();
        int batchSize = minibatch.size()/2;
        if(minibatch instanceof List) {
            List<LabeledInstance> instances = (List<LabeledInstance>) minibatch;
            for (int i = 0; i < batchSize; i++) {
                LabeledInstance validationInstance = instances.get(i);
                LabeledInstance trainInstance = instances.get(i + 1);
                //accumlate gradient
                updateVec.add(getUpdate(trainInstance));
                //update regularization
                updateRegularization(validationInstance);
                model.truncate(trainInstance);
                model.epoch++;
            }
            updateVec.mul(1.0 / minibatch.size()); // do a single update, scaling weights by the
        }
        System.out.println("Regularization Value: " + gaussian);
        //average gradient over the minibatch
        return updateVec;
    
    }
    
    /**
 *     * Update the gradient with a training instance using L2 regularization which supposedly performs
 *         * best for adaptive regularization.
 *             */
    @Override
    public StringKeyedVector getUpdate(LabeledInstance instance) {
        //Using Elastic Net but with no laplace set
        StringKeyedVector gradients = model.getGradients(instance);
        double learningRate = getDecreasingLearningRate(model.epoch);
        gradients.mul(-learningRate);
        return gradients;
    }

    public void updateRegularization(LabeledInstance instance) {
        StringKeyedVector parameters = model.getParam();
        StringKeyedVector gradient = model.getGradients(instance);
        Iterator it = gradient.iterator();

        double gradientSum = 0;
        while(it.hasNext()) {
            Map.Entry<String,Double> pairs = (Map.Entry)it.next();
            String key = pairs.getKey();
            double value = pairs.getValue();
            double param = parameters.getCoordinate(key);
            gradientSum += (value * param);
        }
        double updateValue = regularizationLearningRate * gaussian * gradientSum;
        gaussian += updateValue;
    }
}

