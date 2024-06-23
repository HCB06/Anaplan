function [output_layer] = pred(inputLayer, activationPotentiation, weights1)

inputLayer = normalization(inputLayer); % inputs in range 0 - 1
inputLayer(inputLayer < activationPotentiation) = 0;
inputLayer(inputLayer > activationPotentiation) = 1;

featureLayer = weights1 * inputLayer;

output_layer = featureLayer;
