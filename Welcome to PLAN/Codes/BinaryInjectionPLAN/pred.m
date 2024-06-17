function [output_layer] = pred(inputLayer, activationPotential, weights1, weights2)

inputLayer = normalization(inputLayer); % inputs in range 0 - 1
inputLayer(inputLayer < activationPotential) = 0;
inputLayer(inputLayer > activationPotential) = 1;
featureLayer = weights1 * inputLayer;

catalystLayer = weights2 * featureLayer;

output_layer = catalystLayer;
