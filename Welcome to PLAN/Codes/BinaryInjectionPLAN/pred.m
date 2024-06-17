function [output_layer] = pred(inputLayer, activationPotential, weights1, weights2)

inputLayer(inputLayer < activation_potential) = 0
inputLayer(inputLayer > activation_potential) = 1
featureLayer = weights1 * inputLayer

catalystLayer = weights2 * featureLayer;

output_layer = catalystLayer;
