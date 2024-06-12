function [output_layer] = pred(inputLayer, activationPotential, weights1, weights2)

fexConnections = find(inputLayer < activationPotential);
weights1(:,fexConnections(:)) = 0;
featureLayer = weights1 * inputLayer;
catalystLayer = weights2 * featureLayer;

output_layer = catalystLayer;