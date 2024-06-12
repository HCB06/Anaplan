function [output_layer] = pred(inputLayer, weights1, weights2)

featureLayer = weights1 * inputLayer;
catalystLayer = weights2 * featureLayer;

output_layer = catalystLayer;