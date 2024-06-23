function [output_layer] = pred(inputLayer, weights1)

featureLayer = weights1 * inputLayer;

output_layer = featureLayer;
