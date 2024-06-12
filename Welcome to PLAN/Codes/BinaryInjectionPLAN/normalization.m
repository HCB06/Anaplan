function [normalized_vector] = normalization(vector)

	abs_vector = abs(vector);
    
    max_abs_vector = max(abs_vector);
    
    normalized_vector = vector / max_abs_vector;