function delete_model ()

		fileName1 = sprintf('weights/weights1.mat');
    load(fileName1);
    
	
	
	weights1(:,:) = 0;
	

		fileName1 = sprintf('weights/weights1.mat');
	save(fileName1, 'weights1');
 
	
	disp('Model deleted.');
