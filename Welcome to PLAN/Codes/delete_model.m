function delete_model ()

		fileName1 = sprintf('weights/weights1.mat');
    load(fileName1);
    
		fileName2 = sprintf('weights/weights2.mat');
    load(fileName2);
	
	
	
	weights1(:,:) = 0;
	
	weights2(:,:) = 0;
	
	
	
		fileName1 = sprintf('weights/weights1.mat');
	save(fileName1, 'weights1');

		fileName2 = sprintf('weights/weights2.mat');
	save(fileName2, 'weights2');  
	
	disp('Model deleted. starting training..');
	
	pause(1);
	
	train()