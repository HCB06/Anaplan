function train()

class = 1;


disp('training');

do

photo = sprintf('train/%d.png',class);

img = imread(photo);

img = rgb2gray(img);

imshow(img);
title("Training Photo");

inputLayer = img(:);
inputLayer = single(inputLayer);

weights1 = ones(9,784); % Matrix formed by ones
weights2 = ones(9,9); % Matrix formed by ones


if class == 1 

	weights1(2:9,:) = 0; % All except the first neuron are zero (synaptic pruning) 

	weights2(2:9,:) = 0; % All except the first neuron are zero (synaptic pruning) 

	elseif class == 2 

		weights1(1,:) = 0;
		weights1(3:9,:) = 0;
		 
		weights2(1,:) = 0;
		weights2(3:9,:) = 0;

		elseif class == 3

			weights1(1:2,:) = 0;
			weights1(4:9,:) = 0;

			weights2(1:2,:) = 0;
			weights2(4:9,:) = 0;

			elseif class == 4

				weights1(1:3,:) = 0;
				weights1(5:9,:) = 0;
				 
				weights2(1:3,:) = 0;
				weights2(5:9,:) = 0;

				elseif class == 5 

					weights1(1:4,:) = 0;
					weights1(6:9,:) = 0;

					weights2(1:4,:) = 0;
					weights2(6:9,:) = 0;

					elseif class == 6 

						weights1(1:5,:) = 0;
						weights1(7:9,:) = 0;
						 
						weights2(1:5,:) = 0;
						weights2(7:9,:) = 0;

						elseif class == 7

							weights1(1:6,:) = 0;
							weights1(8:9,:) = 0;

							weights2(1:6,:) = 0;
							weights2(8:9,:) = 0;

							elseif class == 8 

								weights1(1:7,:) = 0;
								weights1(9,:) = 0;

								weights2(1:7,:) = 0;
								weights2(9,:) = 0;

								elseif class == 9 

									weights1(1:8,:) = 0;

									weights2(1:8,:) = 0;
									
end

inputLayer = normalization(inputLayer); % inputs in range 0 - 1

%% FEATURE EXTRACTION LAYER %%
  
  weights1(class,:) = inputLayer;
  fexLayer = (weights1 * inputLayer);


fexLayer = normalization(fexLayer); % fex neurons in range 0 - 1
  
%% CATALYST LAYER %%
  fexConnections = find(fexLayer == 0);
   weights2(:,fexConnections(:)) = 0;
  catLayer = (weights2 * fexLayer);
  
catLayer


if class ~= 1
	
		newWeights1 = weights1;
		newWeights2 = weights2;
		
		fileName1 = sprintf('weights/weights1.mat');
    load(fileName1);
    
		fileName2 = sprintf('weights/weights2.mat');
    load(fileName2);
		
		
		weights1 += newWeights1;
		weights2 += newWeights2;
		
end


		fileName1 = sprintf('weights/weights1.mat');
	save(fileName1, 'weights1');

		fileName2 = sprintf('weights/weights2.mat');
	save(fileName2, 'weights2'); 
		
pause(1); % Wait 1 sec.

class++;

until(class > 9)

disp('train finished');

disp('validation starting in 3..');

pause(3)

validate() % with train inputs
