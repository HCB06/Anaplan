function train()

class = 1;

activationPotential = 0.5;

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
weights2 = eye(9); % Diagonal matrix


inputLayer = normalization(inputLayer); % inputs in range 0 - 1

%% FEATURE EXTRACTION LAYER %%
 inputLayer(inputLayer < activationPotential) = 0;
 inputLayer(inputLayer > activationPotential) = 1;
 weights1(class,:) = inputLayer;


if class ~= 1
	
		newWeights1 = weights1;
		newWeights2 = weights2;
		
		fileName1 = sprintf('weights/weights1.mat');
    load(fileName1);
    
		fileName2 = sprintf('weights/weights2.mat');
    load(fileName2);
		
		
		weights1 += newWeights1;
		
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

validate(activationPotential) % with train inputs
