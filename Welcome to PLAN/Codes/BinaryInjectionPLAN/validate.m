function validate(activationPotentiation)

class = 1;

wrong_predict = 0;

disp('validating');

do

photo = sprintf('train/%d.png',class);

img = imread(photo);

img = single(img);
img = rgb2gray(img);

input_layer = img(:);
inputLayer = single(input_layer);

		fileName1 = sprintf('weights/weights1.mat');
    load(fileName1);
	
	
prediction = 'None';

imshow(img);
title(prediction);
hold on

inputLayer = normalization(inputLayer); % inputs in range 0 - 1

output_layer = pred(inputLayer, activationPotentiation, weights1);

i = 1;

do

if max(output_layer) == output_layer(i)
 
 disp(i);
 
 prediction = i;
 
end
 
i++;
 
until(i > 9)


imshow(img);
title(prediction);
hold on

if prediction ~= class

wrong_predict++;

end

pause(1) % wait 1 sec

class++;

until(class > 9)

disp(['Wrong predict count: ' num2str(wrong_predict)]);

key = input('press "y" for test or "n" for re-train.', 's');
    
    if strcmp(key, 'y')
        improved_test(activationPotentiation)
    elseif strcmp(key, 'n')
		delete_model()
	end
	
	
end
