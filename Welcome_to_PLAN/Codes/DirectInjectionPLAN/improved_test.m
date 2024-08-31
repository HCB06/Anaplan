function improved_test ()


disp('customize testing');
disp('sıfırlamak için sağ alt köşeye tıklayın. / for reset click right down corner.');

t = 1;

do

photo = sprintf('custom.png');

img = imread(photo);
img = double(img);
%img = rgb2gray(img);

input_layer = img(:);
input_layer = single(input_layer);

		fileName1 = sprintf('weights/weights1.mat');
    load(fileName1);
    

prediction = 'None';

imshow(img);
title(prediction);
hold on

inputLayer = normalization(input_layer); % inputs in range 0 - 1	

output_layer = pred(inputLayer, weights1);



j = 1;

do

 if max(output_layer) == output_layer(j)
 
 prediction = j;
 
 end
 
 j++;
 
 until(j > 9)

prediction

imshow(img);
title(prediction);
hold on

[x, y] = ginput(1);

x = round(x);
y = round(y);

img(y, x) = 255;

if x == 28 && y == 28

img = zeros(28,28);

end


	custom = img;
		imwrite(uint8(custom), 'custom.png');
	
	t++;
	
	until(t>1000)
	
end
