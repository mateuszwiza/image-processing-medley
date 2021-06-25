
%Read the image as grayscale and as martrix of doubles
im = double(rgb2gray(imread('maas.jpg')));
%Get size of the image
[n1,n2] = size(im);

%% ADDING NOISE
%Generate 2D grid of coordinates 
[x,y] = meshgrid(1:n2,1:n1);
%Apply noise function
P = sin(x/2+y/2);

%Add noise to the image (large factor of P results is noise being strongly
%visible in the image)
noisy = (im + 200*P);

%Display the image with noise
figure
imagesc(noisy);
colormap(gray)

%Calculate FFT of original and noisy image
F1 = fftshift(fft2(im));
F2 = fftshift(fft2(noisy));

%Display noisy FT in 1D
figure
plot(log(abs(F2(n1/2,:))));

%Display noisy FT in 2D
figure
imagesc(log(abs(F2)));
colormap(gray)

%Display noisy FT in 3D
figure
surf(log(abs(F2))), shading flat;

%% NOISE REMOVAL - Ring Filter

%Generate matrix of distances from centre
z = sqrt ( ( x - n2/2 ).^2 + ( y - n1/2 ).^2 );
%Create filter by making everything beetween 100 and 110 units from the
%centre black
filter = ( z < 90 | z > 120 );
%Apply filter to the FT of noisy image
filtered = F2 .* filter ;
%Get inverse 2D FFT
result = ifft2 ( filtered );

%Display FT and image after removing noise
figure
imagesc(log(abs(filtered)));
colormap(gray)

figure
imagesc(abs(result));
colormap(gray)


