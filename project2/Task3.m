clear all;
%Read the image as grayscale and as martrix of doubles
im = double(rgb2gray(imread('maas.jpg')));
%Get size of the image
[n1,n2] = size(im);

%Initiate matrix in size of the image with value between -1 and 1
[x ,y ] = meshgrid(-1+2/n2:2/n2:1, -1+2/n1:2/n1:1);

%Calculate 2D FFT of the original image
F = fft2(im);
%Display the 2D FFT
figure;imagesc(log(abs(fftshift(F))));colormap(gray)

%% ADDING BLUR

%Set parameters for bluring function
a = 0.1;
b = 0.1;
k = 0.5;

%Spatial kernel H for motion blur
H = sinc((x*a + y*b)).*exp(-1j*pi*(x*a + y*b));
%H = exp(-k*((x.^2)+(y.^2)).^(5/6));
%Display spatial kernel
figure;imagesc(abs(H));colormap(gray)
%Calculate 2D FFT of blurring function
FH = fft2(H);
%Display the 2D FFT of blurring function
figure;imagesc(log(abs(fftshift(FH))));colormap(gray)

%Apply blurring mask to the FT of the original image
G = F.*H;
%Display the FT after applying blur 
figure;imagesc(log(abs(fftshift(G))));colormap(gray)

%Display blurred image by finding inverse 2D FFT
blurred = ifft2(G);
figure;imagesc(abs(blurred)/255);colormap(gray)

%% ADD RANDOM NOISE

%Adding Gaussian noise with mean 0 and variance 0.01 to the blurred image
noisy = double(imnoise(uint8(abs(blurred)),'gaussian',0,0.01));
figure;imagesc(abs(noisy)/255);colormap(gray)

%Calculate 2D FFT of the blurred image after adding noise
F2 = fft2(noisy);

%% RESTORE

%Get noise power spectrum
noise = noisy-im;
noise_spectrum = abs(fft2(nn)).^2;

%Get original image power spectrum
im_spectrum = abs(F).^2;

%2D Wiener's Transfer Function
dh = abs(H).^2+ noise_spectrum./im_spectrum;
Hw = conj(H)./dh;

%Apply Wiener's filter to FT of noisy and blurred image
R = Hw.*F2;
restored = ifft2(R);
figure;imagesc(abs(restored)/255);colormap(gray)


%MATLAB function for 2D Wiener Filter
r0 = wiener2(abs(noisy));
figure;imagesc(r0/255);colormap(gray)


