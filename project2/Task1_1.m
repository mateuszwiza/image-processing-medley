%Generate 2D grid of coordinates 
[x,y] = meshgrid(1:0.1:10,1:0.1:10);
%Apply signal function
P = cos(2*pi*(x+y));

%Calculate 2D FFT of the signal
F = fft2(P);

%Display the signal in 3D
surf(P), shading flat;

%Display the magnitude signal FT in 2D
figure;imagesc(abs(fftshift(F)));colormap(gray)
figure;imagesc(log(abs(fftshift(F))));colormap(gray)

%Display the phase signal FT in 2D
figure;imagesc(angle(fftshift(F)));colormap(gray)