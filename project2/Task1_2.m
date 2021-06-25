%Generate a 10-by-10 pixels image with random pixels intensities
P = peaks(10);
%Repean the image 50 times to get a pattern
X = repmat(P,[5 10]);
%Display the initial image
figure;
imagesc(X);
colormap(gray);

%Calculate FFT of original image
F = fft2(X);
F = fftshift(F);

%Display the magnitude of FT
figure
imagesc(log(abs(F)));
colormap(gray)
%Display the phase of FT
figure
imagesc(angle(F));
colormap(gray)

%Remove strongest frequency of the FT (twice because of symmetry)
F2 = F;
[max_value,max_index] = max(F2(:));
[I_row, I_col] = ind2sub(size(F2),max_index);
F2(I_row,I_col) = 0;
[max_value,max_index] = max(F2(:));
[I_row, I_col] = ind2sub(size(F2),max_index);
F2(I_row,I_col) = 0;

%Display FT after removing strongest frequency
figure
imagesc(abs(F2));
colormap(gray)

%Find inverse 2D FFT
restore = ifft2(F); %without change to FT
restore2 = ifft2(F2);  %after removing strongest frequency

%Display the results
figure
imagesc(abs(restore));
colormap(gray)
figure
imagesc(abs(restore2));
colormap(gray)