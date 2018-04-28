% Austin Welch
% EC503 Project: FastICA Algorithm for Blind Source Separation of images

% (If results don't look good on first run try again)

% clear vars and window
clear all; clc;

%% Load, resize, mix images

% load 3 images of similar size
Orig1 = imread('img2.jpg');
Orig2 = imread('img1.jpg');
Orig3 = imread('img4.jpg');

% Plot original images
subplot(3,3,1);
imshow(Orig1);
title('Original 1'); 
subplot(3,3,2);
imshow(Orig2);
title('Original 2');
subplot(3,3,3);
imshow(Orig3);
title('Original 3');

% get average height and width
avgH = round(mean([size(Orig1,1),size(Orig2,1),size(Orig3,1)]));
avgW = round(mean([size(Orig1,2),size(Orig2,2),size(Orig3,2)]));

% resize all to same (average) size
Orig1 = imresize(Orig1,[avgH avgW]);
Orig2 = imresize(Orig2,[avgH avgW]);
Orig3 = imresize(Orig3,[avgH avgW]);

% dimensions
dimensions = size(Orig1); 

% easy way to reshape 3D rgb matrices to 1D vectors 
vecOrig1 = Orig1(:)';
vecOrig2 = Orig2(:)';
vecOrig3 = Orig3(:)';

% stack image vectors
MatOrig = [vecOrig1; vecOrig2; vecOrig3];     

% mixing matrix
mixingMat = rand(size(MatOrig,1));   

% mix
MixedImgMat = mixingMat*double(MatOrig); 

% pull out mixed images to display (will use MixedImgMat [same thing])
mixedImg1 = uint8(reshape(MixedImgMat(1,:),dimensions)); 
mixedImg2 = uint8(reshape(MixedImgMat(2,:),dimensions)); 
mixedImg3 = uint8(reshape(MixedImgMat(3,:),dimensions)); 

% display mixed images
subplot(3,3,4);
imshow(mixedImg1);
title('Mixed 1'); 
subplot(3,3,5);
imshow(mixedImg2);
title('Mixed 2');
subplot(3,3,6);
imshow(mixedImg3);
title('Mixed 3');

%% Preprocessing steps

% Center data

% backup (uncentered) mixed matrix 
savedMixedMat = MixedImgMat;

% [subtract row mean (vector) from each column of the data]
MixedImgMat = bsxfun(@minus, MixedImgMat, mean(MixedImgMat,2));

% PCA Whitening

% Derivation: https://stats.stackexchange.com/questions/95806/how-to-whiten-the-data-using-principal-component-analysis
% Image (simple explanation): http://cvecOrig231n.github.io/assets/nn2/prepro2.jpeg
%{
Whitening ensures that all dimensions are treated equally a priori before 
the algorithm is run.

A whitening transformation or sphering transformation is a linear 
transformation that transforms a vector of random variables with a known 
covariance matrix into a set of new variables whose covariance is the 
identity matrix meaning that they are uncorrelated and all have variance 1.
The transformation is called "whitening" because it changes the input 
vector into a white noise vector.
%}

% covariance matrix of mixed signal
Sigma = cov(MixedImgMat'); 

% PCA (eigendecomposition of covariance matrix)
% V is orthogonal rotation matrix composed of eigenvectors of cov. matrix
% D is diagonal matrix of eigenvalues
% Matrix V' gives a rotation needed to de-correlate the data (maps original
% features to principal components)
[V,D]=eig(Sigma);  

% after the rotation each component will have variance given by a 
% corresponding eigenvalue. So to make variances equal to 1, you need to 
% divide by the square root of lambda (D)

% So, inv(sqrt(D)) just amounts to dividing by the standard deviation
% and V' rotates the data so that the x-y coordinates are along the
% principal components

% whitening matrix
WhiteMat = inv(sqrt(D)) * (V)'; %#ok<MINV> % could also do: sqrt(D) / V'

% Apply whitening transformation to centered data
% covariance matrix of X' should now == I (identity matrix)
X = WhiteMat*MixedImgMat;    

%% FastICA
% https://en.wikipedia.org/wiki/FastICA

% Multiple component extraction
% https://en.wikipedia.org/wiki/FastICA#Multiple_component_extraction
                    
W = zeros(size(mixingMat)); % Weights matrix (3x3)  
M = size(X,2); % Length of each component (number of pixels in an image)
C = length(mixingMat); % number of desired components

% loop through each component
for p=1:C 
    
    % max number of iterations
    Tmax = 10000;    
    
    % random vector of length N (N-dimensional sample, C <= N [3])
    w_p = rand(size(dimensions,2),1);                
    
    % normalize weight vector
    w_p = w_p/norm(w_p,2);     

    % pseudo do...while loop
    condition = true;  
    while condition 
        % save previous weight for convergence checking
        wpPrevious = w_p;      
        
        % To measure non-Gaussianity, FastICA relies on a nonquadratic 
        % nonlinearity function f(u), its first derivative g(u), 
        % and its second derivative g'(u).
        
        % input of nonlinearity function first and second derivatives
        u = w_p' * X;
        % first derivative of nonlinearity function logcosh(u)
        %g = tanh(u); 
        % first derivative of robust nonlinearity function -exp(-u.^2/2)
        g = u .* exp(-u.^2 ./ 2); % more robust nonlinearity f
        % second derivative of nonlinearity function
        %dg = 1 - tanh(u).^2;
        dg = (1 - u.^2).*exp(-u.^2 ./ 2);
        
        % Update weights
        w_p = (1/M)*X*g' - (1/M)*dg*ones(M,1)*w_p; % Step 1
        w_p = w_p - (w_p'*(W*W'))'; % Step 2
        w_p = w_p / norm(w_p); % Step 3

        % convergence check
        threshold = 1.0*exp(-15); 
        if abs(abs(w_p) - abs(wpPrevious)) < threshold
            condition = false;
            W(:,p) = w_p; % insert weight vector into weight matrix
        end            
    end    
end 

%% Determine good amplitudes

% Now ICA can't recover the amplitudes, 
% so you need to take the whitened but uncentered version
% of the data and multiply it by a constant to restore the original
% luminance which was reduced from the weight factor.

% Find a constant to balance the average pixel luminance:
% Since this doesn't necessarily balance each image individually, 
% take the best matrix for each row's average and then only use that
% corresponding row from the matrix
target = 255/2; % try to make avg pixel value 255/2 (middle value)
minAvg1 = 255; minAvg2 = 255; minAvg3 = 255; % starting mins
% best separated components matrix (one for each)
S1 = zeros(size(X)); S2 = zeros(size(X)); S3 = zeros(size(X));
for i=1:100
    P = abs(i*W'*WhiteMat*savedMixedMat);
    avg1 = mean(P(1,:));
    avg2 = mean(P(2,:));
    avg3 = mean(P(3,:));     
    if abs(target - avg1) < minAvg1
        minAvg1 = abs(target - avg1);
        S1 = P;
    end  
    if abs(target - avg2) < minAvg2
        minAvg2 = abs(target - avg2);
        S2 = P;
    end  
    if abs(target - avg3) < minAvg3
        minAvg3 = abs(target - avg3);
        S3 = P;
    end  
end

%% Display separated images

% separate and reshape images
Separated1 = uint8(reshape(S1(1,:),dimensions)); 
Separated2 = uint8(reshape(S2(2,:),dimensions)); 
Separated3 = uint8(reshape(S3(3,:),dimensions)); 

% display
subplot(3,3,7);
imshow(Separated1);
title('Separated 1'); 
subplot(3,3,8);
imshow(Separated2);
title('Separated 2');
subplot(3,3,9);
imshow(Separated3);
title('Separated 3');

% write images to file
imwrite(Separated1 ,'output1.png','png');
imwrite(Separated2 ,'output2.png','png');
imwrite(Separated3 ,'output3.png','png');

%% 

% Evaluate separated images

