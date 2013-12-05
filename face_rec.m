function [status,image_name] = face_rec(input_image,images,num_images) 
input_dir = 'croppedimages/';
image_dims = [48, 64];
%inputimage1=imread('3.jpg'); 
input_image = imresize(input_image,[48 64]);
filenames = dir(fullfile(input_dir, '*.jpg'));
%num_images = numel(filenames);
%images = [];
%for n = 1:num_images
%    filename = fullfile(input_dir, filenames(n).name);
%    img1 = imread(filename);
%    img = imresize(img1,[64 48]);
%    if n == 1
%        images = zeros(prod(image_dims), num_images);
%    end
%    images(:, n) = img(:);
%end

% steps 1 and 2: find the mean image and the mean-shifted input images
mean_face = mean(images, 2);
shifted_images = images - repmat(mean_face, 1, num_images);
 
% steps 3 and 4: calculate the ordered eigenvectors and eigenvalues
[evectors, score, evalues] = princomp(images');
 
% step 5: only retain the top 'num_eigenfaces' eigenvectors (i.e. the principal components)
num_eigenfaces = 20;
evectors = evectors(:, 1:num_eigenfaces);
 
% step 6: project the images into the subspace to generate the feature vectors
features = evectors' * shifted_images;

% calculate the similarity of the input to each training image
input_image = im2double(input_image);
feature_vec = evectors' * (input_image(:) - mean_face);
similarity_score = arrayfun(@(n) 1 / (1 + norm(features(:,n) - feature_vec)), 1:num_images);
 
% find the image with the highest similarity
[match_score, match_ix] = max(similarity_score);
 
% display the result
%figure, imshow([input_image reshape(images(:,match_ix), image_dims)]);
%title(sprintf('matches %s, score %f', filenames(match_ix).name, match_score));
if match_score>=0.005
    %figure, imshow([input_image reshape(images(:,match_ix), image_dims)]);
    %title(sprintf('Matches %s, score %f', filenames(match_ix).name, match_score));
    status=1;
    image_name=filenames(match_ix).name;
    %exit(0)
else
    %title(sprintf('UNMatched %d', match_score));
    status=0;
    image_name=0;
end

% display the eigenvectors
%figure;
%for n = 1:num_eigenfaces
%    subplot(2, ceil(num_eigenfaces/2), n);
%    evector = reshape(evectors(:,n), image_dims);
%    imshow(evector);
%end

% display the eigenvalues
%normalised_evalues = evalues / sum(evalues);
%figure, plot(cumsum(normalised_evalues));
%xlabel('No. of eigenvectors'), ylabel('Variance accounted for');
%xlim([1 30]), ylim([0 1]), grid on;

