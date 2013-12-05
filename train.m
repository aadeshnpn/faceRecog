%Image Training Script
%@author: Aadesh Neupane
%aadeshnpn.com.np
%
clear all
clc
%Detect objects using Viola-Jones Algorithm

%Read the input image
%I = imread('sample.jpg');
input_dir = 'trainingImages/';
image_dims = [480, 640];
 
filenames = dir(fullfile(input_dir, '*.jpg'));
num_images = numel(filenames);
images = [];
for n = 1:num_images
    filename = fullfile(input_dir, filenames(n).name);
    img1 = imread(filename);
    img = imresize(img1,[480 640]);
    %To detect Face
    FDetect = vision.CascadeObjectDetector;
    %Returns Bounding Box values based on number of objects
    BB = step(FDetect,img);
    if size(BB)~=0
        %figure,
        %imshow(img); hold on
        for i = 1:size(BB,1)
            rectangle('Position',BB(i,:),'LineWidth',5,'LineStyle','-','EdgeColor','r');
        end
        %croppingRectangle = BB(:,:);
        face1 = imcrop(img, BB(:,:));
        face=rgb2gray(face1);
        face=imresize(face,[320 243]);
        %figure, 
        %imshow(face);
        name=strcat('croppedimages/',num2str(n),'aadesh.jpg'); %%Change the name according the person's name
        imwrite(face,name);
        title('Face Detection');
        hold off;
    end

    %if n == 1
    %    images = zeros(prod(image_dims), num_images);
    %end
    %images(:, n) = img1(:);
end


%
%figure,
%imshow(I); hold on
%for i = 1:size(BB,1)
%    rectangle('Position',BB(i,:),'LineWidth',5,'LineStyle','-','EdgeColor','r');
%end
%croppingRectangle = BB(:,:);
%face = imcrop(I, BB(:,:));
%figure, 
%imshow(face);
%imwrite(face,'face.jpg');
%title('Face Detection');
%hold off;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%input_dir = '/path/to/my/images';
%image_dims = [48, 64];
 
%filenames = dir(fullfile(input_dir, '*.png'));
%num_images = numel(filenames);
%images = [];
%for n = 1:num_images
 %   filename = fullfile(input_dir, filenames(n).name);
 %   img = imread(filename);
 %   if n == 1
 %       images = zeros(prod(image_dims), num_images);
%    end
 %   images(:, n) = img(:);
%end


