function []=finalreco()
%Final Recoginition System
clear;
%%Load The Training Images
input_dir = 'croppedimages/';
image_dims = [48, 64];
%inputimage1=imread('3.jpg'); 
%input_image = imresize(input_image,[64 48]);
filenames = dir(fullfile(input_dir, '*.jpg'));
num_images = numel(filenames);
images = [];
for n = 1:num_images
    filename = fullfile(input_dir, filenames(n).name);
    img1 = imread(filename);
    img = imresize(img1,[64 48]);
    if n == 1
        images = zeros(prod(image_dims), num_images);
    end
    images(:, n) = img(:);
end
save('faces.mat','images')
%% Step 1: Detect a Face To Track
faceDetector = vision.CascadeObjectDetector();
videoFileReader = imaq.VideoDevice('winvideo', 1, 'YUY2_640x480','ROI', [1 1 640 480],'ReturnedColorSpace', 'rgb');
videoFrame = step(videoFileReader);
bbox = step(faceDetector, videoFrame);
boxInserter = vision.ShapeInserter('BorderColor','Custom','CustomBorderColor',[255 255 0]);
videoOut = step(boxInserter, videoFrame, bbox);
figure(1), imshow(videoOut), title('Detected face');
%% Step 2: Identify Facial Features To Track
%[hueChannel,~,~] = rgb2hsv(videoFrame);
%figure, imshow(hueChannel), title('Hue channel data');
%rectangle('Position',bbox(1,:),'LineWidth',2,'EdgeColor',[1 1 0]);
%% Step 3: Track the Face
noseDetector = vision.CascadeObjectDetector('Nose');
faceImage = imcrop(videoFrame,bbox);
noseBBox = step(noseDetector,faceImage);
%noseBBox(1:2) = noseBBox(1:2) + bbox(1:2);
%tracker = vision.HistogramBasedTracker;
%initializeObject(tracker, hueChannel, noseBBox);
videoInfo = info(videoFileReader);
videoPlayer = vision.VideoPlayer('Position',[300 100 670 510]);

nFrames = 0;
while (nFrames < 25)
    videoFrame = step(videoFileReader);
    [hueChannel,~,~] = rgb2hsv(videoFrame);
    bbox = step(faceDetector, videoFrame);
    videoOut = step(boxInserter, videoFrame, bbox);
    face1 = imcrop(videoFrame, bbox(:,:));
    face=rgb2gray(face1);
    %imsave('face.jpg',face);
    [status,file_name]=face_rec(face,images,num_images);
    figure(1),subplot(1,2,1), subimage(face1)
    subplot(1,2,2), subimage(face)
    %figure(1), imshow(face1, image_dims);
    %figure(1), imshow(face1, image_dims);
    title(sprintf('Searching Not Found'));
    step(videoPlayer, videoOut);
    nFrames = nFrames + 1;
end
if (status==1)
        subplot(1,2,2),subimage([input_image reshape(images(:,match_ix), image_dims)]);
        title(sprintf('Matches %s, score %f', filenames(match_ix).name, match_score));
end
release(videoFileReader);
release(videoPlayer);
close all;