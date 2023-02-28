% Input:  (1) img1: low-light image
%         (2) img2: enhacned image
% Output: (1) score: the quality score
%%
%  Please make sure that
%  demo.m, estimateggdparam.m, computefeature.m MFRQA_features.m, MFRQA_SVR.m, and 
%  libsvm files(including svmpredict.mexw64 svm-predict.exe svmtrain.mexw64 svm-train.exe svm-scale.exe)
%  are in the same directory
clear;
clc;
load model
img = imread(imgpath);
oriimg = imread(oriimgpath);
score = MFRQA_SVR(img, oriimg);