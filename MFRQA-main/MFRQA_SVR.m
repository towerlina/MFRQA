function score = MFRQA_SVR(img1,img2)
%% svm module
load model;
FeatureTest = MFRQA_features(img1,img2);
[score, ~, ~] = svmpredict(1,FeatureTest,model);
end


