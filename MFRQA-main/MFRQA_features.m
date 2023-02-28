function feat = MFRQA_features(img1,img2)
%% pre-processing
%img1 enhanced image
%img2 low-light image
feat = [];
% Darkpiror pre
blocksizerow_d    = 8;
blocksizecol_d    = 8;
[row,col,~]           = size(img1);
patch_row_num         = floor(row/blocksizerow_d);
patch_col_num         = floor(col/blocksizecol_d);
I_dark                = img1(1:patch_row_num*blocksizerow_d,1:patch_col_num*blocksizecol_d,1:3);           
[row,col,~]           = size(I_dark);
patch_row_num         = floor(row/blocksizecol_d);
patch_col_num         = floor(col/blocksizecol_d);
I_dark                = I_dark(1:patch_row_num*blocksizecol_d,1:patch_col_num*blocksizecol_d,1:3);
R_dark  	= double(I_dark(:,:,1))./255;                             
G_dark      = double(I_dark(:,:,2))./255;                             
B_dark  	= double(I_dark(:,:,3))./255;                             
% % RGB
% img1R  	= double(img1(:,:,1))./255;                             % Red
% img1G   = double(img1(:,:,2))./255;                             % Green
% img1B  	= double(img1(:,:,3))./255;                             % Blue
% img2R  	= double(img2(:,:,1))./255;                             
% img2G   = double(img2(:,:,2))./255;                             
% img2B  	= double(img2(:,:,3))./255;                             
% % YIQ
% img1_y = 0.299 * img1R + 0.587 * img1G + 0.114 * img1B;
% img1_i = 0.596 * img1R - 0.274 * img1G - 0.322 * img1B;
% img1_q = 0.211 * img1R - 0.523 * img1G + 0.312 * img1B;
% img2_y = 0.299 * img2R + 0.587 * img2G + 0.114 * img2B;
% img2_i = 0.596 * img2R - 0.274 * img2G - 0.322 * img2B;
% img2_q = 0.211 * img2R - 0.523 * img2G + 0.312 * img2B;
% RGB2YIQ
yiq1 = rgb2ntsc(img1);
yiq2 = rgb2ntsc(img2);
img1_y = yiq1(:,:,1);
img1_i = yiq1(:,:,2);
img1_q = yiq1(:,:,3);
img2_y = yiq2(:,:,1);
img2_i = yiq2(:,:,2);
img2_q = yiq2(:,:,3);

%gary
img1_gray=rgb2gray(img1);
img2_gray=rgb2gray(img2);
[M,N] = size(img1_gray);
% other
blocksizerow    = 96;
blocksizecol    = 96;
blockrowoverlap = 0;
blockcoloverlap = 0;
featnum      = 4;
block_rownum     = floor(M/blocksizerow);
block_colnum     = floor(N/blocksizecol);

if ((M < 11) || (N < 11))
    ssim_index = -Inf;
    ssim_map = -Inf;
    return
end
window1 = fspecial('gaussian', 11, 1.5); %
window2 = fspecial('gaussian',7,7/6);
K(1) = 0.01;     % default settings
K(2) = 0.03;     %
K(3) = 10.^(-5);
L = 255;   


%% Brightness
Brightness = entropy(img1_y);
feat = [feat Brightness];
%% ColorSimilar
im = (2 .* img1_i .* img2_i + eps) ./ (img1_i.^2 + img2_i.^2 + eps);
qm = (2 .* img1_q .* img2_q + eps) ./ (img1_q.^2 + img2_q.^2 + eps);
colorsimilar = real(mean2((im.*qm).^0.05));
feat = [feat colorsimilar];
%% Contrast
I_contrast=im2double(img1_gray);
Ia = 6 * (I_contrast.^ 0.1);
Ia = Ia(:);
meanIa = repmat(mean(Ia),[M*N,1]);
chebyshev = (max(abs(Ia-meanIa))).^0.9;
feat = [feat chebyshev];
%% Darkpiror
Id = min(min(R_dark,G_dark),B_dark);
mu_darkprior  = nanmean(mean(im2col(Id, [blocksizerow_d blocksizecol_d], 'distinct')));
feat = [feat mu_darkprior];
%% Consistency
img1_double = double(img1_gray);
img2_double = double(img2_gray);
% automatic downsampling
f = max(1,round(min(M,N)/256));
%downsampling by f
%use a simple low-pass filter
if(f>1)
    lpf = ones(f,f);
    lpf = lpf/sum(lpf(:));
    img1_con = imfilter(img1_double,lpf,'symmetric','same');
    img2_con = imfilter(img2_double,lpf,'symmetric','same');
    img1_con = img1_con(1:f:end,1:f:end);
    img2_con = img2_con(1:f:end,1:f:end);
else
    img1_con = img1_double;
    img2_con = img2_double;
end

C2 = (K(2)*L)^2;

window1 = window1/sum(sum(window1));

mu1   = filter2(window1, img1_con, 'valid');
mu2   = filter2(window1, img2_con, 'valid');

mu1_sq = mu1.*mu1;
mu2_sq = mu2.*mu2;
mu1_mu2 = mu1.*mu2;

sigma1_sq = filter2(window1, img1_con.*img1_con, 'valid') - mu1_sq;
sigma2_sq = filter2(window1, img2_con.*img2_con, 'valid') - mu2_sq;
sigma12 = filter2(window1, img1_con.*img2_con, 'valid') - mu1_mu2;

c = (2*sigma12 + C2) ./ (sigma1_sq + sigma2_sq + C2);
c = mean2(c);
feat = [feat c];
%% Contour
f3_t=svd(img1_double);
% gini
data = abs(f3_t(:));
N1=length(data);
data = sort(data); 
y=[0 ;cumsum(data)]/sum(data);
a=sum(y(1:end-1)+y(2:end))/(2*N1);
f3_1=(1-2*a);
%hoyer
data2=f3_t(:);
N2=length(data2);
f3_2=(N2^(1/2)-(sum(data2))/((sum(data2.^2))^(1/2)))*(N2^(1/2)-1)^(-1);
prof = [f3_1, f3_2];
feat = [feat prof];
%% Naturalness dim-4
img               = img1_double(1:block_rownum*blocksizerow,1:block_colnum*blocksizecol);   

window2          = window2/sum(sum(window2));
mu               = imfilter(img,window2,'replicate');
mu_sq            = mu.*mu;
sigma            = sqrt(abs(imfilter(img.*img,window2,'replicate') - mu_sq));
structdis        = (img-mu)./(sigma+1);                             
feat_scale       = blkproc(structdis,[blocksizerow blocksizecol], ...
                           [blockrowoverlap blockcoloverlap], ...
                           @computefeature);
feat_scale       = reshape(feat_scale,[featnum,size(feat_scale,1)*size(feat_scale,2)/featnum]); 
feat_scale       = feat_scale';
mu_distparam     = nanmean(feat_scale); 
feat = [feat mu_distparam];
end