%% LDA at each timepoint using MATLAB functions

% Load in data
load('C:\Users\JoshB\OneDrive\Documents\University\PhD\workingmemory\Code\data\coef_FIR1000.mat');
load('C:\Users\JoshB\OneDrive\Documents\University\PhD\workingmemory\Code\data\coef_FIRwrong1000.mat');


% Assume X is your data matrix of size (24x4x15)
% Y is your label vector of size (24x1)

% Example data setup (replace with your actual data)
% X = rand(24, 4, 15); % Dummy data
% Y = [labels]; % Actual labels
coef_FIR = coef_FIR1000;
coef_FIRwrong = coef_FIRwrong1000;

data = [coef_FIR; coef_FIRwrong];
data_z = zscore(data(:,1:1082,:), [], 2);
data_rs = reshape(permute(data_z, [3 1 2]), [], 1082);

numSubjects = size(data_z, 3);
nRois = size(data_z, 2);
[pc_vec, pc_val, ~, ~, explained, ~] = pca(data_rs);
pc_val_rs = reshape(pc_val, numSubjects, [], nRois);

easy = permute(pc_val_rs(:, 1:15, :), [1 3 2]);
med = permute(pc_val_rs(:, 16:30, :), [1 3 2]);
hard = permute(pc_val_rs(:, 31:45, :), [1 3 2]);
hard_wrong = permute(pc_val_rs(:, 46:60, :), [1 3 2]);

X = [easy; med; hard; hard_wrong];
Y = repelem((1:4)',24,1);
numTimepoints = size(X, 3);

% Define a range of PCA components to test
componentRange = 1:40; % Example range, adjust based on your needs

% Initialize arrays to store results
accuracyEasyVsHard = zeros(numTimepoints, length(componentRange));
accuracyHardVsWrong = zeros(numTimepoints, length(componentRange));
sensitivityEasyVsHard = zeros(numTimepoints, length(componentRange));
sensitivityHardVsWrong = zeros(numTimepoints, length(componentRange));
specificityEasyVsHard = zeros(numTimepoints, length(componentRange));
specificityHardVsWrong = zeros(numTimepoints, length(componentRange));
cvLossEasyVsHard = zeros(numTimepoints, length(componentRange));
cvLossHardVsWrong = zeros(numTimepoints, length(componentRange));

% Loop over each timepoint
for t = 1:numTimepoints
    % Reshape data for current timepoint
    X_t = squeeze(X(:, :, t)); % Size (96x502)
       
    % Loop over each number of PCA components
    for k = 1:length(componentRange)
        numPCAComponents = componentRange(k);
        X_pca = X_t(:, 1:numPCAComponents); % Reduce dimensions
        
        % Create labels for easy-correct vs. hard-correct
        Y1 = zeros(size(X_pca, 1), 1);
        Y1(Y == 1) = 1; % Easy-correct
        Y1(Y == 3) = 2; % Hard-correct

        X_train1 = X_pca(Y1 ~= 0, :);
        Y_train1 = Y1(Y1 ~= 0);
        
        % Fit LDA model for easy-correct vs. hard-correct
        lda1 = fitcdiscr(X_train1, Y_train1, 'DiscrimType','linear', 'ClassNames', [1, 2], ...
            'CrossVal', 'on', 'KFold', 5);
        %predictions1 = predict(lda1, X_train1);
        predictions1 = kfoldPredict(lda1);
        cvLossEasyVsHard(t, k) = kfoldLoss(lda1);
        %disp(cvLoss);

        % Make confusion matrix
        confmat1 = confusionmat(Y_train1, predictions1);
        % Calculate accuracy, sensitivity, specificity
        %accuracyEasyVsHard(t, k) = sum(predictions1 == Y_train1) / numSubjects;
        accuracyEasyVsHard(t, k) = sum(predictions1 == Y_train1) / sum(confmat1,'all');
        sensitivityEasyVsHard(t, k) = confmat1(1,1) / (confmat1(1,1) + confmat1(2,1));
        specificityEasyVsHard(t, k) = confmat1(2,2) / (confmat1(2,2) + confmat1(1,2));
        
        % Create labels for hard-correct vs. hard-wrong
        Y2 = zeros(size(X_pca, 1), 1);
        Y2(Y == 3) = 1; % Hard-correct
        Y2(Y == 4) = 2; % Hard-wrong

        X_train2 = X_pca(Y2 ~= 0, :);
        Y_train2 = Y2(Y2 ~= 0);
        
        % Fit LDA model for hard-correct vs. hard-wrong
        lda2 = fitcdiscr(X_train2, Y_train2, 'DiscrimType', 'linear', 'ClassNames', [1, 2], ...
            'CrossVal', 'on', 'KFold', 5);
        %predictions2 = predict(lda2, X_train2);
        predictions2 = kfoldPredict(lda2);
        cvLossHardVsWrong(t, k) = kfoldLoss(lda2);

        % Make confusion matrix
        confmat2 = confusionmat(Y_train2, predictions2);
        % Calculate accuracy, sensitivity, specificity
        %accuracyHardVsWrong(t, k) = sum(predictions2 == Y_train2) / numSubjects;
        accuracyHardVsWrong(t, k) = sum(predictions2 == Y_train2) / sum(confmat2,'all');
        sensitivityHardVsWrong(t, k) = confmat2(1,1) / (confmat2(1,1) + confmat2(2,1));
        specificityHardVsWrong(t, k) = confmat2(2,2) / (confmat2(2,2) + confmat2(1,2));

        disp([t k])
    end
end

% Calculate the average accuracy across timepoints
%avgAccuracyEasyVsHard = mean(accuracyEasyVsHard, 1);
%avgAccuracyHardVsWrong = mean(accuracyHardVsWrong, 1);

% Find the best number of PCA components
%[~, bestComponentsEasyVsHard] = max(avgAccuracyEasyVsHard);
%[~, bestComponentsHardVsWrong] = max(avgAccuracyHardVsWrong);

%fprintf('Best number of PCA components for easy-correct vs. hard-correct: %d\n', componentRange(bestComponentsEasyVsHard));
%fprintf('Best number of PCA components for hard-correct vs. hard-wrong: %d\n', componentRange(bestComponentsHardVsWrong));

% Plot accuracy per timepoint across PCs
%figure; plot(accuracyEasyVsHard');
%[~, ord] = sort(accuracyEasyVsHard,1,'descend');
%figure; histogram(ord(1,:));

%figure; plot(accuracyHardVsWrong');
%[~, ord2] = sort(accuracyHardVsWrong,1,'descend');
%figure; histogram(ord2(1,:));

% Check overall
%figure; histogram([ord(1,:) ord2(1,:)]); % Time = 7

% Check cross-validation performance
% Get minimum error/best performance for differing number of PCs
minCVEasyVsHard = min(cvLossEasyVsHard,[],1);
minCVHardVsWrong = min(cvLossHardVsWrong,[],1);

% Find which timepoints
minTime1 = cvLossEasyVsHard==minCVEasyVsHard;
minTime2 = cvLossHardVsWrong==minCVHardVsWrong;

% Sum across to summarise which were most accurate
minTotal1 = sum(minTime1,2);
minTotal2 = sum(minTime2,2);
% Visualise
figure; 
plot(minTotal1,'LineWidth',2);
hold on
plot(minTotal2,'LineWidth',2);
set(gca,'box','off','FontSize',24, 'FontName', 'Arial','linew',1.5);
axis('tight');


% Calculate Balanced Accuracy
bAcc_EasyVsHard = (sensitivityEasyVsHard + specificityEasyVsHard) ./ 2;
bAcc_HardVsWrong = (sensitivityHardVsWrong + specificityHardVsWrong) ./ 2;

% Sort Balanced Accuracy
[sort1, ord1] = sort(bAcc_EasyVsHard, 'descend');
[sort2, ord2] = sort(bAcc_HardVsWrong, 'descend');
figure; histogram(ord1(1,:));
figure; histogram(ord2(1,:));

% Get maximum accuracy for each number of PCs
maxbAcc1 = max(bAcc_EasyVsHard,[],1);
maxbAcc2 = max(bAcc_HardVsWrong,[],1);
maxAcc1 = max(accuracyEasyVsHard,[],1);
maxAcc2 = max(accuracyHardVsWrong,[],1);

% Find which timepoints had maximum accuracy
maxTime1 = bAcc_EasyVsHard==maxbAcc1;
maxTime2 = bAcc_HardVsWrong==maxbAcc2;
% maxTime3 = accuracyEasyVsHard==maxAcc1;
% maxTime4 = accuracyHardVsWrong==maxAcc2;
% Sum across to summarise which were most accurate
maxTotal1 = sum(maxTime1,2);
maxTotal2 = sum(maxTime2,2);
% maxTotal3 = sum(maxTime3,2);
% maxTotal4 = sum(maxTime4,2);
% Visualise
figure; 
plot(maxTotal1,'LineWidth',2);
hold on
plot(maxTotal2,'LineWidth',2);
%plot(maxTotal3,'LineWidth',2);
%plot(maxTotal4,'LineWidth',2);
set(gca,'box','off','FontSize',24, 'FontName', 'Arial','linew',1.5);
axis('tight');

% Find the best number of PCA components for Timepoints 6 and 15
[~, bestComponentsEasyVsHard] = max(bAcc_EasyVsHard(6,:));
[~, bestComponentsHardVsWrong] = max(bAcc_HardVsWrong(15,:)); 

fprintf('Best number of PCA components for easy-correct vs. hard-correct: %d\n', componentRange(bestComponentsEasyVsHard));
fprintf('Best number of PCA components for hard-correct vs. hard-wrong: %d\n', componentRange(bestComponentsHardVsWrong));
% 5 and 12 PCs respectively --> Choose 10 PCs
sum(explained(1:12)) % 35.9583% variance explained --> 1st PC where explained variance < 1%

% Plot explained variance (first 50 PCs)
figure;
scatter(1:50,explained(1:50),50,'black','filled');
hold on
plot(explained(1:50),'-k','LineWidth',2);
yline(1,'--k','LineWidth',1);
set(gca,'box','off','FontSize',24,'FontName','Arial','TickDir','out','linew',1.5);
axis('tight')


%% Rerun but only for the chosen timepoint and PC

clear all
% Load in data
load('C:\Users\JoshB\OneDrive\Documents\University\PhD\workingmemory\Code\data\coef_FIR1000.mat');
load('C:\Users\JoshB\OneDrive\Documents\University\PhD\workingmemory\Code\data\coef_FIRwrong1000.mat');


% Assume X is your data matrix of size (24x4x15)
% Y is your label vector of size (24x1)

% Example data setup (replace with your actual data)
% X = rand(24, 4, 15); % Dummy data
% Y = [labels]; % Actual labels
coef_FIR = coef_FIR1000;
coef_FIRwrong = coef_FIRwrong1000;

data = [coef_FIR; coef_FIRwrong];
data_z = zscore(data(:,1:1082,:), [], 2);
data_rs = reshape(permute(data_z, [3 1 2]), [], 1082); % Subjects X time*cond X ROIs

numSubjects = size(data_z, 3);
nRois = size(data_z, 2);
[pc_vec, pc_val, ~, ~, explained, ~] = pca(data_rs);
pc_val_rs = reshape(pc_val, numSubjects, [], nRois);

easy = permute(pc_val_rs(:, 1:15, :), [1 3 2]);
med = permute(pc_val_rs(:, 16:30, :), [1 3 2]);
hard = permute(pc_val_rs(:, 31:45, :), [1 3 2]);
hard_wrong = permute(pc_val_rs(:, 46:60, :), [1 3 2]);

X = [easy; med; hard; hard_wrong];
Y = repelem((1:4)',24,1);

% Set optimal parameters
t1 = 6;
t2 = 15;
k = 12;

X_t1 = squeeze(X(:, :, t1)); % Size (24x502)  
X_pca1 = X_t1(:, 1:k); % Reduce dimensions
        
% Create labels for easy-correct vs. hard-correct
Y1 = zeros(size(X_pca1, 1), 1);
Y1(Y == 1) = 1; % Easy-correct
Y1(Y == 3) = 2; % Hard-correct

X_train1 = X_pca1(Y1 ~= 0, :);
Y_train1 = Y1(Y1 ~= 0);
        
% Fit LDA model for easy-correct vs. hard-correct
lda1 = fitcdiscr(X_train1, Y_train1, 'DiscrimType','linear', 'ClassNames', [1, 2]);
% Calculate eigenvalues and eigenvectors
[W, LAMBDA] = eig(lda1.BetweenSigma, lda1.Sigma);
lambda = diag(LAMBDA);
[lambda, SortOrder] = sort(lambda, 'descend');
lda_vec1 = W(:, SortOrder);
%lda_val1 = X_train1 * W;

% Create labels for hard-correct vs. hard-wrong
X_t2 = squeeze(X(:, :, t2));
X_pca2 = X_t2(:, 1:k);

Y2 = zeros(size(X_pca2, 1), 1);
Y2(Y == 3) = 1; % Hard-correct
Y2(Y == 4) = 2; % Hard-wrong

X_train2 = X_pca2(Y2~=0,:);
Y_train2 = Y2(Y2~=0);
        
% Fit LDA model for hard-correct vs. hard-wrong
lda2 = fitcdiscr(X_train2, Y_train2, 'DiscrimType', 'linear', 'ClassNames', [1, 2]);
% Calculate eigenvalues and eigenvectors
[W2, LAMBDA2] = eig(lda2.BetweenSigma, lda2.Sigma);
lambda2 = diag(LAMBDA2);
[lambda2, SortOrder2] = sort(lambda2, 'descend');
lda_vec2 = W2(:, SortOrder2);
%lda_val2 = X_train2 * W2;

% Get coefficients of LDA
% coeff1 = lda1.Coeffs(1,2).Linear;
% coeff2 = lda2.Coeffs(1,2).Linear;

% Check if vectors are normalised
norm(lda_vec1(:,1))
norm(lda_vec2(:,1))
% Normalise vectors
lda1_n = lda_vec1(:,1) / norm(lda_vec1(:,1));
lda2_n = lda_vec2(:,1) / norm(lda_vec2(:,1));
norm(lda1_n)
norm(lda2_n)

% Check orthogonality of LDA vectors
dot(lda1_n,lda2_n)
% Orthogonalise LDA vectors
Q = orth([lda1_n lda2_n]);
dot(Q(:,1),Q(:,2))
% Check correlation to original vectors
[Q_corr, pval] = corr(Q,[lda1_n lda2_n]);

% Project original data into LDA space
X_lda1 = X_train1 * (-1.*Q(:,1));
X_lda2 = X_train2 * (-1.*Q(:,2));

% Check separation
figure; histogram(X_lda1(1:24));
hold on
histogram(X_lda1(25:end));

figure; histogram(X_lda2(1:24));
hold on
histogram(X_lda2(25:end));

% Distribution Plots
% Input data
data1 = X_lda1(1:24);
data2 = X_lda1(25:end);
data = [data1;data2];

limits = [min(data) max(data)];

% Group colours
cl1 = [88 80 144]./255;
cl2 = [255 99 97]./255;

% Plot
figure;

[a,b] = ksdensity(data1);

%wdth = 0.5; % width of boxplot
% TODO, should probably be some percentage of max.height of kernel density plot

% density plot
d1 = area(b,a);
set(d1, 'FaceColor', cl1);
set(d1, 'EdgeColor', 'black');
set(d1, 'LineWidth', 1.5);
alpha(d1, 0.5);

hold on
[c,d] = ksdensity(data2);
d2 = area(d,c); 
set(d2, 'FaceColor', cl2);
set(d2, 'EdgeColor', 'black');
set(d2, 'LineWidth', 1.5);
alpha(d2, 0.7);

set(gca,'box','off','FontSize',24,'FontName','Arial','TickDir','out','linew',1.5);
axis('tight');

% Distribution Plot - LDA2 Response
% Input data
data1 = X_lda2(1:24);
data2 = X_lda2(25:end);
data = [data1;data2];

limits = [min(data) max(data)];

% Group colours
cl1 = [255 166 0]./255;
cl2 = [0 63 92]./255;

% Plot
figure;

[a,b] = ksdensity(data1);

%wdth = 0.5; % width of boxplot
% TODO, should probably be some percentage of max.height of kernel density plot

% density plot
d1 = area(b,a);
set(d1, 'FaceColor', cl1);
set(d1, 'EdgeColor', 'black');
set(d1, 'LineWidth', 1.5);
alpha(d1, 0.5);

hold on
[c,d] = ksdensity(data2);
d2 = area(d,c); 
set(d2, 'FaceColor', cl2);
set(d2, 'EdgeColor', 'black');
set(d2, 'LineWidth', 1.5);
alpha(d2, 0.7);

set(gca,'box','off','FontSize',24,'FontName','Arial','TickDir','out','linew',1.5);
axis('tight');


% Project for spatial map
lda_vec1 = pc_vec(:,1:k) * (-1.*Q(:,1));
lda_vec2 = pc_vec(:,1:k) * (-1.*Q(:,2));

% Visualise on brain surface
limits = [min(lda_vec1) max(lda_vec1)]; % [-0.1003 0.1173]
surf_schaef1000(lda_vec1(1:1000),limits);
surf_cbm(lda_vec1(455:482),limits);
subcort_plot(lda_vec1); colormap(custom());

limits = [min(lda_vec2) max(lda_vec2)]; % [-0.1039 0.0725]
surf_schaef1000(lda_vec2(1:1000),limits);
surf_cbm(lda_vec2(455:482),limits);
subcort_plot(lda_vec2); colormap(custom());

% Project all original data into LDA space
lda_val1 = pc_val(:,1:k) * (-1.*Q(:,1));
lda_val2 = pc_val(:,1:k) * (-1.*Q(:,2));

% Plot mean loadings per difficulty
lda_val_diff = reshape(lda_val1, numSubjects, []);

mean_ldaval_diff = mean(lda_val_diff,1);
figure;
plot(mean_ldaval_diff(1:15),'LineWidth',3,'Color',[0/255 153/255 227/255]);
%xlabel('Time');
%ylabel('Mean LDA loading');
hold on
plot(mean_ldaval_diff(16:30),'LineWidth',3,'Color',[117/255 134/255 148/255]);
plot(mean_ldaval_diff(31:45),'LineWidth',3,'Color',[227/255 74/255 0/255]);
%plot(mean_ldaval_diff(46:60),'LineWidth',3,'Color','red');
yline(0,'--','LineWidth',1);
set(gca,'box','off','FontSize',24, 'FontName', 'Arial','linew',1.5);
axis('tight');

% Shaded Error Bar - Difficulty
data = lda_val_diff;
figure;
shadedErrorBar(1:15,mean(data(:,1:15),1),std(data(:,1:15),[],1)/sqrt(size(data(:,1:15),1)),{'-','color',[88 80 144]./255,'LineWidth',1.5},0);
hold on
shadedErrorBar(1:15,mean(data(:,16:30),1),std(data(:,16:30),[],1)/sqrt(size(data(:,16:30),1)),{'--','color',[188 80 144]./255,'LineWidth',1.5},0.3);
shadedErrorBar(1:15,mean(data(:,31:45),1),std(data(:,31:45),[],1)/sqrt(size(data(:,31:45),1)),{'-','color',[255 99 97]./255,'LineWidth',1.5},0.3);
yline(0,'--','LineWidth',1);
set(gca,'box','off','FontSize',24,'FontName','Arial','TickDir','out','linew',1.5);
axis('tight');
ylim([-12 8]);

% Plot mean loadings per response
lda_val_resp = reshape(lda_val2, numSubjects, []);

mean_ldaval_resp = mean(lda_val_resp,1);
figure;
plot(mean_ldaval_resp(31:45),'LineWidth',3,'Color',[227/255 74/255 0/255]);
%xlabel('Time');
%ylabel('Mean LDA loading');
hold on
plot(mean_ldaval_resp(46:60),'LineWidth',3,'Color',[0/255 153/255 227/255]);
%plot(mean_ldaval_resp(1:15),'LineWidth',3,'Color',[0/255 153/255 227/255]);
%plot(mean_ldaval_resp(16:30),'LineWidth',3,'Color',[117/255 134/255 148/255]);
yline(0,'--','LineWidth',1);
set(gca,'box','off','FontSize',24, 'FontName', 'Arial','linew',1.5);
axis('tight');

% Shaded Error bar - Response
data = lda_val_resp;
figure;
shadedErrorBar(1:15,mean(data(:,31:45),1),std(data(:,31:45),[],1)/sqrt(size(data(:,31:45),1)),{'-','color',[255 166 0]./255,'LineWidth',1.5},0);
hold on
shadedErrorBar(1:15,mean(data(:,46:60),1),std(data(:,46:60),[],1)/sqrt(size(data(:,46:60),1)),{'-','color',[0 63 92]./255,'LineWidth',1.5},0.3);
yline(0,'--','LineWidth',1);
set(gca,'box','off','FontSize',24,'FontName','Arial','TickDir','out','linew',1.5);
axis('tight');
ylim([-2 5]);

%% LDA Quadrants

% Make quadrant IDs
quad_id = zeros(482,1);
quad_id(lda_vec1>0 & lda_vec2>0) = 1;
quad_id(lda_vec1>0 & lda_vec2<=0) = 2;
quad_id(lda_vec1<=0 & lda_vec2<=0) = 3;
quad_id(lda_vec1<=0 & lda_vec2>0) = 4;

% Plot clusters of interest - using gscatter()
RGB_color = [255 127 80; 189 190 200; 189 190 200; 189 190 200]./255;
figure; 
f = gscatter(lda_vec1,lda_vec2,quad_id,RGB_color,'o',5);
set(f(1),'MarkerEdgeColor',uint8([255 127 80]),'MarkerFaceColor',uint8([255 127 80]));
set(f(2),'MarkerEdgeColor',uint8([189 190 200]),'MarkerFaceColor',uint8([189 190 200]));
set(f(3),'MarkerEdgeColor',uint8([189 190 200]),'MarkerFaceColor',uint8([189 190 200]));
set(f(4),'MarkerEdgeColor',uint8([189 190 200]),'MarkerFaceColor',uint8([189 190 200]));
drawnow
set(f(1).MarkerHandle,'FaceColorType','truecoloralpha','FaceColorData',uint8([255;127;80;255*0.7]));
set(f(2).MarkerHandle,'FaceColorType','truecoloralpha','FaceColorData',uint8([189;190;200;255*0.7]));
set(f(3).MarkerHandle,'FaceColorType','truecoloralpha','FaceColorData',uint8([189;190;200;255*0.7]));
set(f(4).MarkerHandle,'FaceColorType','truecoloralpha','FaceColorData',uint8([189;190;200;255*0.7]));
legend('off');
%xlabel("PC1"); ylabel("PC2");
xlabel(''); ylabel(''); 
xline(0,'--k','LineWidth',2);
yline(0,'--k','LineWidth',2);
set(gca,'box','off','FontSize',24,'FontName','Arial','linew',1.5,'TickDir','out');

% Plot clusters - using scatter()
[~,idx] = sort(quad_id,'ascend');
data1 = lda_vec1(idx);
data2 = lda_vec2(idx);
g1 = sum(quad_id==1); % 264
g2 = sum(quad_id==2); % 233
g3 = sum(quad_id==3); % 257
g4 = sum(quad_id==4); % 328
RGB1 = [255 127 80]./255;
RGB2 = [189 190 200]./255;
RGB_color2 = [repmat(RGB1,g1,1);repmat(RGB2,g2+g3+g4,1)];
cmap = RGB_color2;
figure; 
scatter(data1,data2,50,cmap,'MarkerFaceColor','flat','MarkerFaceAlpha',0.7);
%h1 = lsline();
%h1.Color = 'r';
%h1.LineWidth = 2;
set(gca,'box','off','FontSize',24,'FontName','Arial','linew',1.5,'TickDir','out');
xline(0,'--k','LineWidth',2);
yline(0,'--k','LineWidth',2);
axis('tight')


% Visualise quadrants
limits = [1 4];
surf_schaef2(quad_id(1:400),limits);
surf_cbm(quad_id(455:482),limits);
subcort_plot(quad_id); colormap(custom());

% Visualise quadrant of interest
limits = [0 1];
C1 = quad_id==1; % 264 regions
surf_schaef1000(C1(1:1000),limits);
surf_cbm(C1(1055:1082),limits);
subcort_plot1000(C1,limits); colormap(custom());

