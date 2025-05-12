%% LDA at each timepoint using MATLAB functions

% Load in data
load('C:\Users\JoshB\OneDrive\Documents\University\PhD\workingmemory\Code\data\coef_FIR.mat');
load('C:\Users\JoshB\OneDrive\Documents\University\PhD\workingmemory\Code\data\coef_FIRwrong.mat');


% Assume X is your data matrix of size (24x4x15)
% Y is your label vector of size (24x1)

% Example data setup (replace with your actual data)
% X = rand(24, 4, 15); % Dummy data
% Y = [labels]; % Actual labels
data = [coef_FIR; coef_FIRwrong];
data_z = zscore(data(:,1:482,:), [], 2);
data_rs = reshape(permute(data_z, [3 1 2]), [], 482);

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
    X_t = squeeze(X(:, :, t)); % Size (96x482)
       
    % Loop over each number of PCA components
    for k = 1:length(componentRange)
        numPCAComponents = componentRange(k);
        X_pca = X_t(:, 1:numPCAComponents); % Reduce dimensions
        
        % Create labels for easy-correct vs. hard-correct
        Y1 = zeros(size(X_pca, 1), 1);
        Y1(Y == 1) = 1; % Easy-correct
        Y1(Y == 3) = 2; % Hard-correct

        % Create labels for participants
        nSubjects = 24;
        sub_id = repmat([1:24]',2,1);

        X_train1 = X_pca(Y1 ~= 0, :);
        Y_train1 = Y1(Y1 ~= 0);
        
        % Create leave-one-out partition
        cvp = cvpartition(sub_id, 'KFold', 5, 'Stratify', true);

        % Fit LDA model for easy-correct vs. hard-correct
        lda1 = fitcdiscr(X_train1, Y_train1, 'DiscrimType','linear', 'ClassNames', [1, 2]);
        
        % Cross-validate using leave-one-out partition
        lda1_xval = crossval(lda1, 'CVPartition', cvp);
        
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
        lda2 = fitcdiscr(X_train2, Y_train2, 'DiscrimType', 'linear', 'ClassNames', [1, 2]);

        % Cross-validate using leave-one-out partition
        lda2_xval = crossval(lda2, 'CVPartition', cvp);
        
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
% 7 and 10 PCs respectively --> Choose 10 PCs
sum(explained(1:13)) % 44.14% variance explained --> 1st PC where explained variance < 1%

% Plot explained variance (first 50 PCs)
figure;
scatter(1:50,explained(1:50),50,'black','filled');
hold on
plot(explained(1:50),'-k','LineWidth',2);
yline(1,'--k','LineWidth',1);
set(gca,'box','off','FontSize',24,'FontName','Arial','TickDir','out','linew',1.5);
axis('tight')


%% Rerun but only for the chosen timepoint and PC

% Load in data
load('C:\Users\JoshB\OneDrive\Documents\University\PhD\workingmemory\Code\data\coef_FIR.mat');
load('C:\Users\JoshB\OneDrive\Documents\University\PhD\workingmemory\Code\data\coef_FIRwrong.mat');


% Assume X is your data matrix of size (24x4x15)
% Y is your label vector of size (24x1)

% Example data setup (replace with your actual data)
% X = rand(24, 4, 15); % Dummy data
% Y = [labels]; % Actual labels
data = [coef_FIR; coef_FIRwrong];
data_z = zscore(data(:,1:482,:), [], 2);
data_rs = reshape(permute(data_z, [3 1 2]), [], 482); % Subjects X time*cond X ROIs

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
k = 13;

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


% FIGURE 2e -----------------------
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


% FIGURE 2f: -------------------
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
limits = [min(lda_vec1) max(lda_vec1)]; % [-0.1202 0.1341]
surf_schaef2(lda_vec1(1:400),limits);
surf_cbm(lda_vec1(455:482),limits);
subcort_plot(lda_vec1); colormap(custom());

limits = [min(lda_vec2) max(lda_vec2)]; % [-0.1764 0.1437]
surf_schaef2(lda_vec2(1:400),limits);
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


% FIGURE 2e (right) ---------------------
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


% FIGURE 2f (right) -----------------------
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


% FIGURE 3a ------------------------------
% Plot clusters - using scatter()
[~,idx] = sort(quad_id,'ascend');
data1 = lda_vec1(idx);
data2 = lda_vec2(idx);
g1 = sum(quad_id==1); % 108
g2 = sum(quad_id==2); % 120
g3 = sum(quad_id==3); % 133
g4 = sum(quad_id==4); % 121
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
C1 = quad_id==1; % 108 regions
surf_schaef2(C1(1:400),limits);
surf_cbm(C1(455:482),limits);
subcort_plot(C1,limits); colormap(custom());

% Visualise each quadrant separately
% Plot clusters - using scatter()
[~,idx] = sort(quad_id,'ascend');
data1 = lda_vec1(idx);
data2 = lda_vec2(idx);
g1 = sum(quad_id==1); % 108
g2 = sum(quad_id==2); % 120
g3 = sum(quad_id==3); % 133
g4 = sum(quad_id==4); % 121
RGB1 = [120 226 213]./255;
RGB2 = [189 190 200]./255;
RGB_color2 = [repmat(RGB2,g1,1);repmat(RGB1,g2,1);repmat(RGB2,g3+g4,1)];
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

limits = [0 1];
CX = quad_id==2; 
surf_schaef2(CX(1:400),limits);
surf_cbm(CX(455:482),limits);
subcort_plot(CX,limits); colormap(custom());

% Plot clusters - using scatter()
[~,idx] = sort(quad_id,'ascend');
data1 = lda_vec1(idx);
data2 = lda_vec2(idx);
g1 = sum(quad_id==1); % 108
g2 = sum(quad_id==2); % 120
g3 = sum(quad_id==3); % 133
g4 = sum(quad_id==4); % 121
RGB1 = [226 120 133]./255;
RGB2 = [189 190 200]./255;
RGB_color2 = [repmat(RGB2,g1+g2,1);repmat(RGB1,g3,1);repmat(RGB2,g4,1)];
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

limits = [0 1];
CX = quad_id==3; 
surf_schaef2(CX(1:400),limits);
surf_cbm(CX(455:482),limits);
subcort_plot(CX,limits); colormap(custom());

% Plot clusters - using scatter()
[~,idx] = sort(quad_id,'ascend');
data1 = lda_vec1(idx);
data2 = lda_vec2(idx);
g1 = sum(quad_id==1); % 108
g2 = sum(quad_id==2); % 120
g3 = sum(quad_id==3); % 133
g4 = sum(quad_id==4); % 121
RGB1 = [255 105 180]./255;
RGB2 = [189 190 200]./255;
RGB_color2 = [repmat(RGB2,g1+g2+g3,1);repmat(RGB1,g4,1)];
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

limits = [0 1];
CX = quad_id==4; 
surf_schaef2(CX(1:400),limits);
surf_cbm(CX(455:482),limits);
subcort_plot(CX,limits); colormap(custom());


%% C1 FIR Timeseries

% Load in coef_FIR.mat, coef_FIRwrong.mat, C1.mat, voltron_id2.mat

% Get C1 FIR timeseries
C1_FIR_diff = coef_FIR(:,C1==1,:);
C1_FIR_wrong = coef_FIRwrong(:,C1==1,:);
% Get network IDs: load in voltron_id2.mat
idx = voltron_id2(C1==1);
counts = hist(idx,[1:12]);


% FIGURE 3c-e --------------------------------
% Custom Network colormap
cmap = [repmat([0 63 92],counts(1),1); repmat([35 109 198],counts(2),1); 
    repmat([78 185 175],counts(3),1); repmat([0 169 120],counts(4),1); 
    repmat([50 205 50],counts(5),1); repmat([255 166 0],counts(6),1);
    repmat([255 127 14],counts(7),1); repmat([249 93 106],counts(8),1); 
    repmat([196 12 12],counts(9),1); repmat([213 58 122],counts(10),1); 
    repmat([160 81 149],counts(11),1); repmat([102 81 145],counts(12),1)]./255;

% Plot
% Easy
data = mean(C1_FIR_diff,3);
figure; plot(data(1:15,:),'LineWidth',1.5);
yline(0,'--','LineWidth',1);
set(gca,'box','off','FontSize',24, 'FontName', 'Arial','linew',1.5,'TickDir','out');
axis('tight');
ylim([-1 1]);
%xline([5 7 10],'--','LineWidth',1);
ax = gca;
ax.ColorOrder = cmap;

% Medium
figure; plot(data(16:30,:),'LineWidth',1.5);
yline(0,'--','LineWidth',1);
set(gca,'box','off','FontSize',24, 'FontName', 'Arial','linew',1.5,'TickDir','out');
axis('tight');
ylim([-1 1]);
%xline([5 7 10],'--','LineWidth',1);
ax = gca;
ax.ColorOrder = cmap;

% Hard
figure; plot(data(31:45,:),'LineWidth',1.5);
yline(0,'--','LineWidth',1);
set(gca,'box','off','FontSize',24, 'FontName', 'Arial','linew',1.5,'TickDir','out');
axis('tight');
ylim([-1 1]);
%xline([5 7 10],'--','LineWidth',1);
ax = gca;
ax.ColorOrder = cmap;

% Hard Wrong
data2 = mean(C1_FIR_wrong,3);
figure; plot(data2(1:15,:),'LineWidth',1.5);
yline(0,'--','LineWidth',1);
set(gca,'box','off','FontSize',24, 'FontName', 'Arial','linew',1.5,'TickDir','out');
axis('tight');
ylim([-1 1]);
%xline([5 7 10],'--','LineWidth',1);
ax = gca;
ax.ColorOrder = cmap;

% Average per network
for ii = 1:length(unique(idx))
    mean_C1diff(:,ii,:) = squeeze(mean(C1_FIR_diff(:,idx==ii,:),2));
    mean_C1wrong(:,ii,:) = squeeze(mean(C1_FIR_wrong(:,idx==ii,:),2));
end

% Plot
% Custom Network colormap
cmap2 = [0 63 92; 35 109 198; 78 185 175; 0 169 120; 50 205 50; 255 166 0; 
    255 127 14; 249 93 106; 160 81 149; 102 81 145]./255;
% Easy
data = mean(mean_C1diff,3);
figure; plot(data(1:15,:),'LineWidth',1.5);
yline(0,'--','LineWidth',1);
set(gca,'box','off','FontSize',24, 'FontName', 'Arial','linew',1.5);
axis('tight');
ylim([-0.8 0.8]);
%xline([5 7 10],'--','LineWidth',1);
ax = gca;
ax.ColorOrder = cmap2;

% Medium
figure; plot(data(16:30,:),'LineWidth',1.5);
yline(0,'--','LineWidth',1);
set(gca,'box','off','FontSize',24, 'FontName', 'Arial','linew',1.5);
axis('tight');
ylim([-0.8 0.8]);
%xline([5 7 10],'--','LineWidth',1);
ax = gca;
ax.ColorOrder = cmap2;

% Hard
figure; plot(data(31:45,:),'LineWidth',1.5);
yline(0,'--','LineWidth',1);
set(gca,'box','off','FontSize',24, 'FontName', 'Arial','linew',1.5);
axis('tight');
ylim([-0.8 0.8]);
%xline([5 7 10],'--','LineWidth',1);
ax = gca;
ax.ColorOrder = cmap2;

% Hard Wrong
data2 = mean(mean_C1wrong,3);
figure; plot(data2(1:15,:),'LineWidth',1.5);
yline(0,'--','LineWidth',1);
set(gca,'box','off','FontSize',24, 'FontName', 'Arial','linew',1.5);
axis('tight');
ylim([-0.8 0.8]);
%xline([5 7 10],'--','LineWidth',1);
ax = gca;
ax.ColorOrder = cmap2;

% Compare amplitude changes at each timepoint per region
% Different regions have different peaks
% Comparing relationship between amplitude and difficulty
C1_easy = C1_FIR_diff(1:15,:,:);
C1_med = C1_FIR_diff(16:30,:,:);
C1_hard = C1_FIR_diff(31:45,:,:);

% Generalised linear-mixed model
sub_id = repmat([1:24]',3,1);
difficulty = repelem([1:3]',24,1);
glme_time = [];
for jj = 1:size(C1_easy,1)
    glme_store = [];
    for ii = 1:size(C1_easy,2)
        message = 'Running GLME on timepoint %d roi %d\n';
        fprintf(message,jj,ii);
        roi = [squeeze(C1_easy(jj,ii,:)); squeeze(C1_med(jj,ii,:)); squeeze(C1_hard(jj,ii,:))];
        tbl = table(roi,difficulty,sub_id);
        glme = fitglme(tbl,'roi ~ 1 + difficulty + (1|sub_id)');
        glme_store(ii,1) = glme.Coefficients(2,2); % difficulty coef
        glme_store(ii,2) = glme.Coefficients(2,6); % p-value
    end
    glme_time = cat(3,glme_time,glme_store);
end
% Get coefficients
coef_time = squeeze(glme_time(:,1,:));
% Plot
data = coef_time;
figure; plot(data','LineWidth',1.5);
yline(0,'--','LineWidth',1);
set(gca,'box','off','FontSize',24, 'FontName', 'Arial','linew',1.5);
axis('tight');
ylim([-0.3 0.4]);
%xline([5 7 10],'--','LineWidth',1);
ax = gca;
ax.ColorOrder = cmap;

% Comparing relationship between amplitude and response
% Paired t-test
paired_time = [];
for jj = 1:size(C1_hard,1)
    paired_store = [];
    for ii = 1:size(C1_FIR_wrong,2)
        message = 'Running paired t-test on timepoint %d roi %d\n';
        fprintf(message,jj,ii);
        [h,p,~,stats] = ttest(squeeze(C1_hard(jj,ii,:)),squeeze(C1_FIR_wrong(jj,ii,:)));
        paired_store(ii,1) = h;
        paired_store(ii,2) = p;
        paired_store(ii,3) = stats.tstat;
    end
    paired_time = cat(3,paired_time,paired_store);
end
% Plot t-stats
tstat_time = squeeze(paired_time(:,3,:));
data = tstat_time;
figure; plot(data','LineWidth',1.5);
yline(0,'--','LineWidth',1);
set(gca,'box','off','FontSize',24, 'FontName', 'Arial','linew',1.5);
axis('tight');
%ylim([-9 6]);
%xline([5 7 10],'--','LineWidth',1);
ax = gca;
ax.ColorOrder = cmap;


%% Area under the curve

% Get C1 FIR timeseries
C1_FIR_diff = coef_FIR(:,C1==1,:);
C1_FIR_wrong = coef_FIRwrong(:,C1==1,:);

C1_easy = C1_FIR_diff(1:15,:,:);
C1_med = C1_FIR_diff(16:30,:,:);
C1_hard = C1_FIR_diff(31:45,:,:);

% Cumulative area under the curve per subject and roi
nROI = 102;
nTime = 15;
area_easy = [];
area_med = [];
area_hard = [];
area_wrong = [];
for ii = 1:size(C1_easy,3)

    sub_easy = C1_easy(:,:,ii);
    sub_med = C1_med(:,:,ii);
    sub_hard = C1_hard(:,:,ii);
    sub_wrong = C1_FIR_wrong(:,:,ii);
    temp_easy = zeros(nROI,nTime);
    temp_med = zeros(nROI,nTime);
    temp_hard = zeros(nROI,nTime);
    temp_wrong = zeros(nROI,nTime);
    for jj = 1:size(sub_easy,2)

        disp([ii jj])
        roi_easy = sub_easy(:,jj);
        roi_med = sub_med(:,jj);
        roi_hard = sub_hard(:,jj);
        roi_wrong = sub_wrong(:,jj);

        % Calculate area under the curve
        temp_easy(jj,:) = cumtrapz(roi_easy);
        temp_med(jj,:) = cumtrapz(roi_med);
        temp_hard(jj,:) = cumtrapz(roi_hard);
        temp_wrong(jj,:) = cumtrapz(roi_wrong);
    end
    
    % Store across subjects
    area_easy = cat(3,area_easy,temp_easy);
    area_med = cat(3,area_med,temp_med);
    area_hard = cat(3,area_hard,temp_hard);
    area_wrong = cat(3,area_wrong,temp_wrong);
end

% Take total area
area_easytot = squeeze(area_easy(:,15,:));
area_medtot = squeeze(area_med(:,15,:));
area_hardtot = squeeze(area_hard(:,15,:));
area_wrongtot = squeeze(area_wrong(:,15,:));

% Group average
mean_areaEasy = mean(area_easytot,2);
mean_areaMed = mean(area_medtot,2);
mean_areaHard = mean(area_hardtot,2);
mean_areaWrong = mean(area_wrongtot,2);


% FIGURE 3f ---------------------------------------
% Boxplot - Comparing "Effect" of Difficulty on BOLD
data1 = mean_areaEasy;
data2 = mean_areaMed;
data3 = mean_areaHard;
% Colormap
obs = length(data1);
RGB_color = [88 80 144; 188 80 144; 255 99 97]/255;
RGB_color2 = [];
RGB_color2(:,1) = repelem(RGB_color(:,1),obs);
RGB_color2(:,2) = repelem(RGB_color(:,2),obs);
RGB_color2(:,3) = repelem(RGB_color(:,3),obs);
% Group labels
N = length(data1);
group_id = [data1; data2; data3];
x =[repelem(1,N); repelem(2,N); repelem(3,N)];
x = x';
x = reshape(x,3*N,1);
% Scatter plot
figure;
boxplot([data1 data2 data3],'Labels',{'','',''},'Colors',RGB_color,'Symbol','','ColorGroup',x);
hold on
scatter(x,group_id,25,RGB_color2,'MarkerFaceColor','flat','MarkerFaceAlpha',0.7,'jitter','on','jitterAmount',0.2);
% line([x(1:N) x(N+1:2*N) x(2*N+1:end)]',[group_id(1:N) group_id(N+1:2*N) group_id(2*N+1:end)]','Color','black')
set(gca,'FontSize',24,'FontName','Arial','linew',1.5,'box','off','TickDir','out');
set(findobj(gca,'type','line'),'linew',3);

% Boxplot - Comparing "Effect" of Response on BOLD
data1 = mean_areaHard;
data2 = mean_areaWrong;
RGB_color = [53 155 124; 34 102 141]./255;
N = length(data1);
group = [data1; data2];
RGB_color2 = [];
RGB_color2(:,1) = repelem(RGB_color(:,1),N);
RGB_color2(:,2) = repelem(RGB_color(:,2),N);
RGB_color2(:,3) = repelem(RGB_color(:,3),N);
x =[repelem(1,N); repelem(2,N)];
x = x';
x = reshape(x,2*N,1);
figure; 
boxplot([data1 data2],'Labels',{'',''},'Colors',RGB_color,'Symbol','','OutlierSize',16);
hold on
% Scatter plot
scatter(x,group,20,RGB_color2,'filled','jitter','on','jitterAmount',0.1);
set(gca,'FontSize',24,'FontName','Arial','linew',1.5);
set(findobj(gca,'type','line'),'linew',2);

% Generalised linear-mixed model comparing difficulty and area
sub_id = repmat([1:24]',3,1);
difficulty = repelem([1:3]',24,1);
glme_area = [];
for ii = 1:size(area_easytot,1)
    message = 'Running GLME on roi %d\n';
    fprintf(message,ii);
    roi = [area_easytot(ii,:)'; area_medtot(ii,:)'; area_hardtot(ii,:)'];
    tbl = table(roi,difficulty,sub_id);
    glme = fitglme(tbl,'roi ~ 1 + difficulty + (1|sub_id)');
    glme_area(ii,1) = glme.Coefficients(2,2); % difficulty coef
    glme_area(ii,2) = glme.Coefficients(2,6); % p-value
end
% False Discovery Rate correction
[h,~,~,adj_p] = fdr_bh(glme_area(:,2));
sum(h) % 84
% Separate fdr corrected regions
area_sig = zeros(size(glme_area,1),1);
area_sig(h==1) = glme_area(h==1,1);
% Put significant regions back on the brain
area_map = zeros(502,1);
area_map(C1==1) = area_sig;
limits = [min(area_map) max(area_map)]; % [-0.7143 2.1637]
surf_schaef2(area_map(1:400),limits);
surf_cbm(area_map(455:482),limits);
subcort_plot(area_map); colormap(custom());


%% Cross-correlation

% Get C1 FIR timeseries
C1_FIR_diff = coef_FIR(:,C1==1,:);
C1_FIR_wrong = coef_FIRwrong(:,C1==1,:);

C1_easy = C1_FIR_diff(1:15,:,:);
C1_med = C1_FIR_diff(16:30,:,:);
C1_hard = C1_FIR_diff(31:45,:,:);

% Cross-correlation of corresponding regions between hard-correct and
% hard-incorrect, per subject
group_xcf = [];
for ii = 1:size(C1_hard,3)
    sub_hard = C1_hard(:,:,ii);
    sub_wrong = C1_FIR_wrong(:,:,ii);
    sub_xcf = [];
    for jj = 1:size(sub_hard,2)
        disp([ii jj])
        roi_hard = sub_hard(:,jj);
        roi_wrong = sub_wrong(:,jj);
        [xcf,lags] = crosscorr(roi_hard,roi_wrong);
        sub_xcf(jj,:) = xcf;
    end
    group_xcf = cat(3,group_xcf,sub_xcf);
end

% Average across subjects
mean_xcf = mean(group_xcf,3);
% Plot
% Get network IDs: load in voltron_id2.mat
idx = voltron_id2(C1==1);
counts = hist(idx,[1:12]);


% FIGURE 4c,d ---------------------------------
% Custom Network colormap
cmap = [repmat([0 63 92],counts(1),1); repmat([35 109 198],counts(2),1); 
    repmat([78 185 175],counts(3),1); repmat([0 169 120],counts(4),1); 
    repmat([50 205 50],counts(5),1); repmat([255 166 0],counts(6),1);
    repmat([255 127 14],counts(7),1); repmat([249 93 106],counts(8),1); 
    repmat([196 12 12],counts(9),1); repmat([213 58 122],counts(10),1); 
    repmat([160 81 149],counts(11),1); repmat([102 81 145],counts(12),1)]./255;
figure; 
plot(lags,mean_xcf,'LineWidth',1.5);
yline(0,'--','LineWidth',1);
set(gca,'box','off','FontSize',24, 'FontName', 'Arial','linew',1.5,'TickDir','out');
axis('tight');
ylim([-0.5 1]);
ax = gca;
ax.ColorOrder = cmap;

% Cross-correlation between medium and hard
group_xcf2 = [];
for ii = 1:size(C1_hard,3)
    sub_hard = C1_hard(:,:,ii);
    sub_med = C1_med(:,:,ii);
    sub_xcf = [];
    for jj = 1:size(sub_hard,2)
        disp([ii jj])
        roi_hard = sub_hard(:,jj);
        roi_med = sub_med(:,jj);
        [xcf,lags] = crosscorr(roi_med,roi_hard);
        sub_xcf(jj,:) = xcf;
    end
    group_xcf2 = cat(3,group_xcf2,sub_xcf);
end

% Average across subjects
mean_xcf2 = mean(group_xcf2,3);
% Plot
figure; 
plot(lags,mean_xcf2,'LineWidth',1.5);
yline(0,'--','LineWidth',1);
set(gca,'box','off','FontSize',24, 'FontName', 'Arial','linew',1.5,'TickDir','out');
axis('tight');
ylim([-0.5 1]);
ax = gca;
ax.ColorOrder = cmap;

% Find regions with delayed effect in wrong trials
max_xcf = max(mean_xcf,[],2);
max_time = mean_xcf == max_xcf; % A lot of regions delayed to t = 18
max_idx = max_time(:,18);

xcf_map = zeros(482,1);
xcf_map(C1==1) = max_idx;
limits = [0 1];
surf_schaef2(xcf_map(1:400),limits);
surf_cbm(xcf_map(455:482),limits);
subcort_plot(xcf_map,limits); colormap(custom());

% Plot FIR timeseries
% Hard
data = mean(C1_FIR_diff,3);
figure; plot(data(31:45,max_idx==1),'LineWidth',1.5);
yline(0,'--','LineWidth',1);
set(gca,'box','off','FontSize',24, 'FontName', 'Arial','linew',1.5);
axis('tight');
ylim([-1 1]);
%xline([5 7 10],'--','LineWidth',1);
ax = gca;
ax.ColorOrder = cmap;

% Hard Wrong
data2 = mean(C1_FIR_wrong,3);
figure; plot(data2(1:15,max_idx==1),'LineWidth',1.5);
yline(0,'--','LineWidth',1);
set(gca,'box','off','FontSize',24, 'FontName', 'Arial','linew',1.5);
axis('tight');
ylim([-1 1]);
%xline([5 7 10],'--','LineWidth',1);
ax = gca;
ax.ColorOrder = cmap;
