%% FIR model - comparing across difficulties

% Load in mr_storage2.mat

% Create HRF
P = [6 16 1 1 6 0 32]; % Specify parameters of response function
T = 16; % Specify microtime resolution
RT = 1; % Repitition time - Change this according to scanning parameter
[hrf,~] = spm_hrf(RT,P,T); % Create hrf
% Plot HRF
figure; plot(hrf);

% Create dummy 3 sec task block
test = zeros(30,1);
test(1:3) = 1;

% Convolve test with hrf
test_conv = conv(test,hrf);
figure; plot(test_conv); % Cross 0 to negative at TR = 15

% Plot both original and convolved together
figure;
plot(test,'--','LineWidth',1.5,'Color',[35/255 109/255 198/255]);
hold on
plot(test_conv(1:30),'LineWidth',3,'Color',[213/255 58/255 122/255]);
set(gca,'box','off','FontSize',24, 'FontName', 'Arial','linew',1.5);
axis('tight');
ylim([-0.2 1.2]);

% Group together onset times by difficulty
% Find first instance (start) of each trial
for ii = 1:length(mr_storage2)
    disp(ii)
    onset_time = mr_storage2(ii).onset_time(:,2:end);
    cond = mr_storage2(ii).condition;
    
    easy = onset_time(:,cond==1);
    easy_start = [];
    for jj = 1:size(easy,2)
        easy_start(jj) = find(easy(:,jj)==1,1);
    end

    med = onset_time(:,cond==2);
    med_start = [];
    for jj = 1:size(med,2)
        med_start(jj) = find(med(:,jj)==1,1);
    end

    hard = onset_time(:,cond==3);
    hard_start = [];
    for jj = 1:size(hard,2)
        hard_start(jj) = find(hard(:,jj)==1,1);
    end

    mr_storage2(ii).easy_start = easy_start;
    mr_storage2(ii).med_start = med_start;
    mr_storage2(ii).hard_start = hard_start;
end

% Create FIR design matrix for each scan
for ii = 1:length(mr_storage2)
    disp(ii)
    easy_start = mr_storage2(ii).easy_start;
    med_start = mr_storage2(ii).med_start;
    hard_start = mr_storage2(ii).hard_start;

    easy_dm = zeros(601,15);
    for jj = 1:length(easy_start)
        for kk = 1:size(easy_dm,2)
            easy_dm(easy_start(jj)+kk-1,kk) = 1;
        end
    end

    med_dm = zeros(601,15);
    for jj = 1:length(med_start)
        for kk = 1:size(med_dm,2)
            med_dm(med_start(jj)+kk-1,kk) = 1;
        end
    end

    hard_dm = zeros(601,15);
    for jj = 1:length(hard_start)
        for kk = 1:size(hard_dm,2)
            hard_dm(hard_start(jj)+kk-1,kk) = 1;
        end
    end

    easy_dm2 = easy_dm(1:601,:);
    med_dm2 = med_dm(1:601,:);
    hard_dm2 = hard_dm(1:601,:);

    FIR_dm = [easy_dm2 med_dm2 hard_dm2];

    mr_storage2(ii).FIR_dm = FIR_dm;
end

% Collate scans across subjects
% Load in sub_id2, mr_collate
for ii = 1:length(sub_id2)
    sub_dm = [];
    for jj = 1:length(mr_storage2)
        name = mr_storage2(jj).name;
        if contains(name,sub_id2(ii))
            FIR_dm = mr_storage2(jj).FIR_dm;
            sub_dm = [sub_dm; FIR_dm];
        end
    end
    mr_collate(ii).FIR_dm = sub_dm;
end

% Fit Generalised Linear Mixed Model to FIR design matrix
% Load in mr_collate1000.mat, mr_collate.mat
coef_FIR1000 = [];
for ii = 1:length(mr_collate)
    dm = mr_collate(ii).FIR_dm;
    ts = mr_collate1000(ii).ts;
    scan_id = mr_collate(ii).scan_id;
    ses_id = mr_collate(ii).ses_id;
    glme_store = [];
    for jj = 1:size(ts,2)
        disp([ii jj])
        tbl = array2table([ts(:,jj) dm ses_id scan_id]);
        glme = fitglme(tbl,['Var1 ~ 1 + Var2 + Var3 + Var4 + Var5 + Var6 + ' ...
            'Var7 + Var8 + Var9 + Var10 + Var11 + Var12 + Var13 + Var14 + ' ...
            'Var15 + Var16 + Var17 + Var18 + Var19 + Var20 + Var21 + ' ...
            'Var22 + Var23 + Var24 + Var25 + Var26 + Var27 + Var28 + ' ...
            'Var29 + Var30 + Var31 + Var32 + Var33 + Var34 + Var35 + ' ...
            'Var36 + Var37 + Var38 + Var39 + Var40 + Var41 + Var42 + ' ...
            'Var43 + Var44 + Var45 + Var46 + Var47 + (1|Var48)']);
        coef = double(glme.Coefficients(2:end-1,2));
        glme_store = [glme_store coef]; % Coefficients X ROI
    end
    coef_FIR1000 = cat(3,coef_FIR1000,glme_store); % Coefficients X ROI X subject
end

% 2nd-level Generalised Linear Mixed Model
% Finding relationship across task difficulty per region
glm_FIR1000 = [];
sub_id = repmat([1:24]',3,1);
difficulty = repelem([1:3]',24,1);
n_time = 15;
for ii = 1:n_time
    glme_store = [];
    roi_easy = squeeze(coef_FIR1000(ii,:,:)); % ROI X subjects
    roi_med = squeeze(coef_FIR1000(ii+15,:,:));
    roi_hard = squeeze(coef_FIR1000(ii+30,:,:));
    for jj = 1:size(roi_easy,1)
        message = 'Running GLME on timepoint %d roi %d\n';
        fprintf(message,ii,jj);
        roi = [roi_easy(jj,:) roi_med(jj,:) roi_hard(jj,:)]';
        tbl = table(roi,difficulty,sub_id);
        glme = fitglme(tbl,'roi ~ 1 + difficulty + (1|sub_id)');
        glme_store(jj,1) = glme.Coefficients(2,2); % difficulty coef
        glme_store(jj,2) = glme.Coefficients(2,6); % p-value
    end
    glm_FIR1000 = cat(3,glm_FIR1000,glme_store); % ROI X output X timepoint
end


%% FIR Model - comparing correct vs. incorrect

% Load in mr_storage2.mat

% Create HRF
P = [6 16 1 1 6 0 32]; % Specify parameters of response function
T = 16; % Specify microtime resolution
RT = 1; % Repitition time - Change this according to scanning parameter
[hrf,~] = spm_hrf(RT,P,T); % Create hrf

% Create FIR Design matrix
for ii = 1:length(mr_storage2)
    disp(ii)
    onset_incorrect = ceil(mr_storage2(ii).onset_wrong);
    onset_correct = ceil(mr_storage2(ii).onset);
    
    correct_dm = zeros(601,15);
    for jj = 1:length(onset_correct)
        for kk = 1:size(correct_dm,2)
            correct_dm(onset_correct(jj)+kk-1,kk) = 1;
        end
    end

    incorrect_dm = zeros(601,15);
    for jj = 1:length(onset_incorrect)
        for kk = 1:size(incorrect_dm,2)
            incorrect_dm(onset_incorrect(jj)+kk-1,kk) = 1;
        end
    end

    correct_dm2 = correct_dm(1:601,:);
    incorrect_dm2 = incorrect_dm(1:601,:);

    FIR_dm2 = [correct_dm2 incorrect_dm2];

    mr_storage2(ii).FIR_dm2 = FIR_dm2;
end

% Collate scans across subjects
% Load in sub_id2, mr_collate
for ii = 1:length(sub_id2)
    sub_dm = [];
    for jj = 1:length(mr_storage2)
        name = mr_storage2(jj).name;
        if contains(name,sub_id2(ii))
            FIR_dm = mr_storage2(jj).FIR_dm2;
            sub_dm = [sub_dm; FIR_dm];
        end
    end
    mr_collate(ii).FIR_dm2 = sub_dm;
end

% Fit Generalised Linear Mixed Model to FIR design matrix
coef_FIR2_1000 = [];
for ii = 1:length(mr_collate)
    dm = mr_collate(ii).FIR_dm2;
    ts = mr_collate1000(ii).ts;
    scan_id = mr_collate(ii).scan_id;
    ses_id = mr_collate(ii).ses_id;
    glme_store = [];
    for jj = 1:size(ts,2)
        disp([ii jj])
        tbl = array2table([ts(:,jj) dm ses_id scan_id]);
        glme = fitglme(tbl,['Var1 ~ 1 + Var2 + Var3 + Var4 + Var5 + Var6 + ' ...
            'Var7 + Var8 + Var9 + Var10 + Var11 + Var12 + Var13 + Var14 + ' ...
            'Var15 + Var16 + Var17 + Var18 + Var19 + Var20 + Var21 + ' ...
            'Var22 + Var23 + Var24 + Var25 + Var26 + Var27 + Var28 + ' ...
            'Var29 + Var30 + Var31 + Var32 + (1|Var33)']);
        coef = double(glme.Coefficients(2:end-1,2));
        glme_store = [glme_store coef]; % Coefficients X ROI
    end
    coef_FIR2_1000 = cat(3,coef_FIR2_1000,glme_store); % Coefficients X ROI X subject
end

% 2nd-level paired t-test
glm_FIR2_1000 = [];
n_time = 15;
for ii = 1:n_time
    glme_store = [];
    roi_correct = squeeze(coef_FIR2_1000(ii,:,:)); % ROI X subjects
    roi_incorrect = squeeze(coef_FIR2_1000(ii+15,:,:));
    for jj = 1:size(roi_correct,1)
        message = 'Running paired t-test on timepoint %d roi %d\n';
        fprintf(message,ii,jj);
        roi = [roi_correct(jj,:); roi_incorrect(jj,:)];
        [h,p] = ttest(roi(1,:)',roi(2,:)');
        glme_store(jj,1) = h;
        glme_store(jj,2) = p;
    end
    glm_FIR2_1000 = cat(3,glm_FIR2_1000,glme_store); % ROI X output X timepoint
end

mean_FIR2 = mean(coef_FIR2_1000,3)';
corr_FIR_1000 = mean_FIR2(:,1:15) - mean_FIR2(:,16:end);

% Load in glm_FIR (difficulty)
diff_FIR_1000 = squeeze(glm_FIR1000(:,1,:));


%% FIR hard-correct vs. hard-incorrect

% Create FIR design matrix for each difficulty
% Group together onset times by difficulty
% Find first instance (start) of each trial
for ii = 1:length(mr_storage2)
    disp(ii)
    onset_wrong = mr_storage2(ii).onset_time2(:,2:end);
    cond = mr_storage2(ii).condition_wrong;
    
    easy = onset_wrong(:,cond==1);
    easy_start = [];
    for jj = 1:size(easy,2)
        easy_start(jj) = find(easy(:,jj)==1,1);
    end

    med = onset_wrong(:,cond==2);
    med_start = [];
    for jj = 1:size(med,2)
        med_start(jj) = find(med(:,jj)==1,1);
    end

    hard = onset_wrong(:,cond==3);
    hard_start = [];
    for jj = 1:size(hard,2)
        hard_start(jj) = find(hard(:,jj)==1,1);
    end

    mr_storage2(ii).easy_start2 = easy_start;
    mr_storage2(ii).med_start2 = med_start;
    mr_storage2(ii).hard_start2 = hard_start;
end

% Create FIR design matrix for each scan
for ii = 1:length(mr_storage2)
    disp(ii)
    easy_start = mr_storage2(ii).easy_start2;
    med_start = mr_storage2(ii).med_start2;
    hard_start = mr_storage2(ii).hard_start2;

    easy_dm = zeros(601,15);
    for jj = 1:length(easy_start)
        for kk = 1:size(easy_dm,2)
            easy_dm(easy_start(jj)+kk-1,kk) = 1;
        end
    end

    med_dm = zeros(601,15);
    for jj = 1:length(med_start)
        for kk = 1:size(med_dm,2)
            med_dm(med_start(jj)+kk-1,kk) = 1;
        end
    end

    hard_dm = zeros(601,15);
    for jj = 1:length(hard_start)
        for kk = 1:size(hard_dm,2)
            hard_dm(hard_start(jj)+kk-1,kk) = 1;
        end
    end

    easy_dm2 = easy_dm(1:601,:);
    med_dm2 = med_dm(1:601,:);
    hard_dm2 = hard_dm(1:601,:);

    FIR_dm = [easy_dm2 med_dm2 hard_dm2];

    mr_storage2(ii).FIR_dmwrong = FIR_dm;
end

% Collate scans across subjects
% Load in sub_id2, mr_collate
for ii = 1:length(sub_id2)
    sub_dm = [];
    for jj = 1:length(mr_storage2)
        name = mr_storage2(jj).name;
        if contains(name,sub_id2(ii))
            FIR_dm = mr_storage2(jj).FIR_dmwrong;
            sub_dm = [sub_dm; FIR_dm];
        end
    end
    mr_collate(ii).FIR_dmwrong = sub_dm;
end

% Fit Generalised Linear Mixed Model to FIR design matrix
coef_FIRwrong1000 = [];
for ii = 1:length(mr_collate)
    dm = mr_collate(ii).FIR_dmwrong(:,31:end);
    ts = mr_collate1000(ii).ts;
    scan_id = mr_collate(ii).scan_id;
    ses_id = mr_collate(ii).ses_id;
    glme_store = [];
    for jj = 1:size(ts,2)
        disp([ii jj])
        tbl = array2table([ts(:,jj) dm ses_id scan_id]);
        glme = fitglme(tbl,['Var1 ~ 1 + Var2 + Var3 + Var4 + Var5 + Var6 + ' ...
            'Var7 + Var8 + Var9 + Var10 + Var11 + Var12 + Var13 + Var14 + ' ...
            'Var15 + Var16 + Var17 + (1|Var18)']);
        coef = double(glme.Coefficients(2:end-1,2));
        glme_store = [glme_store coef]; % Coefficients X ROI
    end
    coef_FIRwrong1000 = cat(3,coef_FIRwrong1000,glme_store); % Coefficients X ROI X subject
end



%% Exploration of FIR

% Check shape of FIR
diff_FIR2 = abs(diff_FIR_1000);
mean_diff = mean(diff_FIR2,1);

corr_FIR2 = abs(corr_FIR_1000);
mean_corr = mean(corr_FIR2,1);

figure; plot(mean_diff);
figure; plot(mean_corr);
% Peaks at TR = 7

% Plot mean and standard deviation of each FIR model
% Difficulty
data = diff_FIR2;
figure; shadedErrorBar(1:size(data,2),mean(data),std(data,[],1)/sqrt(size(data,1)));
xlabel(''); ylabel(''); 
xline(7,'--b','LineWidth',2);
set(gca,'box','off','FontSize',24, 'FontName', 'Arial','linew',1.5);
xlim('tight');

% Response
data = corr_FIR2;
figure; shadedErrorBar(1:size(data,2),mean(data),std(data,[],1)/sqrt(size(data,1)));
xlabel(''); ylabel(''); 
xline(7,'--b','LineWidth',2);
set(gca,'box','off','FontSize',24, 'FontName', 'Arial','linew',1.5);
xlim('tight');

% Take slice of FIR at TR = 7
corr_FIRt = corr_FIR_1000(:,7);
diff_FIRt = diff_FIR_1000(:,7);

% Plot slice on brain surface
% Difficulty
limits = [min(diff_FIRt) max(diff_FIRt)]; % [-0.1496 0.3553]
surf_schaef2(diff_FIRt(1:400),limits);
surf_cbm(diff_FIRt(455:482),limits);
subcort_plot(diff_FIRt); colormap(custom());

% Response
limits = [min(corr_FIRt) max(corr_FIRt)]; % [-0.7184 0.3583]
surf_schaef2(corr_FIRt(1:400),limits);
surf_cbm(corr_FIRt(455:482),limits);
subcort_plot(corr_FIRt); colormap(custom());

% Threshold Task difficulty map
% FDR correction for TR = 7
[h,~,~,adj_p] = fdr_bh(glm_FIR1000(:,2,7));
temp_diff1000 = zeros(1102,1);
temp_diff1000(h==1) = glm_FIR1000(h==1,1,7);

% Plot Thresholded map on brain
limits = [min(temp_diff1000) max(temp_diff1000)]; % [-0.1646 0.3907]
surf_schaef1000(temp_diff1000(1:1000),limits);
surf_cbm(temp_diff1000(1055:1082),limits);
subcort_plot1000(temp_diff1000); colormap(custom());

% Threshold correct vs. incorrect map
% FDR correction for TR = 7
[h,~,~,adj_p] = fdr_bh(glm_FIR2_1000(:,2,7));
temp_corr1000 = zeros(1102,1);
temp_corr1000(h==1) = corr_FIR_1000(h==1,7);

% Plot Thresholded map on brain
limits = [min(temp_corr1000) max(temp_corr1000)]; % [-0.6567 0.3475]
surf_schaef1000(temp_corr1000(1:1000),limits);
surf_cbm(temp_corr1000(1055:1082),limits);
subcort_plot1000(temp_corr1000); colormap(custom());
