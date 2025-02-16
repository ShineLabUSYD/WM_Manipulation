%% FIR Analysis

% Load in mr_storage2.mat

% Design onset timing of trials (boxcar)
time_tot = 601; % Length of scan timeseries
trial_length = 2; % Extra length (not including the first frame)
for ii = 1:length(mr_storage2)
    trial_num = length(mr_storage2(ii).condition) + 1;
    ts_temp = zeros(time_tot,trial_num);
    onset = mr_storage2(ii).onset;
    onset2 = ceil(onset);
    onset_wrong = mr_storage2(ii).onset_wrong;
    start = mr_storage2(ii).timing(1);
    ts_temp(start:start+4,1) = 1;
    ts_temp2 = ts_temp;
    for jj = 1:size(onset2,1)
        ts_temp(onset2(jj):onset2(jj)+trial_length,jj+1) = 1;
    end

    if size(onset_wrong,1) ~= 0
        onset_wrong2 = ceil(onset_wrong);
        for jj = 1:size(onset_wrong2,1)
            ts_temp2(onset_wrong2(jj):onset_wrong2(jj)+trial_length,jj+1) = 1;
        end 
    else
        ts_temp2 = [];
    end 
    mr_storage2(ii).onset_time = ts_temp;
    mr_storage2(ii).onset_time2 = ts_temp2;
end


%% FIR Model: Task Difficulty

% Create HRF
P = [6 16 1 1 6 0 32]; % Specify parameters of response function
T = 16; % Specify microtime resolution
RT = 1; % Repitition time - Change this according to scanning parameter
[hrf,~] = spm_hrf(RT,P,T); % Create hrf
% Plot HRF
figure; plot(hrf);

% Create dummy 3 sec task block
test = zeros(30,1);
test(5:7) = 1;

% Convolve test with hrf
test_conv = conv(test,hrf);
figure; plot(test_conv); % Cross 0 to negative at TR = 15

% Plot both original and convolved together
figure;
plot(test,'--','LineWidth',1.5,'Color',[35/255 109/255 198/255]);
hold on
plot(test_conv(1:30),'LineWidth',3,'Color',[213/255 58/255 122/255]);
%plot(test_conv(1:30),'LineWidth',3,'Color',[35/255 109/255 198/255]);
set(gca,'box','off','FontSize',24, 'FontName', 'Arial','linew',1.5,'TickDir','out');
axis('tight');
%xline([5 8 11 14],'--','LineWidth',1,'Color','black');
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
coef_FIR = [];
for ii = 1:length(mr_collate)
    dm = mr_collate(ii).FIR_dm;
    ts = mr_collate(ii).ts;
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
    coef_FIR = cat(3,coef_FIR,glme_store); % Coefficients X ROI X subject
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
coef_FIR2 = [];
for ii = 1:length(mr_collate)
    dm = mr_collate(ii).FIR_dm2;
    ts = mr_collate(ii).ts;
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
    coef_FIR2 = cat(3,coef_FIR2,glme_store); % Coefficients X ROI X subject
end


% FIGURE 2a (right): Line plot of single-subject FIR timeseries
data = coef_FIR2(1:15,:,1);
figure;
plot(data,'LineWidth',0.6,'Color',[213/255 58/255 122/255 0.3]);
hold on
plot(mean(data,2),'LineWidth',5,'Color',[213/255 58/255 122/255 1]);
yline(0,'--','Color',[0 0 0],'LineWidth',1);
set(gca,'box','off','FontSize',24,'FontName','Arial','linew',1.5,'TickDir','out');
axis('tight');



% Create FIR design matrix for each difficulty - Incorrect Trials
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
coef_FIRwrong = [];
for ii = 1:length(mr_collate)
    dm = mr_collate(ii).FIR_dmwrong(:,31:end);
    ts = mr_collate(ii).ts;
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
    coef_FIRwrong = cat(3,coef_FIRwrong,glme_store); % Coefficients X ROI X subject
end

