%% Load in behavioural data

% Baseline + Rotation
rootdir = 'D:\PhD\mentalrotation_DiedrichsenLab\BIDS'; %set path of root directory
filelist = dir(fullfile(rootdir, '**\*events.tsv*')); %get list of .tsv files for all subfolders

% Create structure to hold the data
storage(length(filelist)) = struct('name',1,'onset',1,'duration',1, ...
    'trial_type',1,'taskName',1,'response_time',1, ...
    'condition',1,'numCorr',1,'numErr',1,'respMade',1,'startTRreal',1);
% Load in each subject and store their data
for ii = 1:length(filelist)
    subjectname = extractBefore(filelist(ii).name,'events.tsv'); 
    storage(ii).name = subjectname;
    fullFileName = fullfile(filelist(ii).folder, filelist(ii).name); %create absolute path to tsv file
    tdfread(fullFileName); %read in tsv file
    storage(ii).onset = onset;
    storage(ii).duration = duration;
    storage(ii).trial_type = trial_type;
    storage(ii).taskName = taskName;
    storage(ii).response_time = response_time;
    storage(ii).condition = condition;
    storage(ii).numCorr = numCorr;
    storage(ii).numErr = numErr;
    storage(ii).respMade = respMade;
    storage(ii).startTRreal = startTRreal;
end

% Change response time n/a to NaN, same as condition, numErr, numCorr
for ii = 1:length(storage)
    disp(ii)
    RT = storage(ii).response_time;
    cond = storage(ii).condition;
    Corr = storage(ii).numCorr;
    Err = storage(ii).numErr;
    respMade = storage(ii).respMade;
    TR = storage(ii).startTRreal;
    double_test = zeros(length(RT),6);
    for jj = 1:length(RT)
        num = zeros(1,6);
        num(1) = str2double(RT(jj,:));
        num(2) = str2double(cond(jj,:));
        num(3) = str2double(Corr(jj,:));
        num(4) = str2double(Err(jj,:));
        num(5) = str2double(respMade(jj,:));
        num(6) = str2double(TR(jj,:));
        double_test(jj,:) = num;
    end
    storage(ii).RT = double_test(:,1);
    storage(ii).condition2 = double_test(:,2);
    storage(ii).numCorr2 = double_test(:,3);
    storage(ii).numErr2 = double_test(:,4);
    storage(ii).respMade = double_test(:,5);
    storage(ii).Frames = double_test(:,6);
end

% Index Mental Rotation task
% Create structure to hold the data
mr_storage(length(filelist)) = struct('name',1,'onset',1,'duration',1, ...
    'trial_type',1,'taskName',1,'response_time',1, ...
    'condition',1,'numCorr',1,'numErr',1,'respMade',1,'Frames',1);

% Loop through and find mental rotation trial information
for ii = 1:length(storage)
    taskName = storage(ii).taskName;
    mr_storage(ii).name = storage(ii).name;
    % Find indices for mental rotation
    a = 2;
    mr_index = zeros(10,1);
    for jj = 1:length(taskName)
        if contains(taskName(jj,:),'mentalRotation')
            mr_index(a) = jj;
            a = a + 1;
        end
    end
    % Make first index = instruction
    mr_index(1) = mr_index(2) - 1;
    storage(ii).mr_index = mr_index;
end

% Get appropriate trial information
for ii = 1:length(storage)
    onset = storage(ii).onset;
    duration = storage(ii).duration;
    trial_type = storage(ii).trial_type;
    RT = storage(ii).RT;
    cond = storage(ii).condition2;
    numCorr = storage(ii).numCorr2;
    numErr = storage(ii).numErr2;
    respMade = storage(ii).respMade;
    Frames = storage(ii).Frames;
    taskName = storage(ii).taskName;
    index = storage(ii).mr_index;
    mr_storage(ii).onset = onset(index);
    mr_storage(ii).duration = duration(index);
    mr_storage(ii).trial_type = trial_type(index,:);
    mr_storage(ii).response_time = RT(index);
    mr_storage(ii).condition = cond(index);
    mr_storage(ii).numCorr = numCorr(index);
    mr_storage(ii).numErr = numErr(index);
    mr_storage(ii).respMade = respMade(index);
    mr_storage(ii).Frames = Frames(index);
    mr_storage(ii).taskName = taskName(index,:);
end

% Number of correct vs. incorrect trials
all_numCorr = horzcat(mr_storage.numCorr);
num_corr = sum(all_numCorr==1,'all'); % 2909
num_wrong = sum(all_numCorr==0,'all'); % 547

% Count wrong trials per difficulty
all_cond = horzcat(mr_storage.condition);
easy_wrong = length(all_numCorr(all_numCorr==0 & all_cond==1)); % 84
med_wrong = length(all_numCorr(all_numCorr==0 & all_cond==2)); % 160
hard_wrong = length(all_numCorr(all_numCorr==0 & all_cond==3)); % 303


%% Select Correct Responses only

% Get correct trial information only
mr_storage2(length(filelist)) = struct('name',1,'onset',1,'duration',1, ...
    'response_time',1,'condition',1);
for ii = 1:length(mr_storage)
    name = mr_storage(ii).name;
    onset = mr_storage(ii).onset;
    duration = mr_storage(ii).duration;
    condition = mr_storage(ii).condition;
    numCorr = mr_storage(ii).numCorr;
    RT = mr_storage(ii).response_time;
    onset2 = onset(numCorr==1);
    duration2 = duration(numCorr==1);
    condition2 = condition(numCorr==1);
    onset3 = onset(numCorr==0);
    condition3 = condition(numCorr==0);
    RT2 = RT(numCorr==1);
    mr_storage2(ii).onset = onset2;
    mr_storage2(ii).name = name;
    mr_storage2(ii).duration = duration2;
    mr_storage2(ii).condition = condition2;
    mr_storage2(ii).response_time = RT2;
    mr_storage2(ii).onset_orig = onset;
    mr_storage2(ii).onset_wrong = onset3;
    mr_storage2(ii).condition_wrong = condition3;
end

% Find start and end TRs for the timeseries
duration = 35; % 35 seconds including instruction
for ii = 1:length(mr_storage2)
    onset = ceil(mr_storage2(ii).onset_orig);
    start = onset(1);
    final = start + duration;
    if final <= 601
        final = final;
    elseif final > 601
        final = 601;
    end
    mr_storage2(ii).timing = [start final];
end

% Save mr_storage2 for later -------


%% Load in timeseries

% Set up folder pathway
myFolder = 'D:\PhD\mentalrotation_DiedrichsenLab\mr_timeseries';

% Get a list of all files in the folder with the desired file name pattern.
filePattern = fullfile(myFolder, '*timeseries_voltron400*.mat');
theFiles = dir(filePattern); % Storing all filenames

%storage(length(theFiles)) = struct('name',1,'ts',1);
% Loop to load in the data, store name and values of zscore
for ii = 1:length(theFiles)
    baseFileName = theFiles(ii).name; 
    fullFileName = fullfile(theFiles(ii).folder, baseFileName); %making absolute path to file
    ts = load(fullFileName); %loads in data
    storage(ii).ts = ts.ts; %stores data under .data
end


%% Get unique identifiers for each subject

scan_name = vertcat(mr_storage2.name);
% Only keep characters for subject ID
sub_id = {};
for ii = 1:size(scan_name,1)
    sub_id{ii,1} = extractBefore(scan_name(ii,:),'_ses');
end
% Check number of unique IDs
length(unique(sub_id)) % 24
sub_id2 = unique(sub_id);

% Save sub_id, sub_id2 for later ------


%% Behavioural data analysis (Mental Rotation)

% Load in mr_storage and sub_id

% Mental Rotation
% Format variables
all_RT = horzcat(mr_storage.response_time);
all_RT = all_RT(2:end,:);
all_cond = horzcat(mr_storage.condition);
all_cond = all_cond(2:end,:);
all_corr = horzcat(mr_storage.numCorr);
all_corr = all_corr(2:end,:);
all_name = vertcat(mr_storage.name);

all_sessions = zeros(size(all_name,1),1);
for ii = 1:length(all_name)
    if contains(all_name(ii,:),'ses-b1')
        all_sessions(ii) = 1;
    elseif contains(all_name(ii,:),'ses-b2')
        all_sessions(ii) = 2;
    end
end
all_runs = zeros(size(all_name,1),1);
for ii = 1:length(all_name)
    if contains(all_name(ii,:),'run-1')
        all_runs(ii) = 1;
    elseif contains(all_name(ii,:),'run-2')
        all_runs(ii) = 2;
    elseif contains(all_name(ii,:),'run-3')
        all_runs(ii) = 3;
    elseif contains(all_name(ii,:),'run-4')
        all_runs(ii) = 4;
    elseif contains(all_name(ii,:),'run-5')
        all_runs(ii) = 5;
    elseif contains(all_name(ii,:),'run-6')
        all_runs(ii) = 6;
    elseif contains(all_name(ii,:),'run-7')
        all_runs(ii) = 7;
    elseif contains(all_name(ii,:),'run-8')
        all_runs(ii) = 8;
    end
end
runs1 = all_runs(all_sessions==1);
runs2 = all_runs(all_sessions==2);

% Plot Average Response time per difficulty across scans and days
% Order response time by scans
[~,run_idx] = sort(all_runs,'ascend');
all_RT_ord = all_RT(:,run_idx);
all_cond_ord = all_cond(:,run_idx);
all_corr_ord = all_corr(:,run_idx);
all_sessions_ord = all_sessions(run_idx);

% Separate response by days
RT1 = all_RT_ord(:,all_sessions_ord==1);
RT2 = all_RT_ord(:,all_sessions_ord==2);
cond1 = all_cond_ord(:,all_sessions_ord==1);
cond2 = all_cond_ord(:,all_sessions_ord==2);
corr1 = all_corr_ord(:,all_sessions_ord==1);
corr2 = all_corr_ord(:,all_sessions_ord==2);

% Separate response by difficulty
RT1_easy = RT1(cond1==1);
RT2_easy = RT2(cond2==1);
RT1_med = RT1(cond1==2);
RT2_med = RT2(cond2==2);
RT1_hard = RT1(cond1==3);
RT2_hard = RT2(cond2==3);

corr1_easy = corr1(cond1==1);
corr1_med = corr1(cond1==2);
corr1_hard = corr1(cond1==3);
corr2_easy = corr2(cond2==1);
corr2_med = corr2(cond2==2);
corr2_hard = corr2(cond2==3);

num_trials = 3;
num_sub = 24;
RT1_easy_rs = reshape(RT1_easy,num_trials*num_sub,[]);
RT2_easy_rs = reshape(RT2_easy,num_trials*num_sub,[]);
RT1_med_rs = reshape(RT1_med,num_trials*num_sub,[]);
RT2_med_rs = reshape(RT2_med,num_trials*num_sub,[]);
RT1_hard_rs = reshape(RT1_hard,num_trials*num_sub,[]);
RT2_hard_rs = reshape(RT2_hard,num_trials*num_sub,[]);

corr1_easy_rs = reshape(corr1_easy,num_trials*num_sub,[]);
corr2_easy_rs = reshape(corr2_easy,num_trials*num_sub,[]);
corr1_med_rs = reshape(corr1_med,num_trials*num_sub,[]);
corr2_med_rs = reshape(corr2_med,num_trials*num_sub,[]);
corr1_hard_rs = reshape(corr1_hard,num_trials*num_sub,[]);
corr2_hard_rs = reshape(corr2_hard,num_trials*num_sub,[]);

% Change incorrect responses to NaN
RT1_easy_rs(corr1_easy_rs==0) = NaN;
RT2_easy_rs(corr2_easy_rs==0) = NaN;
RT1_med_rs(corr1_med_rs==0) = NaN;
RT2_med_rs(corr2_med_rs==0) = NaN;
RT1_hard_rs(corr1_hard_rs==0) = NaN;
RT2_hard_rs(corr2_hard_rs==0) = NaN;

% Calculate accuracy for each scan and difficulty
for ii = 1:length(mr_storage)
    cond = mr_storage(ii).condition(2:end);
    corr = mr_storage(ii).numCorr(2:end);
    resp = mr_storage(ii).respMade(2:end);
    easy_corr = corr(cond==1);
    med_corr = corr(cond==2);
    hard_corr = corr(cond==3);
    easy_acc = sum(easy_corr)/length(easy_corr);
    med_acc = sum(med_corr)/length(med_corr);
    hard_acc = sum(hard_corr)/length(hard_corr);
    mr_storage(ii).easy_acc = easy_acc;
    mr_storage(ii).med_acc = med_acc;
    mr_storage(ii).hard_acc = hard_acc;
end


% FIGURE 1c: Scatter plot of response time vs. accuracy for each difficulty
all_acc = reshape(mean_acc,[],1);
all_RT = [mean_RTeasy; mean_RTmed; mean_RThard];
data1 = all_RT;
data2 = all_acc;
obs = size(mean_acc,1);
RGB_color = [88 80 144; 188 80 144; 255 99 97]/255;
RGB_color2 = [];
RGB_color2(:,1) = repelem(RGB_color(:,1),obs);
RGB_color2(:,2) = repelem(RGB_color(:,2),obs);
RGB_color2(:,3) = repelem(RGB_color(:,3),obs);
cmap = RGB_color2;
figure; 
scatter(data1,data2,50,cmap,'MarkerFaceColor','flat','MarkerFaceAlpha',0.7);
%h1 = lsline();
%h1.Color = 'r';
%h1.LineWidth = 2;
set(gca,'box','off','FontSize',24,'FontName','Arial','linew',1.5,'TickDir','out');
axis('tight')



% GLMM for accuracy
% check relationship between accuracy and task difficulty
% acc ~ 1 + task_diff + scan_id + ses_id + (1|subject) + (1|ses_id:scan_id)
all_easyacc = vertcat(mr_storage.easy_acc);
all_medacc = vertcat(mr_storage.med_acc);
all_hardacc = vertcat(mr_storage.hard_acc);

acc = [all_easyacc; all_medacc; all_hardacc];
task_diff = [repelem(1,size(all_easyacc,1),1); repelem(2,size(all_medacc,1),1); repelem(3,size(all_hardacc,1),1);];
sub_id3 = repmat(sub_id',3,1);
ses_id = repmat(all_sessions,3,1);
scan_id = repmat(all_runs,3,1);
tbl = table(acc,task_diff,sub_id3,ses_id,scan_id);
glme = fitglme(tbl,'acc ~ 1 + task_diff + ses_id + (1|sub_id3) + (1|ses_id:scan_id)');
% task_diff: beta = -0.09158, p = 0, tstat = -12.892, df = 1149, CI =
% [-0.10552 -0.077642]
% ses_id: beta = 0.012153, p = 0.55202, tstat = 0.59491, df = 1149, CI =
% [-0.027928 0.052233]

% Plot accuracy for each difficulty across days
% Separate accuracy by days
runs1 = all_runs(all_sessions==1);
runs2 = all_runs(all_sessions==2);
all_easyacc1 = all_easyacc(all_sessions==1);
all_easyacc2 = all_easyacc(all_sessions==2);
all_medacc1 = all_medacc(all_sessions==1);
all_medacc2 = all_medacc(all_sessions==2);
all_hardacc1 = all_hardacc(all_sessions==1);
all_hardacc2 = all_hardacc(all_sessions==2);


% FIGURE 1d: Boxplot of participant accuracy across difficulty for each day
% -----
num_scans = 8;

all_easyacc1_rs = reshape(all_easyacc1,num_scans,[]);
mean_easyacc1 = mean(all_easyacc1_rs,1);
all_easyacc2_rs = reshape(all_easyacc2,num_scans,[]);
mean_easyacc2 = mean(all_easyacc2_rs,1);

all_medacc1_rs = reshape(all_medacc1,num_scans,[]);
mean_medacc1 = mean(all_medacc1_rs,1);
all_medacc2_rs = reshape(all_medacc2,num_scans,[]);
mean_medacc2 = mean(all_medacc2_rs,1);

all_hardacc1_rs = reshape(all_hardacc1,num_scans,[]);
mean_hardacc1 = mean(all_hardacc1_rs,1);
all_hardacc2_rs = reshape(all_hardacc2,num_scans,[]);
mean_hardacc2 = mean(all_hardacc2_rs,1);

data = [mean_easyacc1'; mean_easyacc2'; mean_medacc1'; mean_medacc2'; 
    mean_hardacc1'; mean_hardacc2'];
group = [1; 1; 3; 3; 5; 5];
group = repelem(group,24);
cgroup = [1; 2; 1; 2; 1; 2];
cgroup = repelem(cgroup,24);

figure;
b = boxchart(group,data,'GroupByColor',cgroup,'MarkerStyle','none');
b(1).BoxFaceColor = [71 147 175]./255;
b(2).BoxFaceColor = [255 196 112]./255;

hold on
RGB_color = [71 147 175; 255 196 112]./255;
RGB_color2 = [];
RGB_color2(:,1) = repelem(RGB_color(:,1),24);
RGB_color2(:,2) = repelem(RGB_color(:,2),24);
RGB_color2(:,3) = repelem(RGB_color(:,3),24);
RGB_color3 = repmat(RGB_color2,3,1);
group2 = [0.75; 1.25; 2.75; 3.25; 4.75; 5.25];
group2 = repelem(group2,24);
scatter(group2,data,20,RGB_color3,'MarkerFaceColor','flat', ...
    'MarkerFaceAlpha',0.7,'jitter','on','jitterAmount',0.1);

ylim([0 1]);
set(gca,'FontSize',24,'FontName','Arial','linew',1.5, ...
    'XTick',[1 3 5],'XTickLabel',[],'box','off','TickDir','out');



% GLMM for Response Time
% check relationship between response time and task difficulty
% RT ~ 1 + task_diff + ses_id + (1|subject)
% + (1|scan_id:trial_id)
RT_rs = reshape(all_RT,[],1);
corr_rs = reshape(all_corr,[],1);
RT_rs2 = RT_rs(corr_rs==1);
scan_id = repelem(all_runs,9,1);
scan_id2 = scan_id(corr_rs==1);
ses_id = repelem(all_sessions,9,1);
ses_id2 = ses_id(corr_rs==1);
num_trials = [1:9]';
num_scans = 384;
trial_id = repmat(num_trials,num_scans,1);
trial_id2 = trial_id(corr_rs==1);
task_diff = reshape(all_cond,[],1);
task_diff2 = task_diff(corr_rs==1);
subject_id = repelem(sub_id',9,1);
subject_id2 = subject_id(corr_rs==1);
tbl = table(RT_rs2,task_diff2,scan_id2,ses_id2,trial_id2,subject_id2);
glme = fitglme(tbl,'RT_rs2 ~ 1 + task_diff2 + ses_id2 + (1|subject_id2) + (1|ses_id2:scan_id2) + (1|scan_id2:trial_id2)');
% task_diff2: beta = 0.26334, tstat = 29.267, df = 2906, p = 0, CI = [0.24569 0.28098];
% ses_id2: beta = -0.031781, tstat = -1.5533, p = 0.12047, CI = [-0.0719 0.0083383];


% FIGURE 1e: Boxplot of participant response time across difficulty for
% each day -----------

num_trials = 3;
num_subj = 24;

RT1_easy_sub = reshape(RT1_easy_rs,num_trials,num_subj,8);
mean_RT1_easy = mean(RT1_easy_sub,[1 3],'omitnan');
RT2_easy_sub = reshape(RT2_easy_rs,num_trials,num_subj,8);
mean_RT2_easy = mean(RT2_easy_sub,[1 3],'omitnan');

RT1_med_sub = reshape(RT1_med_rs,num_trials,num_subj,8);
mean_RT1_med = mean(RT1_med_sub,[1 3],'omitnan');
RT2_med_sub = reshape(RT2_med_rs,num_trials,num_subj,8);
mean_RT2_med = mean(RT2_med_sub,[1 3],'omitnan');

RT1_hard_sub = reshape(RT1_hard_rs,num_trials,num_subj,8);
mean_RT1_hard = mean(RT1_hard_sub,[1 3],'omitnan');
RT2_hard_sub = reshape(RT2_hard_rs,num_trials,num_subj,8);
mean_RT2_hard = mean(RT2_hard_sub,[1 3],'omitnan');

data = [mean_RT1_easy'; mean_RT2_easy'; mean_RT1_med'; mean_RT2_med'; 
    mean_RT1_hard'; mean_RT2_hard'];
group = [1; 1; 3; 3; 5; 5];
group = repelem(group,24);
cgroup = [1; 2; 1; 2; 1; 2];
cgroup = repelem(cgroup,24);

figure;
b = boxchart(group,data,'GroupByColor',cgroup,'MarkerStyle','none');
b(1).BoxFaceColor = [71 147 175]./255;
b(2).BoxFaceColor = [255 196 112]./255;

hold on
RGB_color = [71 147 175; 255 196 112]./255;
RGB_color2 = [];
RGB_color2(:,1) = repelem(RGB_color(:,1),24);
RGB_color2(:,2) = repelem(RGB_color(:,2),24);
RGB_color2(:,3) = repelem(RGB_color(:,3),24);
RGB_color3 = repmat(RGB_color2,3,1);
group2 = [0.75; 1.25; 2.75; 3.25; 4.75; 5.25];
group2 = repelem(group2,24);
scatter(group2,data,20,RGB_color3,'MarkerFaceColor','flat', ...
    'MarkerFaceAlpha',0.7,'jitter','on','jitterAmount',0.1);

ylim([0 2.5]);
set(gca,'FontSize',24,'FontName','Arial','linew',1.5, ...
    'XTick',[1 3 5],'XTickLabel',[],'box','off','TickDir','out');

