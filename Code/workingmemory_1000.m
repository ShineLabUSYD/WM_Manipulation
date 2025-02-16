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

% Do the same for spatial map working memory task
% Create structure to hold the data
sm_storage(length(filelist)) = struct('name',1,'onset',1,'duration',1, ...
    'trial_type',1,'taskName',1,'response_time',1, ...
    'condition',1,'numCorr',1,'numErr',1,'respMade',1,'Frames',1);

% Loop through and find spatial map trial information
for ii = 1:length(storage)
    taskName = storage(ii).taskName;
    sm_storage(ii).name = storage(ii).name;
    % Find indices for mental rotation
    a = 2;
    sm_index = zeros(7,1);
    for jj = 1:length(taskName)
        if contains(taskName(jj,:),'spatialMap')
            sm_index(a) = jj;
            a = a + 1;
        end
    end
    % Make first index = instruction
    sm_index(1) = sm_index(2) - 1;
    storage(ii).sm_index = sm_index;
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
    index = storage(ii).sm_index;
    sm_storage(ii).onset = onset(index);
    sm_storage(ii).duration = duration(index);
    sm_storage(ii).trial_type = trial_type(index,:);
    sm_storage(ii).response_time = RT(index);
    sm_storage(ii).condition = cond(index);
    sm_storage(ii).numCorr = numCorr(index);
    sm_storage(ii).numErr = numErr(index);
    sm_storage(ii).respMade = respMade(index);
    sm_storage(ii).Frames = Frames(index);
    sm_storage(ii).taskName = taskName(index,:);
end

% Index Response Alternative task (finger sequencing)
% Create structure to hold the data
ra_storage(length(filelist)) = struct('name',1,'onset',1,'duration',1, ...
    'trial_type',1,'taskName',1,'response_time',1, ...
    'condition',1,'numCorr',1,'numErr',1,'respMade',1,'Frames',1);

% Loop through and find mental rotation trial information
for ii = 1:length(storage)
    taskName = storage(ii).taskName;
    ra_storage(ii).name = storage(ii).name;
    % Find indices for response alternative
    a = 2;
    ra_index = zeros(7,1);
    for jj = 1:length(taskName)
        if contains(taskName(jj,:),'respAlt')
            ra_index(a) = jj;
            a = a + 1;
        end
    end
    % Make first index = instruction
    ra_index(1) = ra_index(2) - 1;
    storage(ii).ra_index = ra_index;
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
    index = storage(ii).ra_index;
    ra_storage(ii).onset = onset(index);
    ra_storage(ii).duration = duration(index);
    ra_storage(ii).trial_type = trial_type(index,:);
    ra_storage(ii).response_time = RT(index);
    ra_storage(ii).condition = cond(index);
    ra_storage(ii).numCorr = numCorr(index);
    ra_storage(ii).numErr = numErr(index);
    ra_storage(ii).respMade = respMade(index);
    ra_storage(ii).Frames = Frames(index);
    ra_storage(ii).taskName = taskName(index,:);
end

% Index Object n-back task
% Create structure to hold the data
nbackPic_storage(length(filelist)) = struct('name',1,'onset',1,'duration',1, ...
    'trial_type',1,'taskName',1,'response_time',1, ...
    'condition',1,'numCorr',1,'numErr',1,'respMade',1,'Frames',1);

% Loop through and find mental rotation trial information
for ii = 1:length(storage)
    taskName = storage(ii).taskName;
    nbackPic_storage(ii).name = storage(ii).name;
    % Find indices for object n-back task
    a = 2;
    nbackPic_index = zeros(16,1);
    for jj = 1:length(taskName)
        if contains(taskName(jj,:),'nBackPic2')
            nbackPic_index(a) = jj;
            a = a + 1;
        end
    end
    % Make first index = instruction
    nbackPic_index(1) = nbackPic_index(2) - 1;
    storage(ii).nbackPic_index = nbackPic_index;
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
    index = storage(ii).nbackPic_index;
    nbackPic_storage(ii).onset = onset(index);
    nbackPic_storage(ii).duration = duration(index);
    nbackPic_storage(ii).trial_type = trial_type(index,:);
    nbackPic_storage(ii).response_time = RT(index);
    nbackPic_storage(ii).condition = cond(index);
    nbackPic_storage(ii).numCorr = numCorr(index);
    nbackPic_storage(ii).numErr = numErr(index);
    nbackPic_storage(ii).respMade = respMade(index);
    nbackPic_storage(ii).Frames = Frames(index);
    nbackPic_storage(ii).taskName = taskName(index,:);
end


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

sm_storage2(length(filelist)) = struct('name',1,'onset',1,'duration',1, ...
    'response_time',1,'condition',1);
for ii = 1:length(sm_storage)
    name = sm_storage(ii).name;
    onset = sm_storage(ii).onset;
    duration = sm_storage(ii).duration;
    condition = sm_storage(ii).condition;
    numCorr = sm_storage(ii).numCorr;
    RT = sm_storage(ii).response_time;
    onset2 = onset(numCorr==1);
    duration2 = duration(numCorr==1);
    condition2 = condition(numCorr==1);
    RT2 = RT(numCorr==1);
    sm_storage2(ii).onset = onset2;
    sm_storage2(ii).name = name;
    sm_storage2(ii).duration = duration2;
    sm_storage2(ii).condition = condition2;
    sm_storage2(ii).response_time = RT2;
    sm_storage2(ii).onset_orig = onset;
end

ra_storage2(length(filelist)) = struct('name',1,'onset',1,'duration',1, ...
    'response_time',1,'condition',1);
for ii = 1:length(ra_storage)
    name = ra_storage(ii).name;
    onset = ra_storage(ii).onset;
    duration = ra_storage(ii).duration;
    condition = ra_storage(ii).condition;
    numCorr = ra_storage(ii).numCorr;
    RT = ra_storage(ii).response_time;
    onset2 = onset(numCorr==1);
    duration2 = duration(numCorr==1);
    condition2 = condition(numCorr==1);
    RT2 = RT(numCorr==1);
    ra_storage2(ii).onset = onset2;
    ra_storage2(ii).name = name;
    ra_storage2(ii).duration = duration2;
    ra_storage2(ii).condition = condition2;
    ra_storage2(ii).response_time = RT2;
    ra_storage2(ii).onset_orig = onset;
end

nbackPic_storage2(length(filelist)) = struct('name',1,'onset',1,'duration',1, ...
    'response_time',1,'condition',1);
for ii = 1:length(nbackPic_storage)
    name = nbackPic_storage(ii).name;
    onset = nbackPic_storage(ii).onset;
    duration = nbackPic_storage(ii).duration;
    condition = nbackPic_storage(ii).condition;
    numCorr = nbackPic_storage(ii).numCorr;
    RT = nbackPic_storage(ii).response_time;
    onset2 = onset(numCorr==1);
    duration2 = duration(numCorr==1);
    condition2 = condition(numCorr==1);
    RT2 = RT(numCorr==1);
    nbackPic_storage2(ii).onset = onset2;
    nbackPic_storage2(ii).name = name;
    nbackPic_storage2(ii).duration = duration2;
    nbackPic_storage2(ii).condition = condition2;
    nbackPic_storage2(ii).response_time = RT2;
    nbackPic_storage2(ii).onset_orig = onset;
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

duration = 35; % 35 seconds including instruction
for ii = 1:length(sm_storage2)
    onset = ceil(sm_storage2(ii).onset_orig);
    start = onset(1);
    final = start + duration;
    if final <= 601
        final = final;
    elseif final > 601
        final = 601;
    end
    sm_storage2(ii).timing = [start final];
end

duration = 35; % 35 seconds including instruction
for ii = 1:length(ra_storage2)
    onset = ceil(ra_storage2(ii).onset_orig);
    start = onset(1);
    final = start + duration;
    if final <= 601
        final = final;
    elseif final > 601
        final = 601;
    end
    ra_storage2(ii).timing = [start final];
end

duration = 35; % 35 seconds including instruction
for ii = 1:length(nbackPic_storage2)
    onset = ceil(nbackPic_storage2(ii).onset_orig);
    start = onset(1);
    final = start + duration;
    if final <= 601
        final = final;
    elseif final > 601
        final = 601;
    end
    nbackPic_storage2(ii).timing = [start final];
end


%% Load in timeseries

% Set up folder pathway
myFolder = 'D:\PhD\mentalrotation_DiedrichsenLab\mr_timeseries1000';

% Get a list of all files in the folder with the desired file name pattern.
filePattern = fullfile(myFolder, '*timeseries_voltron1000*.mat');
theFiles = dir(filePattern); % Storing all filenames

%storage(length(theFiles)) = struct('name',1,'ts',1);
% Loop to load in the data, store name and values of zscore
for ii = 1:length(theFiles)
    baseFileName = theFiles(ii).name; 
    fullFileName = fullfile(theFiles(ii).folder, baseFileName); %making absolute path to file
    ts = load(fullFileName); %loads in data
    storage(ii).ts = ts.ts; %stores data under .data
end


%% Create Design Matrix for each Task using Hemodynamic Response Function (HRF)

% Create HRF
P = [6 16 1 1 6 0 32]; % Specify parameters of response function
T = 16; % Specify microtime resolution
RT = 1; % Repitition time - Change this according to scanning parameter
[hrf,~] = spm_hrf(RT,P,T); % Create hrf
% Plot HRF
figure; plot(hrf);

% Mental Rotation
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

% Group onset times by condition
for ii = 1:length(mr_storage2)
    condition = mr_storage2(ii).condition;
    condition2 = [0; condition];
    onset_time = mr_storage2(ii).onset_time;
    onset_time2 = mr_storage2(ii).onset_time2;
    easy = onset_time(:,condition2==1);
    med = onset_time(:,condition2==2);
    hard = onset_time(:,condition2==3);
    easy2 = sum(easy,2);
    med2 = sum(med,2);
    hard2 = sum(hard,2);
    if size(onset_time2,1) ~= 0
        easy3 = onset_time2(:,condition2==1);
        med3 = onset_time2(:,condition2==2);
        hard3 = onset_time2(:,condition2==3);
        easy4 = sum(easy3,2);
        med4 = sum(med3,2);
        hard4 = sum(hard3,2);
    else
        easy4 = [];
        med4 = [];
        hard4 = [];
    end
    mr_storage2(ii).easy_OT = easy2;
    mr_storage2(ii).med_OT = med2;
    mr_storage2(ii).hard_OT = hard2;
    mr_storage2(ii).easy_OT2 = easy4;
    mr_storage2(ii).med_OT2 = med4;
    mr_storage2(ii).hard_OT2 = hard4;
    mr_storage2(ii).instruct = onset_time(:,condition2==0);
end

% Convolve each onset time with HRF and create design matrix
time_tot = 601;
for ii = 1:length(mr_storage2)
    easy = mr_storage2(ii).easy_OT;
    med = mr_storage2(ii).med_OT;
    hard = mr_storage2(ii).hard_OT;
    instruct = mr_storage2(ii).instruct;
    easy2 = mr_storage2(ii).easy_OT2;
    med2 = mr_storage2(ii).med_OT2;
    hard2 = mr_storage2(ii).hard_OT2;
    easy_conv = conv(easy,hrf);
    med_conv = conv(med,hrf);
    hard_conv = conv(hard,hrf);
    
    if size(easy2,1) ~= 0
        easy_conv2 = conv(easy2,hrf);
        med_conv2 = conv(med2,hrf);
        hard_conv2 = conv(hard2,hrf);
    else
        easy_conv2 = zeros(601,1);
        med_conv2 = zeros(601,1);
        hard_conv2 = zeros(601,1);
    end

    instruct_conv = conv(instruct,hrf);
    onset_hrf = [instruct_conv(1:time_tot) easy_conv(1:time_tot) med_conv(1:time_tot) hard_conv(1:time_tot)];
    onset_hrf_wrong = [easy_conv2(1:time_tot) med_conv2(1:time_tot) hard_conv2(1:time_tot)];
    mr_storage2(ii).onset_hrf = onset_hrf;
    mr_storage2(ii).onset_hrf_wrong = onset_hrf_wrong;
end

% Spatial Map
% Design onset timing of trials (event-related)
% Encoding
time_tot = 601; % Length of scan timeseries
trial_length = 1; % Extra length (not including the first frame)
for ii = 1:length(sm_storage2)
    trial_num = length(sm_storage2(ii).condition) + 1;
    ts_temp = zeros(time_tot,trial_num);
    onset = sm_storage2(ii).onset;
    onset2 = ceil(onset);
    start = sm_storage2(ii).timing(1);
    ts_temp(start:start+4,1) = 1;
    for jj = 1:size(onset2,1)
        ts_temp(onset2(jj):onset2(jj)+trial_length,jj+1) = 1;
    end
    sm_storage2(ii).onset_time = ts_temp;
end

% Group onset times by condition
for ii = 1:length(sm_storage2)
    condition = sm_storage2(ii).condition;
    condition2 = [0; condition];
    onset_time = sm_storage2(ii).onset_time;
    easy = onset_time(:,condition2==1);
    med = onset_time(:,condition2==2);
    hard = onset_time(:,condition2==3);
    easy2 = sum(easy,2);
    med2 = sum(med,2);
    hard2 = sum(hard,2);
    sm_storage2(ii).easy_OT = easy2;
    sm_storage2(ii).med_OT = med2;
    sm_storage2(ii).hard_OT = hard2;
    sm_storage2(ii).instruct = onset_time(:,condition2==0);
end

% Convolve each onset time with HRF and create design matrix
time_tot = 601;
for ii = 1:length(sm_storage2)
    easy = sm_storage2(ii).easy_OT;
    med = sm_storage2(ii).med_OT;
    hard = sm_storage2(ii).hard_OT;
    instruct = sm_storage2(ii).instruct;
    easy_conv = conv(easy,hrf);
    med_conv = conv(med,hrf);
    hard_conv = conv(hard,hrf);
    instruct_conv = conv(instruct,hrf);
    onset_hrf = [instruct_conv(1:time_tot) easy_conv(1:time_tot) med_conv(1:time_tot) hard_conv(1:time_tot)];
    sm_storage2(ii).onset_hrf_encode = onset_hrf;
end

% Response
time_tot = 601; % Length of scan timeseries
trial_length = 2; % Extra length (not including the first frame)
for ii = 1:length(sm_storage2)
    trial_num = length(sm_storage2(ii).condition);
    ts_temp = zeros(time_tot,trial_num);
    onset = sm_storage2(ii).onset;
    onset2 = ceil(onset)+2;
    for jj = 1:size(onset2,1)
        ts_temp(onset2(jj):onset2(jj)+trial_length,jj) = 1;
    end
    sm_storage2(ii).onset_time = ts_temp;
end

% Group onset times by condition
for ii = 1:length(sm_storage2)
    condition = sm_storage2(ii).condition;
    onset_time = sm_storage2(ii).onset_time;
    easy = onset_time(:,condition==1);
    med = onset_time(:,condition==2);
    hard = onset_time(:,condition==3);
    easy2 = sum(easy,2);
    med2 = sum(med,2);
    hard2 = sum(hard,2);
    sm_storage2(ii).easy_OT = easy2;
    sm_storage2(ii).med_OT = med2;
    sm_storage2(ii).hard_OT = hard2;
end

% Convolve each onset time with HRF and create design matrix
time_tot = 601;
for ii = 1:length(sm_storage2)
    easy = sm_storage2(ii).easy_OT;
    med = sm_storage2(ii).med_OT;
    hard = sm_storage2(ii).hard_OT;
    easy_conv = conv(easy,hrf);
    med_conv = conv(med,hrf);
    hard_conv = conv(hard,hrf);
    onset_hrf = [easy_conv(1:time_tot) med_conv(1:time_tot) hard_conv(1:time_tot)];
    sm_storage2(ii).onset_hrf_resp = onset_hrf;
end

% Spatial Map as whole task
time_tot = 601; % Length of scan timeseries
trial_length = 4; % Extra length (not including the first frame)
for ii = 1:length(sm_storage2)
    trial_num = length(sm_storage2(ii).condition) + 1;
    ts_temp = zeros(time_tot,trial_num);
    onset = sm_storage2(ii).onset;
    onset2 = ceil(onset);
    start = sm_storage2(ii).timing(1);
    ts_temp(start:start+4,1) = 1;
    for jj = 1:size(onset2,1)
        ts_temp(onset2(jj):onset2(jj)+trial_length,jj+1) = 1;
    end
    sm_storage2(ii).onset_time = ts_temp;
end

% Group onset times by condition
for ii = 1:length(sm_storage2)
    condition = sm_storage2(ii).condition;
    condition2 = [0; condition];
    onset_time = sm_storage2(ii).onset_time;
    easy = onset_time(:,condition2==1);
    med = onset_time(:,condition2==2);
    hard = onset_time(:,condition2==3);
    easy2 = sum(easy,2);
    med2 = sum(med,2);
    hard2 = sum(hard,2);
    sm_storage2(ii).easy_OT = easy2;
    sm_storage2(ii).med_OT = med2;
    sm_storage2(ii).hard_OT = hard2;
    sm_storage2(ii).instruct = onset_time(:,condition2==0);
end

% Convolve each onset time with HRF and create design matrix
time_tot = 601;
for ii = 1:length(sm_storage2)
    easy = sm_storage2(ii).easy_OT;
    med = sm_storage2(ii).med_OT;
    hard = sm_storage2(ii).hard_OT;
    instruct = sm_storage2(ii).instruct;
    easy_conv = conv(easy,hrf);
    med_conv = conv(med,hrf);
    hard_conv = conv(hard,hrf);
    instruct_conv = conv(instruct,hrf);
    onset_hrf = [instruct_conv(1:time_tot) easy_conv(1:time_tot) med_conv(1:time_tot) hard_conv(1:time_tot)];
    sm_storage2(ii).onset_hrf_all = onset_hrf;
end

% Response Alternative
time_tot = 601; % Length of scan timeseries
trial_length = 4; % Extra length (not including the first frame)
for ii = 1:length(ra_storage2)
    trial_num = length(ra_storage2(ii).condition) + 1;
    ts_temp = zeros(time_tot,trial_num);
    onset = ra_storage2(ii).onset;
    onset2 = ceil(onset);
    start = ra_storage2(ii).timing(1);
    ts_temp(start:start+4,1) = 1;
    for jj = 1:size(onset2,1)
        ts_temp(onset2(jj):onset2(jj)+trial_length,jj+1) = 1;
    end
    ra_storage2(ii).onset_time = ts_temp;
end

% Group onset times by condition
for ii = 1:length(ra_storage2)
    condition = ra_storage2(ii).condition;
    condition2 = [0; condition];
    onset_time = ra_storage2(ii).onset_time;
    easy = onset_time(:,condition2==1);
    med = onset_time(:,condition2==2);
    hard = onset_time(:,condition2==3);
    easy2 = sum(easy,2);
    med2 = sum(med,2);
    hard2 = sum(hard,2);
    ra_storage2(ii).easy_OT = easy2;
    ra_storage2(ii).med_OT = med2;
    ra_storage2(ii).hard_OT = hard2;
    ra_storage2(ii).instruct = onset_time(:,condition2==0);
end

% Convolve each onset time with HRF and create design matrix
time_tot = 601;
for ii = 1:length(ra_storage2)
    easy = ra_storage2(ii).easy_OT;
    med = ra_storage2(ii).med_OT;
    hard = ra_storage2(ii).hard_OT;
    instruct = ra_storage2(ii).instruct;
    easy_conv = conv(easy,hrf);
    med_conv = conv(med,hrf);
    hard_conv = conv(hard,hrf);
    instruct_conv = conv(instruct,hrf);
    onset_hrf = [instruct_conv(1:time_tot) easy_conv(1:time_tot) med_conv(1:time_tot) hard_conv(1:time_tot)];
    ra_storage2(ii).onset_hrf_all = onset_hrf;
end

% Object n-back task
time_tot = 601; % Length of scan timeseries
trial_length = 1; % Extra length (not including the first frame)
for ii = 1:length(nbackPic_storage2)
    trial_num = length(nbackPic_storage2(ii).condition) + 1;
    ts_temp = zeros(time_tot,trial_num);
    onset = nbackPic_storage2(ii).onset;
    onset2 = ceil(onset);
    start = nbackPic_storage2(ii).timing(1);
    ts_temp(start:start+4,1) = 1;
    for jj = 1:size(onset2,1)
        ts_temp(onset2(jj):onset2(jj)+trial_length,jj+1) = 1;
    end
    nbackPic_storage2(ii).onset_time = ts_temp;
end

% Group onset times by condition
for ii = 1:length(nbackPic_storage2)
    condition = nbackPic_storage2(ii).condition;
    condition2 = [0; condition];
    onset_time = nbackPic_storage2(ii).onset_time;
    easy = onset_time(:,condition2==1);
    med = onset_time(:,condition2==2);
    hard = onset_time(:,condition2==3);
    easy2 = sum(easy,2);
    med2 = sum(med,2);
    hard2 = sum(hard,2);
    nbackPic_storage2(ii).easy_OT = easy2;
    nbackPic_storage2(ii).med_OT = med2;
    nbackPic_storage2(ii).hard_OT = hard2;
    nbackPic_storage2(ii).instruct = onset_time(:,condition2==0);
end

% Convolve each onset time with HRF and create design matrix
time_tot = 601;
for ii = 1:length(nbackPic_storage2)
    easy = nbackPic_storage2(ii).easy_OT;
    med = nbackPic_storage2(ii).med_OT;
    hard = nbackPic_storage2(ii).hard_OT;
    instruct = nbackPic_storage2(ii).instruct;
    easy_conv = conv(easy,hrf);
    med_conv = conv(med,hrf);
    hard_conv = conv(hard,hrf);
    instruct_conv = conv(instruct,hrf);
    onset_hrf = [instruct_conv(1:time_tot) easy_conv(1:time_tot) med_conv(1:time_tot) hard_conv(1:time_tot)];
    nbackPic_storage2(ii).onset_hrf_all = onset_hrf(:,1:2);
end


%% Shorten design matrices and timeseries to only include TRs for
% appropriate task
for ii = 1:length(mr_storage2)
    onset_hrf = mr_storage2(ii).onset_hrf;
    timing = mr_storage2(ii).timing;
    onset_hrf2 = onset_hrf(timing(1):timing(2),:);
    mr_storage2(ii).onset_hrf2 = onset_hrf2;
end

for ii = 1:length(sm_storage2)
    onset_hrf1 = sm_storage2(ii).onset_hrf_encode;
    onset_hrf2 = sm_storage2(ii).onset_hrf_resp;
    timing = sm_storage2(ii).timing;
    onset_hrf3 = onset_hrf1(timing(1):timing(2),:);
    onset_hrf4 = onset_hrf2(timing(1):timing(2),:);
    sm_storage2(ii).onset_hrf_encode2 = onset_hrf3;
    sm_storage2(ii).onset_hrf_resp2 = onset_hrf4;
end

% Shorten timeseries to only task block
for ii = 1:length(storage)
    ts = storage(ii).ts;
    timing_mr = mr_storage2(ii).timing;
    timing_sm = sm_storage2(ii).timing;
    ts_mr = ts(timing_mr(1):timing_mr(2),:);
    ts_sm = ts(timing_sm(1):timing_sm(2),:);
    mr_storage2(ii).ts_mr = ts_mr;
    sm_storage2(ii).ts_sm = ts_sm;
    mr_storage2(ii).ts = ts;
    sm_storage2(ii).ts = ts;
end


%% Collate scans across runs

% Get unique identifiers for each subject
scan_name = vertcat(mr_storage2.name);
% Only keep characters for subject ID
sub_id = {};
for ii = 1:size(scan_name,1)
    sub_id{ii,1} = extractBefore(scan_name(ii,:),'_ses');
end
% Check number of unique IDs
length(unique(sub_id)) % 24
sub_id2 = unique(sub_id);

% Collate all scans from same subject
mr_collate = struct('name',1,'response_time',1,'dm',1,'scan_id',1,'ses_id',1);
for ii = 1:length(sub_id2)
    mr_collate(ii).name = sub_id2(ii);
    a = 1;
    dm = [];
    scan_id = [];
    ses_id = [];
    RT = [];
    ts = [];
    dm2 = [];
    for jj = 1:length(mr_storage2)
        name = mr_storage2(jj).name;
        if contains(name,sub_id2(ii))
            dm = [dm; mr_storage2(jj).onset_hrf];
            dm2 = [dm2; mr_storage2(jj).onset_hrf_wrong];
            scan_id = [scan_id; repelem(a,size(mr_storage2(jj).onset_hrf,1),1)];
            RT = [RT; mr_storage2(jj).response_time];
            ts = [ts; storage(jj).ts];
            a = a + 1;
            if contains(name,'b1')
                ses_id = [ses_id; repelem(1,size(mr_storage2(jj).onset_hrf,1),1)];
            elseif contains(name,'b2')
                ses_id = [ses_id; repelem(2,size(mr_storage2(jj).onset_hrf,1),1)];
            end
        end
    end
    mr_collate(ii).dm = dm;
    mr_collate(ii).scan_id = scan_id;
    mr_collate(ii).ses_id = ses_id;
    mr_collate(ii).response_time = RT;
    mr_collate(ii).ts = ts;
    mr_collate(ii).dm_wrong = dm2;
end

% Generate generalised linear mixed model for each subject
% Compares correct vs. incorrect
% Scan id = random effect, task difficulty = fixed effect, ts = response
rois = 1102;
coef_mr = zeros(8,rois,length(sub_id2));
for ii = 1:length(mr_collate)
    ts = mr_collate(ii).ts;
    dm = mr_collate(ii).dm;
    dm2 = mr_collate(ii).dm_wrong;
    instruct = dm(:,1); 
    easy = dm(:,2);
    med = dm(:,3);
    hard = dm(:,4);
    easy2 = dm2(:,1);
    med2 = dm2(:,2);
    hard2 = dm2(:,3);
    scan_id = mr_collate(ii).scan_id;
    ses_id = mr_collate(ii).ses_id;
    glme_store_all = [];
    for jj = 1:size(ts,2)
        disp([ii,jj])
        tbl = table(ts(:,jj),instruct,easy,med,hard,easy2,med2,hard2,scan_id,ses_id);
        glme = fitglme(tbl,['Var1 ~ 1 + instruct + easy + med + hard + easy2 + ' ...
            'med2 + hard2 + ses_id + (1|scan_id)']);
        glme_store_all(1,jj) = glme.Coefficients(2,2); % instruct
        glme_store_all(2,jj) = glme.Coefficients(3,2); % easy
        glme_store_all(3,jj) = glme.Coefficients(4,2); % med
        glme_store_all(4,jj) = glme.Coefficients(5,2); % hard
        glme_store_all(5,jj) = glme.Coefficients(6,2); % easy wrong
        glme_store_all(6,jj) = glme.Coefficients(7,2); % med wrong
        glme_store_all(7,jj) = glme.Coefficients(8,2); % hard wrong
        glme_store_all(8,jj) = glme.Coefficients(9,2); % session (day)
        % glme_res(:,jj) = residuals(glme,'Conditional',true,'ResidualType','Pearson');
    end
    coef_mr(:,:,ii) = glme_store_all; % regressor X ROI X subject
    % mr_collate(ii).glme_res = glme_res;
end

% Group-level analysis (one-sample t-test)
group_coef = zeros(size(coef_mr,2),2);
for ii = 1:size(coef_mr,2)
    [h,p] = ttest(squeeze(coef_mr(3,ii,:)));
    group_coef(ii,1) = h;
    group_coef(ii,2) = p;
end
mr_sig = zeros(length(group_coef),1);
mean_mr = mean(squeeze(coef_mr(3,:,:)),2);
mr_sig(group_coef(:,1)==1) = mean_mr(group_coef(:,1)==1);
figure; scatter(1:length(mr_sig),mr_sig,20,'filled');

% Plot on surface
limits = [min(mr_sig) max(mr_sig)]; % [-0.7021 1.2914]
surf_schaef2(mr_sig(1:400),[-0.7021 1.2914]);
surf_cbm(mr_sig(455:482));
subcort_plot(mr_sig); colormap(bluewhitered());


%% Combined GLM with all relevant task regressors

% Load in mr_collate1000, sm_collate3, ra_collate, nbackPic_collate, sub_id2

rois = 1102;
coef_glm = zeros(15,rois,length(sub_id2));
for ii = 1:length(mr_collate)
    ts = mr_collate(ii).ts;
    dm_mr = mr_collate(ii).dm;
    dm_sm = sm_collate(ii).dm;
    dm_ra = ra_collate(ii).dm;
    dm_n2 = nbackPic_collate(ii).dm;
    instruct_mr = dm_mr(:,1);
    easy_mr = dm_mr(:,2);
    med_mr = dm_mr(:,3);
    hard_mr = dm_mr(:,4);
    instruct_sm = dm_sm(:,1);
    easy_sm = dm_sm(:,2);
    med_sm = dm_sm(:,3);
    hard_sm = dm_sm(:,4);
    instruct_ra = dm_ra(:,1);
    easy_ra = dm_ra(:,2);
    med_ra = dm_ra(:,3);
    hard_ra = dm_ra(:,4);
    instruct_n2 = dm_n2(:,1);
    task_n2 = dm_n2(:,2);
    scan_id = mr_collate(ii).scan_id;
    ses_id = mr_collate(ii).ses_id;
    glme_store_all = [];
    for jj = 1:size(ts,2)
        disp([ii,jj])
        tbl = table(ts(:,jj),instruct_mr,easy_mr,med_mr,hard_mr,instruct_sm,easy_sm, ...
            med_sm,hard_sm,instruct_ra,easy_ra,med_ra,hard_ra,instruct_n2, ...
            task_n2,scan_id,ses_id);
        glme = fitglme(tbl,['Var1 ~ 1 + instruct_mr + easy_mr + med_mr + hard_mr + ' ...
            'instruct_sm + easy_sm + med_sm + hard_sm + instruct_ra + easy_ra + ' ...
            'med_ra + hard_ra + instruct_n2 + task_n2 + ses_id + (1|scan_id)']);
        glme_store_all(1,jj) = glme.Coefficients(2,2); % instruct mr
        glme_store_all(2,jj) = glme.Coefficients(3,2); % easy mr
        glme_store_all(3,jj) = glme.Coefficients(4,2); % med mr
        glme_store_all(4,jj) = glme.Coefficients(5,2); % hard mr
        glme_store_all(5,jj) = glme.Coefficients(6,2); % instruct sm
        glme_store_all(6,jj) = glme.Coefficients(7,2); % easy sm
        glme_store_all(7,jj) = glme.Coefficients(8,2); % med sm
        glme_store_all(8,jj) = glme.Coefficients(9,2); % hard sm
        glme_store_all(9,jj) = glme.Coefficients(10,2); % instruct ra
        glme_store_all(10,jj) = glme.Coefficients(11,2); % easy ra
        glme_store_all(11,jj) = glme.Coefficients(12,2); % med ra
        glme_store_all(12,jj) = glme.Coefficients(13,2); % hard ra
        glme_store_all(13,jj) = glme.Coefficients(14,2); % instruct n2
        glme_store_all(14,jj) = glme.Coefficients(15,2); % task n2
        glme_store_all(15,jj) = glme.Coefficients(16,2); % day
        % glme_res(:,jj) = residuals(glme,'Conditional',true,'ResidualType','Pearson');
    end
    coef_glm(:,:,ii) = glme_store_all; % regressor X ROI X subject
    % mr_collate(ii).glme_res = glme_res;
end
% 2nd-level statistic testing - Comparing linear relationship across task
% difficulty
mr_coef = coef_glm(2:4,:,:);
glme_store = [];
sub_id = repelem([1:24]',3,1);
difficulty = repmat([1:3]',24,1);
for jj = 1:size(mr_coef,2)
    message = 'Running GLME on roi %d\n';
    fprintf(message,jj);
    roi = reshape(squeeze(mr_coef(:,jj,:)),[],1);
    tbl = table(roi,difficulty,sub_id);
    glme = fitglme(tbl,'roi ~ 1 + difficulty + (1|sub_id)');
    glme_store(jj,1) = glme.Coefficients(2,2); % difficulty coef
    glme_store(jj,2) = glme.Coefficients(2,6); % p-value
end

% False Discovery Rate correction
[h,~,~,adj_p] = fdr_bh(glme_store(:,2));
% Take significant regions based off Q-value
mr_sig = zeros(1102,1);
mr_sig(h==1,1) = glme_store(h==1,1);

% Plot on surface
limits = [min(mr_sig) max(mr_sig)]; % [-0.2273 0.5513]
surf_schaef1000(mr_sig(1:1000),limits);
surf_cbm(mr_sig(1001:1028),limits);
% Basal Ganglia
bg_plot = zeros(1,14);
bg_plot(1,1) = mean(mr_sig(1049:1050));
bg_plot(1,3) = mean(mr_sig(1045:1048));
bg_plot(1,5) = mean(mr_sig(1051:1052));
bg_plot(1,6) = mean(mr_sig(1041:1044));
bg_plot(1,8) = mean(mr_sig(1037:1038));
bg_plot(1,10) = mean(mr_sig(1033:1036));
bg_plot(1,12) = mean(mr_sig(1039:1040));
bg_plot(1,13) = mean(mr_sig(1029:1032));
figure; plot_subcortical(bg_plot,'ventricles','False');
colormap(custom());
