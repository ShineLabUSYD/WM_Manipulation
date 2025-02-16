%% Attractor Landscape (Task Difficulty)

% Calculate the likelihood of change during a trial per difficulty (MSD)
% Calculate the likelihood of BOLD amplitude during a trial
% Compare landscapes across difficulties for each ROI/network

% Load in data
load('C:\Users\JoshB\OneDrive\Documents\University\PhD\workingmemory\Code\data\coef_FIR.mat');
load('C:\Users\JoshB\OneDrive\Documents\University\PhD\workingmemory\Code\data\coef_FIRwrong.mat');
load('C:\Users\JoshB\OneDrive\Documents\University\PhD\workingmemory\Code\data\latest2\C1.mat');
load('C:\Users\JoshB\OneDrive\Documents\University\PhD\workingmemory\Code\data\latest2\xcf_map.mat');


% Data is timepoints X ROI X subject
% Timepoints = 15 X difficulty

% Select difficulty
sample_FIR = coef_FIRwrong(1:15,C1==1,:);
sample_FIR = coef_FIR(31:45,C1==1,:);

% Check min and max values of sample_FIR
%sample_FIR = mean(sample_FIR,3);
limits = [min(sample_FIR,[],'all') max(sample_FIR,[],'all')];
ds = 0:0.3:1.8;
% Loop through each timepoint and fit a pdf
nrg = [];
for ii = 1:size(sample_FIR,1)
    disp(ii)
    
    % Flatten out data
    BOLD_t = abs(reshape(sample_FIR(ii,:),[],1));

    % Fit pdf on data
    pd = fitdist(BOLD_t,'Kernel','BandWidth',0.1);
    pdfEstimate = pdf(pd,ds);

    % Calculate energy
    nrg(:,ii) = -1.*log(pdfEstimate);
end

% Define the colors in RGB
brown = [143/255, 52/255, 13/255]; % Brown
blue = [15/255, 162/255, 240/255];    % Blue
white = [1, 1, 1];                  % White

% Create the colormap
nColors = 256; % Number of colors in the colormap
cMap = zeros(nColors, 3); % Initialize the colormap

% Interpolate between brown and blue
for i = 1:nColors/2
    cMap(i, :) = brown + (blue - brown) * (i - 1) / (nColors/2 - 1);
end

% Interpolate between blue and white
for i = nColors/2+1:nColors
    cMap(i, :) = blue + (white - blue) * (i - nColors/2 - 1) / (nColors/2 - 1);
end



% 3D plot
figure;
mesh(1:15,ds,nrg_easy,'EdgeColor','black','FaceColor','interp');
xlabel('Time points (s)')
ylabel('BOLD Amp')
zlabel('MSD  energy')
zlim([-10 60]);
colormap(cMap);

figure;
mesh(1:15,ds,nrg_med,'EdgeColor','black','FaceColor','interp');
xlabel('Time points (s)')
ylabel('BOLD Amp')
zlabel('MSD  energy')
zlim([-10 60]);
colormap(cMap);

figure;
mesh(1:15,ds,nrg_hard,'EdgeColor','black','FaceColor','interp');
%xlabel('Time points (s)')
%ylabel('BOLD Amp')
%zlabel('MSD  energy')
zlim([-10 65]);
colormap(cMap);
set(gca,'box','off','FontSize',24,'FontName','Arial','linew',1.5);

figure;
mesh(1:15,ds,nrg_hardwrong,'EdgeColor','black','FaceColor','interp');
%xlabel('Time points (s)')
%ylabel('BOLD Amp')
%zlabel('MSD  energy')
zlim([-10 65]);
colormap(cMap);
set(gca,'box','off','FontSize',24,'FontName','Arial','linew',1.5);

