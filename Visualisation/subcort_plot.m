function [f,mean_subcort] = subcort_plot(data,lim)
%% plot subcortical from voltron
% Usage
% data = nodes x 1
%function plot_subcortical.m
%data (double array) - vector of data, size = [1 x v]. One value per 
%       subcortical structure, in this order: L-accumbens, L-amygdala, 
%       L-caudate, L-hippocampus, L-pallidum L-putamen, L-thalamus, 
%       L-ventricle, R-accumbens, R-amygdala, R-caudate, R-hippocampus, 
%       R-pallidum, R-putamen, R-thalamus, R-ventricle
%RH first

if nargin == 2
    limits = lim;
else
    limits = [min(data) max(data)];
end

%RHs ROIs
R_hippo_loc = [401,402,405:407];
R_hippocampus = mean(data(R_hippo_loc),1);
R_thal_loc = [403:404,408:412,423];
R_thalamus = mean(data(R_thal_loc),1);
R_putamen_loc = [413:416];
R_putamen = mean(data(R_putamen_loc),1);
R_caudate_loc = [417:420];
R_caudate = mean(data(R_caudate_loc),1);
R_amygdala_loc = [421:422];
R_amygdala = mean(data(R_amygdala_loc),1);
R_accumbens_loc = [424:425];
R_accumbens = mean(data(R_accumbens_loc),1);
R_pallidum_loc = [426:427];
R_pallidum = mean(data(R_pallidum_loc),1);

L_hippo_loc = [428:429,432:434];
L_hippocampus = mean(data(L_hippo_loc),1);
L_thal_loc = [430:431,435:439,450];
L_thalamus = mean(data(L_thal_loc),1);
L_putamen_loc = [440:443];
L_putamen = mean(data(L_putamen_loc),1);
L_caudate_loc = [444:447];
L_caudate = mean(data(L_caudate_loc),1);
L_amygdala_loc = [448:449];
L_amygdala = mean(data(L_amygdala_loc),1);
L_accumbens_loc = [451:452];
L_accumbens = mean(data(L_accumbens_loc),1);
L_pallidum_loc = [453:454];
L_pallidum = mean(data(L_pallidum_loc),1);

mean_subcort = zeros(14,1);
mean_subcort(1,:) = L_accumbens;
mean_subcort(2,:) = L_amygdala;
mean_subcort(3,:) = L_caudate;
mean_subcort(4,:) = L_hippocampus;
mean_subcort(5,:) = L_pallidum;
mean_subcort(6,:) = L_putamen;
mean_subcort(7,:) = L_thalamus;

mean_subcort(8,:) = R_accumbens;
mean_subcort(9,:) = R_amygdala;
mean_subcort(10,:) = R_caudate;
mean_subcort(11,:) = R_hippocampus;
mean_subcort(12,:) = R_pallidum;
mean_subcort(13,:) = R_putamen;
mean_subcort(14,:) = R_thalamus;

%mean_subcort(mean_subcort>0) = 1;
%mean_subcort = ceil(mean_subcort);

% mean_subcort = zeros(1,14);
% mean_subcort(1,1) = mean(data(449:450));
% mean_subcort(1,3) = mean(data(445:448));
% mean_subcort(1,5) = mean(data(451:452));
% mean_subcort(1,6) = mean(data(441:444));
% mean_subcort(1,8) = mean(data(437:438));
% mean_subcort(1,10) = mean(data(433:436));
% mean_subcort(1,12) = mean(data(439:440));
% mean_subcort(1,13) = mean(data(429:432));
% mean_subcort = mean_subcort';

    if size(mean_subcort,1)==14
        sprintf('%s','subcort array correct!')
    else 
        sprintf('%s','does not fit!')
    end

%plotting function

%can select own colour scheme
% max1 = max(mean_subcort);
% min1 = min(mean_subcort);
% clims = [min1 max1];
% limits = [-max1 max1];
%load('C:/Users/natas/Documents/PhD/Code/OrangeBlueColorMap.mat')
%C = bluewhitered();
f = figure;
plot_subcortical(mean_subcort, 'ventricles','False','color_range',limits)
%plot_subcortical(mean_subcort, 'ventricles','False')
%colormap(C)
%caxis(limits)

end

