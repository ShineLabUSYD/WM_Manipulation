function newmap = custom(m)
%BLUEWHITERED   Blue, white, and red color map.
%   BLUEWHITERED(M) returns an M-by-3 matrix containing a blue to white
%   to red colormap, with white corresponding to the CAXIS value closest
%   to zero.  This colormap is most useful for images and surface plots
%   with positive and negative values.  BLUEWHITERED, by itself, is the
%   same length as the current colormap.
%
%   Examples:
%   ------------------------------
%   figure
%   imagesc(peaks(250));
%   colormap(bluewhitered(256)), colorbar
%
%   figure
%   imagesc(peaks(250), [0 8])
%   colormap(bluewhitered), colorbar
%
%   figure
%   imagesc(peaks(250), [-6 0])
%   colormap(bluewhitered), colorbar
%
%   figure
%   surf(peaks)
%   colormap(bluewhitered)
%   axis tight
%
%   See also HSV, HOT, COOL, BONE, COPPER, PINK, FLAG, 
%   COLORMAP, RGBPLOT.
if nargin < 1
   m = size(get(gcf,'colormap'),1);
end

% Blue to Orange
bottom = [0 153 227]./255;
middle = [1 1 1];
top = [227 74 0]./255;
botmiddle = ((bottom + middle)./2);
topmiddle = ((top + middle)./2);

% Purple to Green
% top = [187 0 146]./255;
% middle = [1 1 1];
% bottom = [0 187 41]./255;
% botmiddle = ((bottom + middle)./2);
% topmiddle = ((top + middle)./2);

% Teal - Red
% bottom = [226 120 133]./255;
% middle = [1 1 1];
% top = [120 226 213]./255;
% % top = [62 185 234]./255;
% botmiddle = ((bottom + middle)./2);
% topmiddle = ((top + middle)./2);

% Teal - Coral
% bottom = [0 153 153]./255;
% middle = [1 1 1];
% top = [255 127 80]./255;
% botmiddle = ((bottom + middle)./2);
% topmiddle = ((top + middle)./2);

% Gold - Purple
% bottom = [255 215 0]./255;
% middle = [1 1 1];
% top = [128 0 128]./255;
% botmiddle = ((bottom + middle)./2);
% topmiddle = ((top + middle)./2);

% Green - Pink
% bottom = [0 128 0]./255;
% middle = [1 1 1];
% top = [255 105 180]./255;
% botmiddle = ((bottom + middle)./2);
% topmiddle = ((top + middle)./2);

% Red to Gold
% top = [230 184 0]./255;
% middle = [1 1 1];
% bottom = [200 0 0]./255;
% botmiddle = (bottom + middle)./2;
% topmiddle = (top + middle)./2;

% Random
% c1 = [227 74 0]./255;
% c2 = [0 187 41]./255;
% gold = (c1 + c2)./2;
% top = [124 68 79]./255;
% middle = [1 1 1];
% bottom = [200 0 0]./255;
% botmiddle = (bottom + middle)./2;
% topmiddle = (top + middle)./2;

% Find middle
lims = get(gca, 'CLim');
% Find ratio of negative to positive
if (lims(1) < 0) & (lims(2) > 0)
    % It has both negative and positive
    % Find ratio of negative to positive
    ratio = abs(lims(1)) / (abs(lims(1)) + lims(2));
    neglen = round(m*ratio);
    poslen = m - neglen;
    
    % Just negative
    new = [bottom; botmiddle; middle];
    len = length(new);
    oldsteps = linspace(0, 1, len);
    newsteps = linspace(0, 1, neglen);
    newmap1 = zeros(neglen, 3);
    
    for i=1:3
        % Interpolate over RGB spaces of colormap
        newmap1(:,i) = min(max(interp1(oldsteps, new(:,i), newsteps)', 0), 1);
    end
    
    % Just positive
    new = [middle; topmiddle; top];
    len = length(new);
    oldsteps = linspace(0, 1, len);
    newsteps = linspace(0, 1, poslen);
    newmap = zeros(poslen, 3);
    
    for i=1:3
        % Interpolate over RGB spaces of colormap
        newmap(:,i) = min(max(interp1(oldsteps, new(:,i), newsteps)', 0), 1);
    end
    
    % And put 'em together
    newmap = [newmap1; newmap];
    
elseif lims(1) >= 0
    % Just positive
    new = [middle; topmiddle; top];
    len = length(new);
    oldsteps = linspace(0, 1, len);
    newsteps = linspace(0, 1, m);
    newmap = zeros(m, 3);
    
    for i=1:3
        % Interpolate over RGB spaces of colormap
        newmap(:,i) = min(max(interp1(oldsteps, new(:,i), newsteps)', 0), 1);
    end
    
else
    % Just negative
    new = [bottom; botmiddle; middle];
    len = length(new);
    oldsteps = linspace(0, 1, len);
    newsteps = linspace(0, 1, m);
    newmap = zeros(m, 3);
    
    for i=1:3
        % Interpolate over RGB spaces of colormap
        newmap(:,i) = min(max(interp1(oldsteps, new(:,i), newsteps)', 0), 1);
    end
    
end
% 
% m = 64;
% new = [bottom; botmiddle; middle; topmiddle; top];
% % x = 1:m;
% 
% oldsteps = linspace(0, 1, 5);
% newsteps = linspace(0, 1, m);
% newmap = zeros(m, 3);
% 
% for i=1:3
%     % Interpolate over RGB spaces of colormap
%     newmap(:,i) = min(max(interp1(oldsteps, new(:,i), newsteps)', 0), 1);
% end
% 
% % set(gcf, 'colormap', newmap), colorbar