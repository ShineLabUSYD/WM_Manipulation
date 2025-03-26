% Plot a vector of schaefer parcellation loadings on the FS average surface
%
% NOTE: this function requires the 32k FS average .gii files and the matching
% Schaefer parcellation .gii files to work.
%
%
% ARGUMENTS:
%        data  -- a 1D array of size (400 x 1) with regional loadings for
%        the Schaefer parcellation
%        name (optional) -- a filename for saving an output .gii of
%        resultant plot
%
% OUTPUT:
%        left(optional) -- left hemisphere schaefer loadings on the 32k FS average
%        surface
%        right(optional) -- right hemisphere schaefer loadings on the 32k FS average
%        surface
%
% REFERENCES:
%        Schaefer, Alexander, et al. Cerebral cortex 28.9 (2018): 3095-3114.
%
% AUTHOR:
%     Original: James Mac Shine
%     Eli J Muller (2020).
%
% USAGE:
%{
    %
    data = zeros(n_time, n_regions);
    [pc_vec, pc_val] = pca(data);
    surf_schaef(pc_vec(:,1))
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [left, right] = surf_schaef2(data,lim,colour)
    
    % -------- Load Structural Files --------    
    % Average surface
    struc_L = gifti('C:/Users/JoshB/OneDrive/Documents/MATLAB/Functions/plotting/surf_schaef2/Conte69.L.midthickness.32k_fs_LR.surf.gii');
    struc_R = gifti('C:/Users/JoshB/OneDrive/Documents/MATLAB/Functions/plotting/surf_schaef2/Conte69.R.midthickness.32k_fs_LR.surf.gii');
    
    %struc_L = gifti('C:\Users\JoshB\OneDrive\Documents/MATLAB_Analysis/MATLAB/ShineRetreat_2022/fs_LR.32k.L.flat.surf.gii');
    %struc_R = gifti('C:\Users\JoshB\OneDrive\Documents/MATLAB_Analysis/MATLAB/ShineRetreat_2022/fs_LR.32k.R.flat.surf.gii');

    % Schaefer Parcellation
    left = gifti('C:/Users/JoshB/OneDrive/Documents/MATLAB/Functions/plotting/surf_schaef2/schaef_left.func.gii');
    right = gifti('C:/Users/JoshB/OneDrive/Documents/MATLAB/Functions/plotting/surf_schaef2/schaef_right.func.gii');
    
    % identity matrix
    idx = zeros(401,2);
    idx(2:401,1) = 1:1:400;
    idx(2:401,1) = 1:1:400;
    idx(2:401,2) = data;
    
    % Set defaults
    limits = [min(data) max(data)];
    cmap = 'parula';
    
    % Optional Arguments
    if nargin == 2
        limits = lim;
        % Map the 400 schaefer parcels to the 32k FS average surface faces
        % 
        % left
        original = left.cdata; % Grab the schaefer index for each face on the FS average surface
        [~,index_net] = ismember(original,idx(:,1)); % Find the matching 400 parcels
        map_net = idx(:,2); % Set the 400 parcel loadings
        left.cdata = map_net(index_net); % Duplicate each schaefer loading for all of the 32k faces within the same parcel
    
    
        % Lateral view
        figure; plot(struc_L,left); % plot the surface
        caxis(limits); % scale the axis
        view(-90,10); % fix view angle
        colormap(custom());
        %colormap(multigradient('preset','div.cb.spectral.10'));
        material([0.1 1 0]);
    
        c = colorbar; 
        c.Location = 'eastoutside'; % 'eastoutside' for vertical colorbar
        c.Position = [0.8 0.11 0.0381 0.815]; % Vertical Colorbar [x y width height]
        c.LineWidth = 2;
    
        c.Location = 'southoutside'; % 'southoutside' for horizontal colorbar
        c.Position(4) = 0.75*c.Position(4); % Horizontal Colorbar
        c.TickDirection = "out";
        
        set(gca,'FontSize',24,'FontName','Arial');
        
        
        % Medial view
        figure; plot(struc_L,left);
        caxis(limits); % scale the axis
        view(90,10);
        lgt = findobj(gcf,'Type','Light');
        lgt = lgt(1);
        [~,el] = lightangle(lgt);
        lightangle(lgt,20,el);
        colormap(custom());
        %colormap(multigradient('preset','div.cb.spectral.10'));
        material([0.1 1 0]);
        
        
        % Map the 400 schaefer parcels to the 32k FS average surface faces
        % 
        % right
        original = right.cdata; % Grab the schaefer index for each face on the FS average surface
        [~,index_net] = ismember(original,idx(:,1)); % Find the matching 400 parcels
        map_net = idx(:,2); % Set the 400 parcel loadings
        right.cdata = map_net(index_net); % Duplicate each schaefer loading for all of the 32k faces within the same parcel
    
    
        % Lateral view
        figure; plot(struc_R,right);
        caxis(limits);
        view(90,10);
        colormap(custom());
        %colormap(multigradient('preset','div.cb.spectral.10'));
        
        lgt = findobj(gcf,'Type','Light');
        lgt = lgt(1);
        [~,el] = lightangle(lgt);
        lightangle(lgt,60,el);
        material([0 1 0]);
        %set('AmbientStrength',0.2);
        %set('SpecularStrength',0.4);
        
        
        % Medial view
        figure; plot(struc_R,right);
        caxis(limits); % scale the axis
        view(-90,10);
        lgt = findobj(gcf,'Type','Light');
        lgt = lgt(1);
        [~,el] = lightangle(lgt);
        lightangle(lgt,240,el);
        colormap(custom());
        %colormap(multigradient('preset','div.cb.spectral.10'));
        material([0.1 1 0]);
    end

    if nargin == 3
        limits = lim;
        cmap = colour;
        % Map the 400 schaefer parcels to the 32k FS average surface faces
        % 
        % left
        original = left.cdata; % Grab the schaefer index for each face on the FS average surface
        [~,index_net] = ismember(original,idx(:,1)); % Find the matching 400 parcels
        map_net = idx(:,2); % Set the 400 parcel loadings
        left.cdata = map_net(index_net); % Duplicate each schaefer loading for all of the 32k faces within the same parcel
    
    
        % Lateral view
        figure; plot(struc_L,left); % plot the surface
        caxis(limits); % scale the axis
        view(-90,10); % fix view angle
        colormap(cmap);
        %colormap(multigradient('preset','div.cb.spectral.10'));
        material([0.1 1 0]);
    
        c = colorbar; 
        c.Location = 'eastoutside'; % 'eastoutside' for vertical colorbar
        c.Position = [0.8 0.11 0.0381 0.815]; % Vertical Colorbar [x y width height]
        c.LineWidth = 2;
    
        %c.Location = 'southoutside'; % 'southoutside' for horizontal colorbar
        %c.Position(4) = 0.5*c.Position(4); % Horizontal Colorbar
        
        set(gca,'FontSize',24,'FontName','Arial');
        
        
        % Medial view
        figure; plot(struc_L,left);
        caxis(limits); % scale the axis
        view(90,10);
        lgt = findobj(gcf,'Type','Light');
        lgt = lgt(1);
        [~,el] = lightangle(lgt);
        lightangle(lgt,20,el);
        colormap(cmap);
        %colormap(multigradient('preset','div.cb.spectral.10'));
        material([0.1 1 0]);
        
        
        % Map the 400 schaefer parcels to the 32k FS average surface faces
        % 
        % right
        original = right.cdata; % Grab the schaefer index for each face on the FS average surface
        [~,index_net] = ismember(original,idx(:,1)); % Find the matching 400 parcels
        map_net = idx(:,2); % Set the 400 parcel loadings
        right.cdata = map_net(index_net); % Duplicate each schaefer loading for all of the 32k faces within the same parcel
    
    
        % Lateral view
        figure; plot(struc_R,right);
        caxis(limits);
        view(90,10);
        colormap(cmap);
        %colormap(multigradient('preset','div.cb.spectral.10'));
        
        lgt = findobj(gcf,'Type','Light');
        lgt = lgt(1);
        [~,el] = lightangle(lgt);
        lightangle(lgt,60,el);
        material([0 1 0]);
        %set('AmbientStrength',0.2);
        %set('SpecularStrength',0.4);
        
        
        % Medial view
        figure; plot(struc_R,right);
        caxis(limits); % scale the axis
        view(-90,10);
        lgt = findobj(gcf,'Type','Light');
        lgt = lgt(1);
        [~,el] = lightangle(lgt);
        lightangle(lgt,240,el);
        colormap(cmap);
        %colormap(multigradient('preset','div.cb.spectral.10'));
        material([0.1 1 0]);
    end  


end

