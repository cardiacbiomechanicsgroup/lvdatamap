function H = FE_RenderPatchMesh(NODES,ECON,varargin)
%FE_RenderPatchMesh: Renders the patches version of the FE mesh from the
%cartesian NODES and element connectivity matrix ECON; it will take DATA
%which must be the same rows number as ECON 
% H = FE_RenderPatchMesh(NODES,ECON,varargin)
%   INPUTS:
%       NODES- cartesian nodes (row number is node number)
%       ECON- element connectivity matrix (FEBio linear hex specifications)
%       varargin: 'data' followed by vector with same length as ECON
%                 'color' followed MATLAB color theme (ie. 'hot')
%                 'title' followed by string for figure window title
%                 'handle' followed by figure handle to plot on top of
%                 'elements' followed by logical or indices of elements to
%                            plot
%                 'facecolor' followed by RGB for face color
%                 'alpha' followed by 0 to 1 transparency
%                 'edgecolor' followed by RGB for edge color
%                 'axes' followed by string for axis handle
%                 'edgealpha' followed by 0 to 1 transparency
%   OUTPUT:
%       H- plot handle
% 
% Created by Thien-Khoi N. Phung (February 24, 2017)
% Updated by Thien-Khoi N. Phung (September 16, 2017)- added alpha,
% edgecolor flags
% Updated by Thien-Khoi N. Phung (February 27, 2018)- added axes and 
% edgealpha flag

% Deal with VARARGIN
data_flag      = false;
color_flag     = false;
plottitle      = 'Mesh Rendering';
handle_flag    = false;
elements_flag  = false;
facecolor_flag = false;
alpha_flag     = false;
edgecolor_flag = false;
axes_flag      = false;
edgealpha_flag = false;
if ~isempty(varargin)
    for jz = 1:2:numel(varargin)
        switch varargin{jz}
            case 'data' % plot data colormap on FE model
                data_flag = true;
                DATA = varargin{jz+1};
            case 'color' % set colormap
                color_flag = true;
                coloring = varargin{jz+1};
            case 'title' % title figure window
                plottitle = varargin{jz+1};
            case 'handle' % plot on existing figure
                handle_flag = true;
                handle = varargin{jz+1};
            case 'elements'
                elements_flag = true;
                elements2plot = varargin{jz+1};
            case 'facecolor'
                facecolor_flag = true;
                coloring = varargin{jz+1};
            case 'alpha'
                alpha_flag = true;
                alphavalue = varargin{jz+1};
            case 'edgecolor'
                edgecolor_flag = true;
                edgecolor = varargin{jz+1};
            case 'axes'
                axes_flag = true;
                axes_handle = varargin{jz+1};
            case 'edgealpha'
                edgealpha_flag = true;
                edgealpha = varargin{jz+1};
            otherwise
                error('ERROR: Check your varargins.')
        end
    end
end
if data_flag && elements_flag
    if numel(DATA) ~= numel(elements2plot)
        error('ERROR: DATA input size does not match number of elements to plot')
    end
elseif data_flag
    if numel(DATA) ~= size(ECON,1)
        error('ERROR: DATA input size does not match number of elements')
    end
end

% Normalize DATA to be 0 to 1
if data_flag
    dataCOL = DATA./max(DATA);
else
    if ~facecolor_flag
        coloring = [128,205,193]./255;
    end
end

% Plot FE Mesh
if handle_flag
    figure(handle)
elseif axes_flag
    axes(axes_handle);
else
    H = figure('Name',plottitle,'NumberTitle','off','WindowStyle','docked');
end

% Cycle through each element and plot 6 patches
% Which elements to plot?
if elements_flag
    if islogical(elements2plot) % convert to indices
        plotelements = find(elements2plot==1);
    else % if it is indices
        plotelements = elements2plot;
    end
else
    plotelements = 1:size(ECON,1);
end

for jz = 1:numel(plotelements)
    wh = plotelements(jz);
    face = [1 2 3 4; 1 5 6 2; 2 6 7 3; 6 5 8 7; 3 7 8 4; 1 4 8 5];
    if data_flag % WITH DATA
        if edgecolor_flag
        p = patch('Vertices',NODES(ECON(wh,:),1:3),'Faces',face,...
              'FaceVertexCData', dataCOL(jz),'FaceColor','flat',...
              'EdgeColor',edgecolor); hold on 
        else
        p = patch('Vertices',NODES(ECON(wh,:),1:3),'Faces',face,...
              'FaceVertexCData', dataCOL(jz),'FaceColor','flat'); hold on 
        end
    else % WITHOUT DATA
        if edgecolor_flag
        p = patch('Vertices',NODES(ECON(wh,:),1:3),'Faces',face,...
              'FaceColor',coloring,'EdgeColor',edgecolor); hold on
        else
        p = patch('Vertices',NODES(ECON(wh,:),1:3),'Faces',face,...
              'FaceColor',coloring); hold on 
        end
    end
    if alpha_flag
        alpha(p,alphavalue);
    end
    if edgealpha_flag
        p.EdgeAlpha = edgealpha;
    end
end

% Format Plot
axis equal off tight

% Colormap?
if data_flag
    if color_flag
        colormap(flipud(colormap(coloring)));
    end
    cb = colorbar('Ticks',[0 0.25 0.5 0.75 1],...
                  'TickLabels',{'0',...
                  num2str(round(max(DATA))*0.25),...
                  num2str(round(max(DATA))*0.50),...
                  num2str(round(max(DATA))*0.75),...
                  num2str(round(max(DATA)))});
    set(cb,'FontSize',12);
end
set(gcf,'color','w');

% set(get(cb,'Title'),'String','Electrical Activation','FontSize',14);
view([-90 -60])