% Shell script for Data Mapping onto LV geometry
% Thien-Khoi N. Phung (April 19, 2018)

% Dependent Functions:
%   FE_RenderPatchMesh
%   MRI_ProcessSegment
%   MRI_RVInsertion
%   MRI_Rotate2MRI
%   Cart2PSC

%% Load and Visualize SAMPLE GEOMETRY for Data Mapping
% LVGEOM.nXYZ     nodes in Cardiac Coordinates (cartesian)
%       .eHEX     element connectivity matrix for tetrahedral elements
%       .ePENT    element connectivity matrix for pentahedral apex elements
%       .eEPI     element numbers for epicardial elements
%       .focus    focus for prolate spheroidal coordinate system
%       .LVLCR    number of longitudinal, circumferential, and radial
%                 elements
load('data\LVGEOM_8x4_noshift.mat')

% EPI node numbers
nEPI = LVGEOM.eHEX(LVGEOM.eEPI,[3 4 7 8]);

% Visualize Geometry
FE_RenderPatchMesh(LVGEOM.nXYZ,LVGEOM.eHEX,'facecolor',[1 1 1]);

%% DATA short axis segmentation with scar
% MRI short axis view segmented in Segment (Medviso)
load('data\LGE\SA_LGE_Scar_Pnpts.mat')

% Pre-process struct of data
% Add KeptSlices and parse Scar information 
setstruct = MRI_ProcessSegment(setstruct);

% Process RV insertion points from Short Axis
setstruct = MRI_RVInsertion(setstruct);

% Rotate all data into MRI imaging coordinates
SAxyz = MRI_Rotate2MRI(setstruct);

% for jz = setstruct.KeptSlices
%     slice_ind = SAxyz.scar(:,end)==jz;
%     slice = SAxyz.scar(slice_ind,:);
%     xmode = mode(slice(:,1));
%     mode_idx = SAxyz.scar(:,1) == xmode;
%     SAxyz.scar(mode_idx,:) = [];
% end

    H = figure('WindowStyle','docked'); hold on
    title('Data in Imaging Coordinates')
    plot3(SAxyz.endo(:,1),SAxyz.endo(:,2),SAxyz.endo(:,3),'r-')
    plot3(SAxyz.epi(:,1),SAxyz.epi(:,2),SAxyz.epi(:,3),'k-')
    plot3(SAxyz.scar(:,1),SAxyz.scar(:,2),SAxyz.scar(:,3),'m.','MarkerSize',20)
    plot3(SAxyz.pnpt(:,1),SAxyz.pnpt(:,2),SAxyz.pnpt(:,3),'k.','MarkerSize',15)
    axis equal
    view(3)
    
% slices with scar
slices = unique(SAxyz.scar(:,end))';
% keep only slices with scar
for jz = 1:numel(slices)
    slice = slices(jz);
    SAxis{jz}.Endo = SAxyz.endo(SAxyz.endo(:,end)==slice,1:3);
    SAxis{jz}.Epi = SAxyz.epi(SAxyz.epi(:,end)==slice,1:3);
    SAxis{jz}.Scar = SAxyz.scar(SAxyz.scar(:,end)==slice,1:3);
end
SAxis{jz+1}.pts  = SAxyz.pnpt;

%% DATA long axis segmentation with scar
LAfiles = {'data\LGE\LA_LGE_2CH_Scar.mat';
           'data\LGE\LA_LGE_3CH_Scar.mat';
           'data\LGE\LA_LGE_4CH_Scar.mat'};
for jz = 1:numel(LAfiles)

    load(LAfiles{jz})

    % Pre-process struct of data
    % Add KeptSlices and parse Scar information 
    setstruct = MRI_ProcessSegment(setstruct);
    
    % Rotate all data into MRI imaging coordinates
    LAxyz = MRI_Rotate2MRI(setstruct);
    
    plot3(LAxyz.endo(:,1),LAxyz.endo(:,2),LAxyz.endo(:,3),'r-')
    plot3(LAxyz.epi(:,1),LAxyz.epi(:,2),LAxyz.epi(:,3),'k-')
    plot3(LAxyz.scar(:,1),LAxyz.scar(:,2),LAxyz.scar(:,3),'m.','MarkerSize',20)

    LAxis{jz}.Endo = LAxyz.endo;
    LAxis{jz}.Epi  = LAxyz.epi;
    LAxis{jz}.Scar = LAxyz.scar;
end

% Long Axis pin points into Imaging Coordinates
LAfilepnpt = 'data\LGE\LA_LGE_2CH_Pnpts.mat';
load(LAfilepnpt)
LAxis{jz+1}.pts = MRI_Rotate2MRI(setstruct);

plot3(LAxis{end}.pts(:,1),LAxis{end}.pts(:,2),LAxis{end}.pts(:,3),'k.-','MarkerSize',15)
axis tight

%% ROTATION of all data to cardiac coordinate system
% Pinpoints
A = LAxis{end}.pts(1,:);
B = LAxis{end}.pts(2,:);
Si = SAxis{end}.pts(3,:);

    % First basis vector, subtract Base from Apex and divide by magnitude
    C = A-B;
    e1 = C ./ norm(C);

    % Calculate Origin location
    origin = B + (1/3)*C;
 
    % Second basis vector using NEW METHOD- plane intersects septal point &
    % e1
    D = Si(1,:) - origin;
    D2 = D - dot(D,e1)*e1;
    e2 = D2 ./ norm(D2);

    % Third basis vector
    E = cross(e1,e2);
    e3 = E ./ norm(E);

    % Transformation matrix
    Transform = [e1; e2; e3];
    
G = figure('WindowStyle','docked'); hold on
axis equal tight,xlabel('X'),ylabel('Y'),zlabel('Z')
title('Data in Cardiac Coordinates')

for jz = 1:numel(SAxis)-1
    SAxis{jz}.REndo(:,1:3) = (SAxis{jz}.Endo(:,1:3) - origin)*Transform';
    SAxis{jz}.REpi(:,1:3)  = (SAxis{jz}.Epi(:,1:3) - origin)*Transform';
    SAxis{jz}.RScar(:,1:3) = (SAxis{jz}.Scar(:,1:3) - origin)*Transform';
    
    plot3(SAxis{jz}.REndo(:,1),SAxis{jz}.REndo(:,2),SAxis{jz}.REndo(:,3),'r-')
    plot3(SAxis{jz}.REpi(:,1),SAxis{jz}.REpi(:,2),SAxis{jz}.REpi(:,3),'k-')
    plot3(SAxis{jz}.RScar(:,1),SAxis{jz}.RScar(:,2),SAxis{jz}.RScar(:,3),'m.','MarkerSize',20)
end
SAxis{jz+1}.Rpts(:,1:3)  = (SAxis{jz+1}.pts(:,1:3) - origin)*Transform';
plot3(SAxis{jz+1}.Rpts(:,1),SAxis{jz+1}.Rpts(:,2),SAxis{jz+1}.Rpts(:,3),'k.','MarkerSize',20)

for jz = 1:numel(LAxis)-1
    LAxis{jz}.REndo(:,1:3) = (LAxis{jz}.Endo(:,1:3) - origin)*Transform';
    LAxis{jz}.REpi(:,1:3)  = (LAxis{jz}.Epi(:,1:3) - origin)*Transform';
    LAxis{jz}.RScar(:,1:3) = (LAxis{jz}.Scar(:,1:3) - origin)*Transform';
    
    plot3(LAxis{jz}.REndo(:,1),LAxis{jz}.REndo(:,2),LAxis{jz}.REndo(:,3),'r-')
    plot3(LAxis{jz}.REpi(:,1),LAxis{jz}.REpi(:,2),LAxis{jz}.REpi(:,3),'k-')
    plot3(LAxis{jz}.RScar(:,1),LAxis{jz}.RScar(:,2),LAxis{jz}.RScar(:,3),'m.','MarkerSize',20)
end
LAxis{jz+1}.Rpts(:,1:3)  = (LAxis{jz+1}.pts(:,1:3) - origin)*Transform';
plot3(LAxis{jz+1}.Rpts(:,1),LAxis{jz+1}.Rpts(:,2),LAxis{jz+1}.Rpts(:,3),'k.-','MarkerSize',20)

%% CONVERT Cardiac Data to Prolate coordinates
% Short Axis
for jz = 1:numel(SAxis)-1
    [L,M,T] = Cart2PSC(SAxis{jz}.REndo(:,1),SAxis{jz}.REndo(:,2),SAxis{jz}.REndo(:,3),LVGEOM.focus);
    SAxis{jz}.LMTEndo = [L M T];
    [L,M,T] = Cart2PSC(SAxis{jz}.REpi(:,1),SAxis{jz}.REpi(:,2),SAxis{jz}.REpi(:,3),LVGEOM.focus);
    SAxis{jz}.LMTEpi = [L M T];
    [L,M,T] = Cart2PSC(SAxis{jz}.RScar(:,1),SAxis{jz}.RScar(:,2),SAxis{jz}.RScar(:,3),LVGEOM.focus);
    SAxis{jz}.LMTScar = [L M T];
end

% Long Axis
for jz = 1:numel(LAxis)-1
    [L,M,T] = Cart2PSC(LAxis{jz}.REndo(:,1),LAxis{jz}.REndo(:,2),LAxis{jz}.REndo(:,3),LVGEOM.focus);
    LAxis{jz}.LMTEndo = [L M T];
    [L,M,T] = Cart2PSC(LAxis{jz}.REpi(:,1),LAxis{jz}.REpi(:,2),LAxis{jz}.REpi(:,3),LVGEOM.focus);
    LAxis{jz}.LMTEpi = [L M T];
    [L,M,T] = Cart2PSC(LAxis{jz}.RScar(:,1),LAxis{jz}.RScar(:,2),LAxis{jz}.RScar(:,3),LVGEOM.focus);
    LAxis{jz}.LMTScar = [L M T];
end

% SAMPLE Geometry
[qL,qM,qT] = Cart2PSC(LVGEOM.nXYZ(:,1),LVGEOM.nXYZ(:,2),LVGEOM.nXYZ(:,3),LVGEOM.focus);

%% Assign data to epicardium using Prolate coordinates (SHORT AXES)
T = figure('WindowStyle','docked'); hold on
xlabel('\theta'),ylabel('\mu')
title('Data Locations on Epi')
colormap(flipud(colormap('parula')))
colorbar
% Plot SAMPLE points
plot(qT(nEPI(:)),qM(nEPI(:)),'ko','MarkerSize',6)
cardiacdata = [];
for jz = 1:numel(SAxis)-1
    % Short Axis use THETA bins
    numbins  = 50;
    angedges = linspace(0,2*pi,numbins+1);
    angcents = (angedges(1:end-1) + angedges(2:end))./2;
    
    % Interpolate LAMBDA and MU for EPI
    LM = interp1(SAxis{jz}.LMTEpi(:,3),SAxis{jz}.LMTEpi(:,1:2),angcents,'linear');
    
    % Group data into THETA bins
    [enN,~,enBIN] = histcounts(SAxis{jz}.LMTEndo(:,3),angedges);
    [epN,~,epBIN] = histcounts(SAxis{jz}.LMTEpi(:,3),angedges);
    [scN,~,scBIN] = histcounts(SAxis{jz}.LMTScar(:,3),angedges);
    
    % Wall Thickness | Scar Transmurality | Scar Depth from EPI
    WTHSCAR = nan(numbins,3);
    for wh = 1:numbins
        % Average Endo and Epi points in Cartesian Coordinates
        endopoint = mean(SAxis{jz}.REndo(enBIN==wh,:),1);
        epipoint  = mean(SAxis{jz}.REpi(epBIN==wh,:),1);
        
        % Store Wall Thickness
        WTHSCAR(wh,1) = sqrt(sum((epipoint-endopoint).^2));
        
        % Scar Points (only if more than 2 pixels scar)
        if sum(scBIN==wh)>2
            % Scar Points Cartesian and Prolate Sph Values
            scarpoints = [SAxis{jz}.RScar(scBIN==wh,:) SAxis{jz}.LMTScar(scBIN==wh,:)];
            
            % endo and epi most scar points determined by LAMDA
            scarpoints = sortrows(scarpoints,4);
            endosp = scarpoints(1,1:3);
            episp  = scarpoints(end,1:3);
            
            % Scar transmurality
            WTHSCAR(wh,2) = sqrt(sum((episp-endosp).^2))/WTHSCAR(wh,1);
            
            % Scar depth as percent from epi depth
            WTHSCAR(wh,3) = sqrt(sum((epipoint-episp).^2))/WTHSCAR(wh,1);
        end
    end
    
    % Store Information
    % [L M T WTH SCARTRANS SCARDEPTH]
    SAxis{jz}.wthscar = [LM angcents(:) WTHSCAR];
    
    data = [LM angcents(:) WTHSCAR];
    data(isnan(data(:,5)),5) = 0;
    cardiacdata = [cardiacdata; data];
    
    % Plot SA data points
    scatter(SAxis{jz}.wthscar(:,3),SAxis{jz}.wthscar(:,2),100,SAxis{jz}.wthscar(:,5),'filled')
end

%% Assign data to epicardium using Prolate coordinates (LONG AXES)
for jz = 1:numel(LAxis)-1
    % Long Axis use MU bins
    numbins  = 20;
    angedges = linspace(0,120*(pi/180),numbins+1);
    angcents = (angedges(1:end-1) + angedges(2:end))./2;
    
    % Group LA data in two theta bins (since there are two sets of lambdas)
    mdpt = mean(LAxis{jz}.LMTEpi(:,3));
    edges = [mdpt-pi mdpt mdpt+pi];
    [~,~,Hepi]  = histcounts(LAxis{jz}.LMTEpi(:,3),edges);
    [~,~,Hendo] = histcounts(LAxis{jz}.LMTEndo(:,3),edges);
    [~,~,Hscar] = histcounts(LAxis{jz}.LMTScar(:,3),edges);
    
    % For each Hemisphere
    data = [];
    for by = 1:2
        LMTendo = LAxis{jz}.LMTEndo(Hendo==by,:);
        LMTepi  = LAxis{jz}.LMTEpi(Hepi==by,:);
        LMTscar = LAxis{jz}.LMTScar(Hscar==by,:);
        Rendo = LAxis{jz}.REndo(Hendo==by,:);
        Repi  = LAxis{jz}.REpi(Hepi==by,:);
        Rscar = LAxis{jz}.RScar(Hscar==by,:);
        
        % Interpolate LAMBDA and THETA for EPI
        LT = interp1(LMTepi(:,2),LMTepi(:,[1 3]),angcents,'linear','extrap');

        % Group data into MU bins
        [enN,~,enBIN] = histcounts(LMTendo(:,2),angedges);
        [epN,~,epBIN] = histcounts(LMTepi(:,2),angedges);
        [scN,~,scBIN] = histcounts(LMTscar(:,2),angedges);
    
        % Wall Thickness | Scar Transmurality | Scar Depth from EPI
        WTHSCAR = nan(numbins,3);
        for wh = 1:numbins
            % Average Endo and Epi points in Cartesian Coordinates
            endopoint = mean(Rendo(enBIN==wh,:),1);
            epipoint  = mean(Repi(epBIN==wh,:),1);

            % Store Wall Thickness
            WTHSCAR(wh,1) = sqrt(sum((epipoint-endopoint).^2));

            % Scar Points (only if more than 2 pixels scar)
            if sum(scBIN==wh)>2
                % Scar Points Cartesian and Prolate Sph Values
                scarpoints = [Rscar(scBIN==wh,:) LMTscar(scBIN==wh,:)];

                % endo and epi most scar points determined by LAMDA
                scarpoints = sortrows(scarpoints,4);
                endosp = scarpoints(1,1:3);
                episp  = scarpoints(end,1:3);

                % Scar transmurality
                WTHSCAR(wh,2) = sqrt(sum((episp-endosp).^2))/WTHSCAR(wh,1);

                % Scar depth as percent from epi depth
                WTHSCAR(wh,3) = sqrt(sum((epipoint-episp).^2))/WTHSCAR(wh,1);
            end
        end
        % Store Information
        % [L M T WTH SCARTRANS SCARDEPTH]
        data = [data; LT(:,1) angcents(:) LT(:,2) WTHSCAR];
    end
    LAxis{jz}.wthscar = data;
    data(isnan(data(:,5)),5) = 0;
    cardiacdata = [cardiacdata; data(~isnan(data(:,5)),:)];
    
    % Plot LA data points
    figure(T); hold on
    scatter(LAxis{jz}.wthscar(:,3),LAxis{jz}.wthscar(:,2),100,LAxis{jz}.wthscar(:,5),'filled')
end

%% INTERPOLATE Scar Transmurality
% Pad Cardiac Data Circumferentially
interpdata = cardiacdata(~isnan(cardiacdata(:,2)),[3 2 5 6]); % Theta Mu Transmurality
interpdata = [interpdata;
              interpdata(:,1)+2*pi interpdata(:,2:end);
              interpdata(:,1)-2*pi interpdata(:,2:end)];

% Interpolant Function
transmurality = scatteredInterpolant(interpdata(:,1),interpdata(:,2),interpdata(:,3),'linear');
nonnan = ~isnan(interpdata(:,4));
depth  = scatteredInterpolant(interpdata(nonnan,1),interpdata(nonnan,2),interpdata(nonnan,4),'linear');

% Interpolate at SAMPLE Points
EPInodes = LVGEOM.nXYZ(nEPI(:),:);
[~,MEPIn,TEPIn]  = Cart2PSC(EPInodes(:,1),EPInodes(:,2),EPInodes(:,3),LVGEOM.focus);

% Interpolate
scartr = transmurality(TEPIn,MEPIn);
scardp = depth(TEPIn,MEPIn);

S = figure('WindowStyle','docked'); hold on
xlabel('\theta'),ylabel('\mu')
title('Interpolated Transmurality')
colormap(flipud(colormap('parula')))
colorbar
s2 = scatter(TEPIn,MEPIn,30,scartr,'s','filled');
s1 = scatter(cardiacdata(:,3),cardiacdata(:,2),50,cardiacdata(:,5),'filled');
s1.LineWidth = 1.4;
s1.MarkerEdgeColor = 'k';
axis tight

S = figure('WindowStyle','docked'); hold on
xlabel('\theta'),ylabel('\mu')
title('Interpolated Depth')
colormap(flipud(colormap('parula')))
colorbar
scatter(cardiacdata(:,3),cardiacdata(:,2),50,cardiacdata(:,6),'filled')
scatter(TEPIn,MEPIn,40,scardp)
axis tight


%% VISUALIZE THE SCAR
% Manipulate to fit to Element data (data was interopolated to nodes)
% Each row is an element (corresponding to eEPI)
% Each column is the nodes making up that elements Epicardial surface
scartrans = reshape(scartr,numel(scartr)/4,4);
scardepth = reshape(scardp,numel(scartr)/4,4);

% Scar depth from Epi has no meaning when there is zero scar
% So we set any scartrans<=0 nodes to have nan scardepth
scardepth(scartrans<=0) = nan;

% Taking the mean of each row consolidates the nodal data to an individual
% element data value.
eEPItr = mean(scartrans,2);
eEPIwd = nanmean(scardepth,2); % mean that ignores nans

% Data projected on epicardium
dataall = zeros(size(LVGEOM.eHEX,1),1);
dataall(LVGEOM.eEPI) = eEPItr;
FE_RenderPatchMesh(LVGEOM.nXYZ,LVGEOM.eHEX,'data',dataall);
colormap(flipud(colormap('parula')))

% How many elements deep is scar?
SCARlayers = round(eEPItr * LVGEOM.LVLCR(3));

% What layer does scar start (from epicardium)?
EPIstart = round(eEPIwd*LVGEOM.LVLCR(3)) + 1;
%       # elems not scar from epi + 1 <- for where scar layer starts

% if scar is too thin (0 elements deep)- remove EPIstart value
EPIstart(SCARlayers==0) = nan;

% if SCARlayers and EPIstart don't add up to LVGEOM.LVLCR(3)+1 due to round error:
% Adjust SCARlayers
SCARlayers((SCARlayers+EPIstart)>(LVGEOM.LVLCR(3)+1)) = ...
    SCARlayers((SCARlayers+EPIstart)>(LVGEOM.LVLCR(3)+1)) - 1;

% scar mask
epl = LVGEOM.LVLCR(1)*LVGEOM.LVLCR(2); % elements per layer

eSCAR = [];
for jz = 1:max(EPIstart)  
    outerscar = LVGEOM.eEPI(EPIstart==jz) - epl*(jz-1);
    thruwallscar = SCARlayers(EPIstart==jz);
    
    for wh = 1:numel(outerscar)
        for by = 1:thruwallscar(wh)
            eSCAR = [eSCAR; outerscar(wh)- epl*(by-1)];
        end 
    end
end

% Plot skeleton mesh
H = FE_RenderPatchMesh(LVGEOM.nXYZ,LVGEOM.eHEX,'facecolor',[229,245,224]./256,...
    'edgecolor',[0.8 0.8 0.8],'alpha',0.1);

FE_RenderPatchMesh(LVGEOM.nXYZ,LVGEOM.eHEX,'elements',eSCAR,...
    'facecolor',[152,78,163]./255,'handle',H)

