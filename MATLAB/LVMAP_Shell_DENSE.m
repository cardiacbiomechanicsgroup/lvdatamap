% Shell script for DENSE Mechanical Activation Data Mapping onto LV 
% geometry
% Thien-Khoi N. Phung (September 10, 2018)

% Dependent Functions:
%   FE_RenderPatchMesh
%   MRI_ProcessSegment
%   MRI_RVInsertion
%   MRI_Rotate2MRI
%   Cart2PSC

%% Load and Visualize QUERY GEOMETRY for Data Mapping
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

    % % Visualize Geometry
    % FE_RenderPatchMesh(LVGEOM.nXYZ,LVGEOM.eHEX,'facecolor',[1 1 1]);

%% Load DATA short axis segmentation and RV insertions
% MRI short axis view segmented in Segment (Medviso)
load('data\DENSE\SA_EPI_pinpts.mat')

% Pre-process struct of data
% Add KeptSlices
setstruct = MRI_ProcessSegment(setstruct);

% Process RV insertion points from Short Axis
setstruct = MRI_RVInsertion(setstruct);

% Rotate all data into MRI imaging coordinates
SAxyz = MRI_Rotate2MRI(setstruct);

%% Load DATA long axis pin points
% MRI long axis view segmented in Segment (Medviso)
load('data\DENSE\LA_pinpts.mat')

% Rotate PinPoints into MRI coordinates
LAxyz.pnpt =  MRI_Rotate2MRI(setstruct);

%% ROTATION of all data to cardiac coordinate system
% Pinpoints
A = LAxyz.pnpt(1,:);
B = LAxyz.pnpt(2,:);
Si = SAxyz.pnpt(3,:);

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
    
% Rotate Data
SAxyz.REpi = (SAxyz.epi(:,1:3) - repmat(origin,size(SAxyz.epi(:,1:3),1),1))*Transform';
SAxyz.Rpts = (SAxyz.pnpt(:,1:3) - repmat(origin,size(SAxyz.pnpt(:,1:3),1),1))*Transform';
LAxyz.Rpts = (LAxyz.pnpt(:,1:3) - repmat(origin,size(LAxyz.pnpt(:,1:3),1),1))*Transform';

    % % Visualize Geometry
    % FE_RenderPatchMesh(LVGEOM.nXYZ,LVGEOM.eHEX,'facecolor',[1 1 1]);
    % hold on
    % axis equal tight,xlabel('X'),ylabel('Y'),zlabel('Z')
    % title('Data in Cardiac Coordinates')
    % plot3(SAxyz.REpi(:,1),SAxyz.REpi(:,2),SAxyz.REpi(:,3),'k-')
    % plot3(SAxyz.Rpts(:,1),SAxyz.Rpts(:,2),SAxyz.Rpts(:,3),'k.','MarkerSize',20)
    % plot3(LAxyz.Rpts(:,1),LAxyz.Rpts(:,2),LAxyz.Rpts(:,3),'k.-','MarkerSize',20)

%% CONVERT Cardiac Points to Prolate coordinates
% Short Axis
[L,M,T] = Cart2PSC(SAxyz.REpi(:,1),SAxyz.REpi(:,2),SAxyz.REpi(:,3),LVGEOM.focus);
SAxyz.LMTEpi = [L M T];

%% Load and Process Mechanical Activation Data
% Load Mechanical Activation Table
% Each row is Base towards Apex
% Each col is MidSeptal-Posterior-Lateral-Anterior-Septal
load('data\DENSE\MechanicalActivation_Table.mat')

% Define DENSE angles
secs = size(DENSE,2); % Spatial Resolution of Mechanical Activation
DENSEtheta = linspace(0,2*pi,2*secs+1);
DENSEtheta = repmat(DENSEtheta(2:2:end),max(SAxyz.epi(:,4)),1);

% Interpolate the Mu coordinate for the mechanical activation data
DENSEmu = DENSEtheta;
for jz = 1:max(SAxyz.epi(:,4)) % for each slice
    sliceIDX = SAxyz.epi(:,4)==jz;
    
    DENSEmu(jz,:) = interp1(SAxyz.LMTEpi(sliceIDX,3),SAxyz.LMTEpi(sliceIDX,2),...
                            DENSEtheta(jz,:),'linear');
end

%% Interpolate data across Epicardial nodes
% Interpolate at Query Points
EPInodes = LVGEOM.nXYZ(nEPI(:),:);
[~,EPImu,EPItheta]  = Cart2PSC(EPInodes(:,1),EPInodes(:,2),EPInodes(:,3),LVGEOM.focus);

% Circumferentially Pad Data
DENSEmupad    = [DENSEmu(:,end) DENSEmu DENSEmu(:,1)];
DENSEthetapad = [DENSEtheta(:,end)-(2*pi) DENSEtheta DENSEtheta(:,1)+(2*pi)];
DENSEpad      = [DENSE(:,end) DENSE DENSE(:,1)];

% Constant Pad the data above and below
DENSEmupad = [repmat(max(DENSEmupad(1,:)),1,size(DENSEmupad,2));
              DENSEmupad;
              repmat(min(DENSEmupad(end,:)),1,size(DENSEmupad,2))];
DENSEthetapad = [DENSEthetapad(1,:); DENSEthetapad; DENSEthetapad(end,:)];
DENSEpad      = [DENSEpad(1,:); DENSEpad; DENSEpad(end,:)];

% Create scattered Interpolant function
% Interpolation- Linear
% Extrapolation- Nearest Neighbor
MechAct  = scatteredInterpolant(DENSEmupad(:),DENSEthetapad(:),DENSEpad(:),'linear','nearest');

% Interpolate!
fitMA = MechAct(EPImu,EPItheta);
elefitMA = mean(reshape(fitMA,numel(fitMA)/4,4),2);

%% Visualize Data and Query Points
S = figure('WindowStyle','docked'); hold on
xlabel('\theta'),ylabel('\mu')
title('Mechanical Activation')
colormap(flipud(colormap('parula')))
colorbar
axis tight
s2 = scatter(EPItheta,EPImu,30,fitMA,'s','filled');
s1 = scatter(DENSEtheta(:),DENSEmu(:),50,DENSE(:),'filled');
s1.LineWidth = 1.4;
s1.MarkerEdgeColor = 'k';

dataall = nan(size(LVGEOM.eHEX,1),1);
dataall(LVGEOM.eEPI) = elefitMA;
FE_RenderPatchMesh(LVGEOM.nXYZ,LVGEOM.eHEX,'data',dataall);
FE_RenderPatchMesh(LVGEOM.nXYZ,LVGEOM.eHEX(LVGEOM.eEPI,:),'data',elefitMA);
colormap(flipud(colormap('parula')))

FE_RenderPatchMesh(LVGEOM.nXYZ,LVGEOM.eHEX,'data',repmat(elefitMA,LVLCR(3),1));
colormap(flipud(colormap('parula')))


%% Epi Shell
figure;
for jz = 1:numel(LVGEOM.eEPI)
    
    p = patch('Vertices',LVGEOM.nXYZ(LVGEOM.eHEX(LVGEOM.eEPI(jz),:),1:3),'Faces',[1 2 6 5],...
              'FaceVertexCData',elefitMA(jz),'FaceColor','flat',...
              'EdgeColor',[0 0 0],...
              'LineWidth',1); hold on
end
axis equal off tight
colormap(flipud(colormap('parula')))
set(gcf,'Renderer','Painters','color','w')
view(-90,-90)