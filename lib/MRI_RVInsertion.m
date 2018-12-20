function imstruct = MRI_RVInsertion(imstruct)
%MRI_RVInsertiont takes in the setstruct from segmented MRI from Segment
%(Medviso)
% imstruct = MRI_RVInsertion(imstruct)
% This script adds RV insertion points and septal point
% INPUT & OUTPUT: imstruct
% 
% Thien-Khoi N. Phung (April 19, 2018)

% Assume time point of pin points
timeID = 1;
    
% Identify RV insertion slice
for jz = imstruct.KeptSlices
    convert = cell2mat(imstruct.EndoPinX(:,jz));
    contents(jz) = sum(convert);
end
septal_slice = find(contents);

% Find endocardial mid-septal point
% Find midpoint of the line connecting the RV insertion points
RVInsertionPts = [imstruct.EndoPinX{timeID,septal_slice},imstruct.EndoPinY{timeID,septal_slice}];
MidPt = mean(RVInsertionPts);

% Find slope of line perpendicular to line connecting RV insertion points
slope = (RVInsertionPts(2,2)-RVInsertionPts(1,2))/(RVInsertionPts(2,1)-RVInsertionPts(1,1));
perpslope = -1/slope;

% Specifically for DENSE case where we do not need to segment Endocardium
% for mapping the mechanical activation data
if isempty(imstruct.EndoX)
    imstruct.EndoX = imstruct.EpiX;
    imstruct.EndoY = imstruct.EpiY;
end

% Find intersection of endocardium with perpendicular line projected
% from midpoint that is on the septum (automated by TNP)
% Shift slice to MidPt as the center of endocardium
    slice = [imstruct.EndoX(:,timeID,septal_slice),imstruct.EndoY(:,timeID,septal_slice)];
    slice = slice-repmat(MidPt,size(slice,1),1);
% Convert to polar coordinates
    [TH,R] = cart2pol(slice(:,1),slice(:,2));
    TH(TH<0) = TH(TH<0) + 2*pi;
    TH = TH(2:end); % beginning and end points are the same
    R = R(2:end);
% Find theta for perpendicular slope (perpslope)
    dotprod =  dot([1 1*perpslope],[1 0]);
    th1 = acos(dotprod/norm([1 1*perpslope]));
    th2 = th1 + pi; % 180 degree rotation
% Interpolate the radii for the two angles
    r1 = interp1(TH,R,th1,'linear');
    r2 = interp1(TH,R,th2,'linear');
% Choose the smaller radius to designate the septal point
    r = r1*(r1<r2) + r2*(r2<r1);
    th = th1*(r1<r2) + th2*(r2<r1);
% Calculate mid-septal endocardial coordinate and shift back to original
% origin
    [septx,septy] = pol2cart(th,r);
    septx = septx + MidPt(1);
    septy = septy + MidPt(2);
% Store those points in the originals etstruct
imstruct.EndoPinX{timeID,septal_slice} = [imstruct.EndoPinX{timeID,septal_slice};septx];
imstruct.EndoPinY{timeID,septal_slice} = [imstruct.EndoPinY{timeID,septal_slice};septy];

end