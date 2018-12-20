function imstruct = MRI_ProcessSegment(imstruct)
%MRI_ProcessSegment takes in the setstruct from segmented MRI from Segment
%(Medviso)
% imstruct = MRI_ProcessSegment(imstruct)
% This script adds field .KeptSlices
% INPUT & OUTPUT: imstruct
% 
% Thien-Khoi N. Phung (April 19, 2018)

% Add KeptSlices (referring to which slices are segmented)
imstruct.KeptSlices = 1:imstruct.ZSize;

% Adjust KeptSlices variable to eliminate untraced images so that 
% epi/endo + scar traces will correctly register with each other
% (Note: scar is only traced on images with endo & epi contours)
% counts slices with no trace
no_trace      = sum(isnan(imstruct.EndoX(:, :,:)));
delete_slices = no_trace ~= 0; % which have tracings
delete_slices = sum(delete_slices,2) == size(delete_slices,2);
imstruct.KeptSlices(:,squeeze(delete_slices)) = []; % removes those slice numbers

% If Scar Segmentation exists
if ~isempty(imstruct.Scar)
    % Add scar information if scar is traced
    if sum(sum(sum(imstruct.Scar.Manual))) <= 0 
        imstruct.Scar.Manual = imstruct.Scar.Result;
    end   

    max_scar = max(sum(sum(imstruct.Scar.Manual,1),2));
    imstruct.Scar.ScarX = zeros(max_scar,1,imstruct.ZSize);
    imstruct.Scar.ScarY = zeros(max_scar,1,imstruct.ZSize);
    for wh = 1:imstruct.ZSize
        [x,y] = find(imstruct.Scar.Manual(:,:,wh) == 1);
        imstruct.Scar.ScarX(1:length(x),1,wh) = x;
        imstruct.Scar.ScarY(1:length(y),1,wh) = y;
    end
end

end