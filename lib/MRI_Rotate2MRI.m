function [XYZ] = MRI_Rotate2MRI(imgstruct)
%MRI_Rotate2MRI: Rotates any segmented information into MRI imaging
%coordinates
% [XYZ] = MRI_Rotate2MRI(imgstruct)
%   INPUT VARIABLE:
%       imgstruct- structure of LV segmentation from Segment (Medviso)
%   OUTPUT VARIABLE:
%       XYZ- struct with different fields of cartesian coordinates of endo
%            [x y z timeindx sliceindx]
% 
% Thien-Khoi N. Phung (April 20, 2018)
% Updated by TNP (September 6. 2018)

slice_counter = 1;
if isfield(imgstruct,'KeptSlices')
    slice_labels = imgstruct.KeptSlices;
else
    slice_labels = 1;
end

if length(slice_labels) > 1 % For multiple a view with multiple slices
    ENDO = [];
    EPI  = [];
    SCAR = [];
    for jdx=slice_labels    % Iterate through each slice
        % Create Transformation Matrix
        M = TransformMatrix(imgstruct);
        
        % ENDO
        [Xs,Ps] = TransformEndo(imgstruct,M,jdx);
        Xd = cell2mat(Xs);
        Pd = Ps;
        if all(isnan(Xd(:))), continue, end
        slice_indicies = ones(size(Xd,1),1) * jdx;
        ENDO = [ENDO; ...
                [Xd, slice_indicies] ];
        
        % EPI
        [Xs] = TransformEpi(imgstruct,M,jdx);
        Xd = cell2mat(Xs);
        if all(isnan(Xd(:))), continue, end
        slice_indicies = ones(size(Xd,1),1) * jdx;
        EPI = [EPI; ...
                [Xd, slice_indicies] ];
        
        % SCAR
        if ~isempty(imgstruct.Scar)
            [Xs] = TransformScar(imgstruct,M,jdx);
            Xd = cell2mat(Xs);
            if all(isnan(Xd(:))), continue, end
            slice_indicies = ones(size(Xd,1),1) * jdx;
            SCAR = [SCAR; ...
                    [Xd, slice_indicies] ];
        end
            
        slice_counter = slice_counter + 1;
    end
    XYZ.endo = ENDO;
    XYZ.pnpt = Pd;
    XYZ.epi  = EPI;
    XYZ.scar = SCAR;
    
else % For a view with one slice
    M = TransformMatrix(imgstruct);
    [Xs,Ps] = TransformEndo(imgstruct,M);
    Xd = cell2mat(Xs);
    Pd = Ps;
    % This exit route is for Long Axis Pin Points file with no segmentation
        if isempty(Xd)
            XYZ = Pd;
            return
        end    
    slice_indicies = ones(size(Xd,1),1);
    XYZ.endo = [Xd, slice_indicies];
    
    [Xs] = TransformEpi(imgstruct,M);
    Xd = cell2mat(Xs);
    slice_indicies = ones(size(Xd,1),1);
    XYZ.epi = [Xd, slice_indicies];
    
    if ~isempty(imgstruct.Scar)
        [Xs] = TransformScar(imgstruct,M);
        Xd = cell2mat(Xs);
        slice_indicies = ones(size(Xd,1),1);
        XYZ.scar = [Xd, slice_indicies];
    end
end

function M = TransformMatrix(imgstruct)
    % Creates 2D to 3D image coordinate transformation matrix
    % Pull image resolution
    x_resolution = imgstruct.ResolutionX;
    y_resolution = imgstruct.ResolutionY;
    % Pull image position
    image_position = imgstruct.ImagePosition;
    % Pull image orientation vectors
    x_image_orientation = imgstruct.ImageOrientation(4:6);
    y_image_orientation = imgstruct.ImageOrientation(1:3);
    z_image_orientation = cross(y_image_orientation,x_image_orientation);
    % Pull slice offset values
    slice_thickness = imgstruct.SliceThickness;
    slice_gap = imgstruct.SliceGap;
    % Create Transformation Matrix
    To = eye(4);
    To(1:3,4) = [-1 -1 0]';
    S = eye(4);
    S(1,1) = x_resolution;
    S(2,2) = y_resolution;
    S(3,3) = slice_thickness+slice_gap; 
    R = eye(4);
    R(1:3,1:3) = [x_image_orientation(:), y_image_orientation(:), z_image_orientation(:)];
    Tipp = eye(4);
    Tipp(1:3,4) = image_position';
    M = Tipp * R * S * To;
end

function [xyz_pts,varargout] = TransformEndo(imgstruct,M,varargin)
    % imgstruct- structure of LV segmentation from Segment (Medviso)
    % varargin- slice number
    % Pick Slice Number
    if ~isempty(varargin)
        slicenumber = varargin{1};
        % Assume slice timepoint 1
        tmpt = 1;
        x_pix = imgstruct.EndoX(:,tmpt,slicenumber);
        y_pix = imgstruct.EndoY(:,tmpt,slicenumber);
    else
        slicenumber = 1;
        x_pix = imgstruct.EndoX;
        y_pix = imgstruct.EndoY;
    end
    
    % Round pixel values
    x_pix_round = round(x_pix);
    y_pix_round = round(y_pix);
    perim_length = zeros(size(x_pix_round,2),1);
    xy_pts = cell(size(x_pix_round,2),1);

    for idx=1:size(x_pix_round,2)
        if any(isnan(x_pix_round(:,idx)))
            xy_pts{idx} = [NaN NaN NaN];
            continue
        end

        xy_pix_round = [x_pix_round(:,idx), y_pix_round(:,idx)];
        perim_length(idx) = size(unique(xy_pix_round,'rows'),1);

        perim_pts = linspace(0,1,size(x_pix,1)+1);
        interp_perim_pts = linspace(0,1,perim_length(idx)+1);
        perim_xy_pts = [[x_pix(:,idx), y_pix(:,idx)]; [x_pix(1,idx), y_pix(1,idx)]];
        interp_xy_pts = interp1(perim_pts,perim_xy_pts,interp_perim_pts,'spline');

        xy_pts{idx} = interp_xy_pts(1:end-1,:);
    end

    z_offset = (slicenumber - 1);
    % Create Z values
    xyz_pts = xy_pts;
    for idx=1:size(x_pix_round,2)
        z_pix = -z_offset * ones(perim_length(idx),1);
        if ~any(isnan(xy_pts{idx}(:)))
            xyz_pts{idx} = [xy_pts{idx}, z_pix];
        end
    end
    
    % Rotate points
    for idx=1:size(x_pix_round,2)
        if ~any(isnan(xyz_pts{idx}(:)))
            X = ( M * [xyz_pts{idx}, ones(size(xyz_pts{idx},1),1)]')';
            X = X(:,1:3);
            xyz_pts{idx} = X;
        end
    end

    [timeID,PPSlice] = find(~cellfun(@isempty,imgstruct.EndoPinX));

    z_offset_PP = (PPSlice - 1);
    x_pinpts = imgstruct.EndoPinX{timeID,PPSlice};
    y_pinpts = imgstruct.EndoPinY{timeID,PPSlice};
    pinpts_round = [round(x_pinpts),round(y_pinpts)];
    pinpts = {pinpts_round};
    z_pix = -z_offset_PP * ones(length(x_pinpts),1);
    pinpts3d{1} = [pinpts{1}, z_pix];

    PP = ( M * [pinpts3d{1}, ones(size(pinpts3d{1},1),1)]')';
    PP = PP(:,1:3);
    varargout{1} = PP;
end

function [xyz_pts] = TransformEpi(imgstruct,M,varargin)
    if ~isempty(varargin)
        slicenumber = varargin{1};
        % Assume slice timepoint 1
        tmpt = 1;
        x_pix = imgstruct.EpiX(:,tmpt,slicenumber);
        y_pix = imgstruct.EpiY(:,tmpt,slicenumber);
    else
        slicenumber = 1;
        tmpt = 1;
        x_pix = imgstruct.EpiX(:,tmpt,slicenumber);
        y_pix = imgstruct.EpiY(:,tmpt,slicenumber);
    end

    x_pix_round = round(x_pix);
    y_pix_round = round(y_pix);
    perim_length = zeros(size(x_pix_round,2),1);
    xy_pts = cell(size(x_pix_round,2),1);

    for idx=1:size(x_pix_round,2)

        if any(isnan(x_pix_round(:,idx)))
            xy_pts{idx} = [NaN NaN NaN];
            continue
        end

        xy_pix_round = [x_pix_round(:,idx), y_pix_round(:,idx)];
        perim_length(idx) = size(unique(xy_pix_round,'rows'),1);

        perim_pts = linspace(0,1,size(x_pix,1)+1);
        interp_perim_pts = linspace(0,1,perim_length(idx)+1);
        perim_xy_pts = [[x_pix(:,idx), y_pix(:,idx)]; [x_pix(1,idx), y_pix(1,idx)]];
        interp_xy_pts = interp1(perim_pts,perim_xy_pts,interp_perim_pts,'spline');

        xy_pts{idx} = interp_xy_pts(1:end-1,:);

    end


    z_offset = (slicenumber - 1);
    xyz_pts = xy_pts;
    for idx=1:size(x_pix_round,2)
        z_pix = -z_offset * ones(perim_length(idx),1);
        if ~any(isnan(xy_pts{idx}(:)))
            xyz_pts{idx} = [xy_pts{idx}, z_pix]; 
        end
    end

    for idx=1:size(x_pix_round,2)
        if ~any(isnan(xyz_pts{idx}(:)))
            X = ( M * [xyz_pts{idx}, ones(size(xyz_pts{idx},1),1)]')';
            X = X(:,1:3);
            xyz_pts{idx} = X;
        end
    end

end

function [xyz_pts] = TransformScar(imgstruct,M,varargin)
    if ~isempty(varargin)
        slicenumber = varargin{1};
        x_pix = imgstruct.Scar.ScarX(:,1,slicenumber);
        y_pix = imgstruct.Scar.ScarY(:,1,slicenumber);
    else
        slicenumber = 1;
        x_pix = imgstruct.Scar.ScarX(:,1,slicenumber);
        y_pix = imgstruct.Scar.ScarY(:,1,slicenumber);
    end

    z_offset = (slicenumber - 1);

    if isnan(sum(sum(imgstruct.Scar.ScarX(:,:,slicenumber)))) < 1
        x_pix_round = round(x_pix);
        y_pix_round = round(y_pix);

        perim_length = zeros(size(x_pix_round,2),1);
        xy_pts = cell(size(x_pix_round,2),1);

        for idx=1:size(x_pix_round,2)
            xy_pix_round = [x_pix_round(:,idx), y_pix_round(:,idx)];
            perim_length(idx) = size(xy_pix_round,1);

            perim_pts = linspace(0,1,size(x_pix,1)+1);
            interp_perim_pts = linspace(0,1,perim_length(idx)+1);
            perim_xy_pts = [[x_pix(:,idx), y_pix(:,idx)]; [x_pix(1,idx), y_pix(1,idx)]];
            interp_xy_pts = interp1(perim_pts,perim_xy_pts,interp_perim_pts,'spline');

            xy_pts{idx} = interp_xy_pts(1:end-1,:);
        end

        xyz_pts = xy_pts;
        for idx=1:size(x_pix_round,2)
            z_pix = -z_offset * ones(perim_length(idx),1);
            xyz_pts{idx} = [xy_pts{idx}, z_pix];
        end

        for idx=1:size(x_pix_round,2)
            if ~any(isnan(xyz_pts{idx}(:)))
                X = ( M * [xyz_pts{idx}, ones(size(xyz_pts{idx},1),1)]')';
                X = X(:,1:3);
                xyz_pts{idx} = X;
            end
        end
        
        xyz_pts{idx}(x_pix_round==0,:) = [];
        
    else
        xyz_pts{1} = [NaN,NaN,NaN];
    end
end

end % End Function