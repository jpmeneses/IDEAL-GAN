%% RUN EACH BLOCK USING CTRL+ENTER
% FIRST BLOCK: LOAD MAT FILE
close all, clearvars, clc
addpath(genpath([pwd '/matlab']));
[file,location] = uigetfile;
%% SECOND BLOCK: DISPLAY SELECTED SLICE AND DRAW ROI
num_slice = 14; % YOU CAN CHANGE THIS
sel_map = 'ECH-1'; % OPTIONS: PDFF, R2s, ECH-1, ECH-2 - YOU CAN CHANGE THIS
if strcmp(sel_map,'PDFF')
    load([location,file])
    outmap = F;
    val_range = [0,100];
elseif strcmp(sel_map,'R2s')
    load([location,file])
    outmap = R2;
    val_range = [0,200];
elseif strcmp(sel_map,'ECH-1')
    [im,filename,pathname] = open_dicom2(location, file);
    S = cse_dicom_processing(im,filename,pathname);
    outmap = abs(S(:,:,:,1,1));
    val_range = [0,max(outmap,[],'all')];
elseif strcmp(sel_map,'ECH-2')
    [im,filename,pathname] = open_dicom2(location, file);
    S = cse_dicom_processing(im,filename,pathname);
    outmap = abs(S(:,:,:,1,1));
    val_range = [0,max(outmap,[],'all')];
end
im_slice = outmap(:,:,num_slice);
figure(1)
imshow(im_slice,val_range)
roi =drawcircle('InteractionsAllowed','translate','LineWidth',1,...
                'Center',[132,232],'Radius',sqrt(200/pi));
%% THIRD BLOCK: RUN TO REPEAT ROI MEASUREMENTS
clc
mask = createMask(roi);
if strcmp(sel_map,'PDFF')
    ROI_val = median(im_slice(mask));
else
    ROI_val = mean(im_slice(mask));
end
ROI_std = std(im_slice(mask));
roi.Label = num2str(round(ROI_val,1));
roi.LabelAlpha = 0;
roi.LabelVisible = "hover";
roi.LabelTextColor = "white";
AREA = pi*roi.Radius^2 *1e-2; % considering that voxel size = 1mm x 1mm
fprintf('  METRICS:\n')
fprintf('    ROI val = %3.1f\n',ROI_val)
fprintf('    ROI sd  = %3.1f\n',ROI_std)
fprintf('    ROI area [cm^2] = %3.1f\n',AREA)
%% FOURTH BLOCK: SHOW ALL SLICES
figure(2)
imshow3D(outmap)