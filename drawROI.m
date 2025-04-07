%% RUN EACH BLOCK USING CTRL+ENTER
% FIRST BLOCK: LOAD MAT FILE
close all, clearvars, clc
[file,location] = uigetfile;
load([location,file])
%% SECOND BLOCK: DISPLAY SELECTED SLICE AND DRAW ROI
num_slice = 8; % YOU CAN CHANGE THIS
PDFF_slice = F(:,:,num_slice);
imshow(PDFF_slice,[0,100])
roi =drawcircle('InteractionsAllowed','translate','LineWidth',1,...
                'Center',[132,232],'Radius',sqrt(200/pi));
%% THIRD BLOCK: RUN TO REPEAT ROI MEASUREMENTS
clc
mask = createMask(roi);
ROI_PDFF = median(PDFF_slice(mask)); % You can change 'mean' by 'median'
roi.Label = num2str(round(ROI_PDFF,1));
roi.LabelAlpha = 0;
roi.LabelVisible = "hover";
roi.LabelTextColor = "white";
AREA = pi*roi.Radius^2 *1e-2; % considering that voxel size = 1mm x 1mm
fprintf('  METRICS:\n')
fprintf('    PDFF = %3.1f\n',ROI_PDFF)
fprintf('    ROI area [cm^2] = %3.1f\n',AREA)
