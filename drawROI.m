%% RUN EACH BLOCK USING CTRL+ENTER
% FIRST BLOCK: LOAD MAT FILE
close all, clearvars, clc
addpath(genpath([pwd '/matlab']));

sel_map = 'AI-PDFF'; % OPTIONS: PDFF,AI-PDFF,AI-PDFF-var,R2s,AI-R2s,
                % AI-R2s-var,AI-FM,ECH-1,ECH-2 - YOU CAN CHANGE THIS
location = uigetdir();
location = [location,'/'];
curFolder = cd;
cd(location)
if strcmp(sel_map,'ECH-1') || strcmp(sel_map,'ECH-2')
    filenames = dir('IM_*');
elseif strcmp(sel_map,'AI-PDFF') || strcmp(sel_map,'AI-PDFF-var') ...
        || strcmp(sel_map,'AI-R2s') || strcmp(sel_map,'AI-R2s-var') ...
        || strcmp(sel_map,'AI-FM')
    prefold = dir('2D_NSA1_*');
    filenames = dir([prefold.name,'/res_*']);
else
    filenames = dir('results_MP_GC/IM*');
end
file = filenames.name;
cd(curFolder)
%% SECOND BLOCK: DISPLAY SELECTED SLICE AND DRAW ROI
num_slice = 15; % YOU CAN CHANGE THIS
if strcmp(sel_map,'PDFF')
    load([location,'results_MP_GC/',file])
    outmap = F;
    val_range = [0,100];
elseif strcmp(sel_map,'AI-PDFF')
    load([location,prefold.name,'/',file])
    outmap = F;
    val_range = [0,100]; %RANGE OF VALUES TO BE PLOTTED - YOU CAN CHANGE THIS
elseif strcmp(sel_map,'AI-PDFF-var')
    load([location,prefold.name,'/',file])
    outmap = F_var;
    val_range = [0,2e4]; %RANGE OF VALUES TO BE PLOTTED - YOU CAN CHANGE THIS
elseif strcmp(sel_map,'R2s')
    load([location,'results_MP_GC/',file])
    outmap = R2;
    val_range = [0,200];
elseif strcmp(sel_map,'AI-R2s')
    load([location,prefold.name,'/',file])
    outmap = R2;
    val_range = [0,200];
elseif strcmp(sel_map,'AI-R2s-var')
    load([location,prefold.name,'/',file])
    outmap = R2_var;
    val_range = [0,3e2]; %RANGE OF VALUES TO BE PLOTTED - YOU CAN CHANGE THIS
elseif strcmp(sel_map,'AI-FM')
    load([location,prefold.name,'/',file])
    outmap = P;
    val_range = [0,200]; %RANGE OF VALUES TO BE PLOTTED - YOU CAN CHANGE THIS
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
                'Center',[115,208],'Radius',sqrt(200/pi));
% roi.Center
%% THIRD BLOCK: RUN TO REPEAT ROI MEASUREMENTS
clc
mask = createMask(roi);
if strcmp(sel_map,'PDFF') || strcmp(sel_map,'AI-PDFF')
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
fprintf('  ROI METRICS:\n')
fprintf('    Val = %3.1f\n',ROI_val)
fprintf('    SD  = %3.1f\n',ROI_std)
fprintf('    Area [cm^2] = %3.1f\n',AREA)
if contains(sel_map,'AI')
    PDFF_slice = F(:,:,num_slice); PDFF = median(PDFF_slice(mask));
    PDFFv_slice= F_var(:,:,num_slice); PDFFv = mean(PDFFv_slice(mask));
    R2_slice = R2(:,:,num_slice); R2c = mean(R2_slice(mask));
    R2v_slice = R2_var(:,:,num_slice); R2v = mean(R2v_slice(mask));
    FM_slice = P(:,:,num_slice); FM = mean(FM_slice(mask));
    FMv_slice = P_var(:,:,num_slice); FMv = mean(FMv_slice(mask));
    fprintf('  Q-Maps at ROI:\n')
    fprintf('    PDFF = %3.1f\n',PDFF)
    fprintf('    PDFF var = %3.1f\n',PDFFv)
    fprintf('    R2* = %3.1f\n',R2c)
    fprintf('    R2* var = %3.1f\n',R2v)
    fprintf('    FM = %3.1f\n',FM)
    fprintf('    FM var = %3.1f\n',FMv)
elseif not(contains(sel_map,'ECH'))
    PDFF_slice = F(:,:,num_slice); PDFF = median(PDFF_slice(mask));
    R2_slice = R2(:,:,num_slice); R2 = mean(R2_slice(mask));
    FM_slice = P(:,:,num_slice); FM = mean(FM_slice(mask));
    fprintf('  Q-Maps at ROI:\n')
    fprintf('    PDFF = %3.1f\n',PDFF)
    fprintf('    R2* = %3.1f\n',R2)
    fprintf('    FM = %3.1f\n',FM)
end
%% FOURTH BLOCK: SHOW ALL SLICES
figure(2)
imshow3D(outmap,val_range)