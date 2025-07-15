function [im, info, filename, pathname] = open_dicom(pathname, filename)

if ~exist('pathname', 'var') && ~exist('filename', 'var') 
    % [filename, pathname] = uigetfile('IM*;*.dcm;00*', 'Select an Image', 'MultiSelect', 'on');
    [filename, pathname] = uigetfile('*', 'Select an Image', 'MultiSelect', 'on');
end

if iscell(filename)
    aux = dicomread(strcat(pathname,filename{:,1}));
    % aux_info = dicominfo(strcat(pathname,filename{:,1}));
    if ndims(aux)<4
        im = cell(1, size(filename,2));
        info = cell(1, size(filename,2));
    %     info = zeros(1, size(filename,2));
    
        h = waitbar(0,'Loading DICOM files... 0%');
        steps = size(filename,2);
    
        for i=1:size(filename,2)
            aux = dicomread(strcat(pathname,filename{:,i}));
            aux2 = dicominfo(strcat(pathname,filename{:,i}));
%           im(:,:,i) = aux;
            im{i} = aux;
            info{i} = aux2;
            waitbar(i / steps, h, ['Loading DICOM files...' num2str(round(i/steps*100)) '%']);
        end
    else
        h = waitbar(0,'Loading DICOM files... 0%');
        steps = size(filename,2);
        ii = 1;
        for i=1:size(filename,2)
            aux = squeeze(dicomread(strcat(pathname,filename{:,i})));
            aux2 = dicominfo(strcat(pathname,filename{:,i}));
%           im(:,:,i) = aux;
            for j=1:size(aux,3)
                im{ii} = aux(:,:,j);
                filename2{ii} = filename{i};
                ii = ii + 1;
            end
            info{i} = aux2;
            waitbar(i / steps, h, ['Loading DICOM files...' num2str(round(i/steps*100)) '%']);
        end
        filename = filename2;
    end        
    close(h);
else
    if filename(:,1) == 0
        im = 0;
        info = 0;
        pathname = 0;
    else
        aux = dicomread(strcat(pathname,filename(1,:)));
        info = dicominfo(strcat(pathname,filename(1,:)));
        if ndims(aux)==4
            for i=1:size(aux,4)
                im{i} = squeeze(aux(:,:,:,i));
            end
        else
            im = dicomread(strcat(pathname,filename(1,:)));
            info = dicominfo(strcat(pathname,filename(1,:)));
        end
    end
%     im = dicomread(strcat(pathname,filename(1,:)));
end