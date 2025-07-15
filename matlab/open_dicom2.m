function [im, filename, pathname] = open_dicom2(pathname, filename)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

if ~exist('pathname', 'var') && ~exist('filename', 'var') 
    [im_aux, ~, filename_aux, pathname_aux] = open_dicom;
elseif exist('pathname', 'var') && exist('filename', 'var')
    [im_aux, ~, filename_aux, pathname_aux] = open_dicom(pathname, filename);
else
    return
end

if ~iscell(filename_aux) && ~isstruct(filename_aux)
    if filename_aux == 0
        im = im_aux;
        filename = filename_aux;
        pathname = pathname_aux;
        error('Aborted operation. No file selected.');
    else
        if ndims(im_aux)==1
            im{1} = im_aux;
        else
            im = im_aux;
        end
        filename = filename_aux;
        pathname = pathname_aux;
        
        if length(im)==1
            aux = squeeze(im{1});
            for i=1:size(aux,3)
                im{i} = aux(:,:,i);
            end
        end
    end
else
    im = im_aux;
    filename = filename_aux;
    pathname = pathname_aux;
end

end

