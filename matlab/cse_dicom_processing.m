function S = cse_dicom_processing(im,filename,pathname)
ref_im = dicominfo([pathname filename]);
    PerframeFunctionalGroup = ref_im.(dicomlookup('5200','9230'));
    fnames = fieldnames(PerframeFunctionalGroup);
    nRows = ref_im.(dicomlookup('0028','0010'));
    nColumns = ref_im.(dicomlookup('0028','0011'));
    nFrames = ref_im.(dicomlookup('0028','0008'));
    nSlices = ref_im.(dicomlookup('2001', '1018'));
    [EchoNumber, SliceNumber, EffectiveEchoTime] = deal(zeros([nFrames*nSlices, 1]));
    dim_im = [nRows, nColumns, nSlices, nFrames/nSlices];
    imageType = cell(1,nFrames);
    imS = zeros([nRows, nColumns, nFrames]);
    for i=1:nFrames
        PixelValueTransformation = PerframeFunctionalGroup.(fnames{i}).(dicomlookup('0028','9145'));
        Private_Tag = PerframeFunctionalGroup.(fnames{i}).(dicomlookup('2005','140f')).Item_1;
        SliceNumber(i) = Private_Tag.(dicomlookup('2001','100a'));
        RescaleIntercept = PixelValueTransformation.Item_1.(dicomlookup('0028','1052'));
        RescaleSlope = PixelValueTransformation.Item_1.(dicomlookup('0028','1053'));
        imS(:,:,i) = RescaleIntercept + RescaleSlope*double(im{i});
        imageType{i} = PerframeFunctionalGroup.(fnames{i}).(dicomlookup('0018','9226')).Item_1.(dicomlookup('0008','9208'));
        EchoNumber(i) = Private_Tag.(dicomlookup('0018','0086'));
        EffectiveEchoTime(i) = Private_Tag.(dicomlookup('0018','0081'));
    end
    maxEcho = max(EchoNumber); Ncoils = 1;
    [Sr, Si] = deal(zeros([nRows, nColumns, nSlices, Ncoils, maxEcho]));
    for i=1:nFrames
        if strcmp(imageType{i},'REAL')
            Sr(:,:,SliceNumber(i),1,EchoNumber(i)) =  imS(:,:,i);
        elseif strcmp(imageType{i},'IMAGINARY')
            Si(:,:,SliceNumber(i),1,EchoNumber(i)) =  imS(:,:,i);
        end
    end
    S =  Sr + 1i*Si;
end