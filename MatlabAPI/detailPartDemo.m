% This is the DEMO for the PASCAL Part Segmentation dataset
% The original pascal has 20 objects, however, we only annotated 16
% objects out of the 20 classes. The "boat", "chair", "diningtable" and
% "sofa" don't have the definitions of the parts. 

% More details, please refer to the following two links:
% https://sites.google.com/view/pasd;
% http://www.stat.ucla.edu/~xianjie.chen/pascal_part_dataset/pascal_part.html;

addpath('./Context/');

image_path = '/media/zhuotun/Data/PASCAL/VOCdevkit/VOC2012/JPEGImages'; % './examples';
json_path = '/media/zhuotun/Work_HD/MSCOCO/zhuotun/part/trainval/PASCAL_part_trainval.json';
%images = part_train;%dir([image_path, '/', '*.jpg']);

% correspondences between the category id (459 classes) to the original PASCAL 20 classes;
load('./Context/part2senmatic.mat');
part_annos = gason(fileread(json_path));
part_annos_imgid = num2str((cell2mat({part_annos.annotations.image_id}))');
part_annos_cates = cell2mat({part_annos.categories.id});

% Shuffle image order
images = cell2mat({part_annos.images.file_name}');
images = images(randperm(length(images)), :);

cmap = VOClabelcolormap();
pimap = part2ind();     % part index mapping
f = figure;

for ii = 1 : length(images) %   numel
    imname = images(ii, :);
    imname = string(imname(1 : end - 4));
    img = imread([image_path, '/', [char(imname) '.jpg']]); %  [char(imname) '.jpg']
    
    % missing anno on training; bypass the missing annos
    if strcmp(imname, "2010_001606") || strcmp(imname, "2010_001592") 
        fprintf('Missing anno for %s.\n', imname);
        continue;
    end
    
    % read the JSON file and get the anno;
    anno = getAnnofromJSON(part_annos, imname, part_annos_imgid, part_annos_cates, part2senmatic);
    
    [cls_mask, inst_mask, part_mask] = mat2map(anno, img, pimap);
    
    %     % determine whether this is a fully annotated image or not
    %     in_train = '';
    %     if ~any(strcmp(part_train, imname))
    %         in_train = ' - NOT ANNOTATED';
    %     end
    %     set(f, 'Name', [char(imname) in_train]);
    
    % display annotation
    subplot(2,2,1); imshow(img); title('Image');
    subplot(2,2,2); imshow(cls_mask, cmap); title('Class Mask');
    subplot(2,2,3); imshow(inst_mask, cmap); title('Instance Mask');
    subplot(2,2,4); imshow(part_mask, cmap);
    if length(unique(part_mask)) == 1
        title('No Defined Part Mask');
    else
        title('Part Mask');
    end
    pause;
end
