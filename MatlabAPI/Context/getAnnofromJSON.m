function anno = getAnnofromJSON(part_annos, imname, part_annos_imgid, part_annos_cates,part2senmatic)

% when you convert from final_ans to anno, your convertion has one more
% sturct, which is 'silh'

anno = {};
anno.imname = imname;
% find the part indices that belong to this image
imname = erase(imname, "_");
index = find(strcmp(imname, part_annos_imgid));
% 
anno.objects = struct('class', '', 'class_ind', '', 'mask', '', 'parts', '');

for ii = 1 : length(index)
    % get the category
    cate_id = part_annos.annotations(index(ii)).category_id;
    anno.objects(ii).class = part_annos.categories(part_annos_cates == cate_id).name;
    partsnames = part_annos.categories(part_annos_cates == cate_id).parts;
    partsindices =cell2mat({partsnames.id});
    
    % transfer the category id (459 classes) to the original PASCAL 20 classes;
    cate_id = find(part2senmatic == cate_id);
    % 
    anno.objects(ii).class_ind = cate_id;
    % get the instance segmentation;
    inst_rle = part_annos.annotations(index(ii)).segmentation;
    inst_mask  = logical(MaskApi.decode(inst_rle));
    anno.objects(ii).mask = inst_mask;
    
    % get the part segmentation
    parts_seg = part_annos.annotations(index(ii)).parts;
    anno.objects(ii).parts = struct('part_name', '', 'mask', '');
    % could have a if to bypass the 'silh' to be exact the same with the anno which is written to the json;
    startindex = 0;
    for jj = 1 : length(parts_seg)
        % part_id doesn't correspond to any part;
        if parts_seg(jj).part_id == 255
            continue;
        end
        % bypass the sihl part for valid object parts;
        if parts_seg(jj).part_id == 100
            if length(parts_seg) > 1
                startindex = 1;
                continue;
            else % length(parts_seg) == 1
                break;
            end
        end
        
        partnameindex = partsindices == parts_seg(jj).part_id;
        anno.objects(ii).parts(jj - startindex).part_name = partsnames(partnameindex).name;
        anno.objects(ii).parts(jj - startindex).mask = logical(MaskApi.decode(parts_seg(jj).segmentation));
    end
    
end

end