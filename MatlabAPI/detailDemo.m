%% Demo for the Detail API (see DetailApi.m)

%% initialize api (please specify dataType/annType below)
annTypes = { 'instances', 'captions', 'person_keypoints' };
dataType='val2014'; annType=annTypes{1}; % specify dataType/annType
annFile=sprintf('../annotations/%s_%s.json',annType,dataType);
detail=DetailApi(annFile);

%% display COCO categories and supercategories
if( ~strcmp(annType,'captions') )
  cats = detail.loadCats(detail.getCatIds());
  nms={cats.name}; fprintf('COCO categories: ');
  fprintf('%s, ',nms{:}); fprintf('\n');
  nms=unique({cats.supercategory}); fprintf('COCO supercategories: ');
  fprintf('%s, ',nms{:}); fprintf('\n');
end

%% get all images containing given categories, select one at random
catIds = detail.getCatIds('catNms',{'person','dog','skateboard'});
imgIds = detail.getImgIds('catIds',catIds);
imgId = imgIds(randi(length(imgIds)));

%% load and display image
img = detail.loadImgs(imgId);
I = imread(sprintf('../images/%s/%s',dataType,img.file_name));
figure(1); imagesc(I); axis('image'); set(gca,'XTick',[],'YTick',[])

%% load and display annotations
annIds = detail.getAnnIds('imgIds',imgId,'catIds',catIds,'iscrowd',[]);
anns = detail.loadAnns(annIds); detail.showAnns(anns);
