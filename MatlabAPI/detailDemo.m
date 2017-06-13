%% Demo for the detailApi
% to-do: first you need to customize the annFile and imgPath for your own path;

clear all; close all;
%% initialize Pascal api (please specify dataType/annType below)
annTypes = { 'instances'}; %, 'captions', 'person_keypoints' 
annType=annTypes{1}; % specify dataType/annType % val

annFile = '../../json/PASCAL_trainval.json';
imgPath = '../../images';

coco=DetailApi(annFile);

%% display COCO categories and supercategories
if( ~strcmp(annType,'captions') )
  cats = coco.loadCats(coco.getCatIds());
  nms={cats.name}; fprintf('categories: ');
  fprintf('%s, ',nms{:}); fprintf('\n');
  nms=unique({cats.supercategory}); fprintf('supercategories: ');
  fprintf('%s, ',nms{:}); fprintf('\n');
end

%% get all images containing given categories, select one at random
catIds = coco.getCatIds('catNms',{'person'}); % ,'dog','skateboard'  dog horse
imgIds = coco.getImgIds('catIds',catIds);
imgId = imgIds(randi(length(imgIds)));

%% load and display image
img = coco.loadImgs(imgId);
I = imread(sprintf([imgPath '/%s'], img.file_name));
figure(1); imagesc(I); axis('image'); set(gca,'XTick',[],'YTick',[])

%% load and display annotations
annIds = coco.getAnnIds('imgIds',imgId,'catIds',catIds,'iscrowd',[]);
anns = coco.loadAnns(annIds); coco.showAnns(anns);
