__author__ = 'mricha56'
__version__ = '3.0'
# Interface for accessing the PASCAL in Detail dataset. detail is a Python API
# that assists in loading, parsing, and visualizing the annotations of PASCAL
# in Detail. Please visit https://sites.google.com/view/pasd/home for more
# information about the PASCAL in Detail challenge. For example usage of the
# detail API, see detailDemo.ipynb.


# The following API functions are defined:
#  Detail     - Detail api class that loads the annotation file and prepares data structures.
#  decodeMask - Decode binary mask M encoded via run-length encoding.
#  encodeMask - Encode binary mask M using run-length encoding.
#  getAnns  - Load annotations that satisfy given filter conditions.
#  getCats  - Load categories that satisfy given filter conditions.
#  getImgs  - Load images that satisfy given filter conditions.
#  annToMask  - Convert segmentation in an annotation to binary mask.
#  showAnns   - Display the specified annotations.
#  loadRes    - Load algorithm results and create API for accessing them.
#  download   - Download COCO images from mscoco.org server.
# Throughout the API "ann"=annotation, "cat"=category, and "img"=image.


# PASCAL in Detail Toolbox     version 3.0
# Modifications of COCO toolbox made by Matt Richard and Zhuotun Zhu
# Forked from:
#     Microsoft COCO Toolbox.      version 2.0
#     Data, paper, and tutorials available at:  http://mscoco.org/
#     Code written by Piotr Dollar and Tsung-Yi Lin, 2014.
#     Licensed under the Simplified BSD License [see bsd.txt]

import json
import time
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np
import skimage.io as io
import copy
import itertools
from . import mask as maskUtils
import os
from collections import defaultdict
import sys
PYTHON_VERSION = sys.version_info[0]
if PYTHON_VERSION == 2:
    from urllib import urlretrieve
elif PYTHON_VERSION == 3:
    from urllib.request import urlretrieve

class Detail:
    def __init__(self, annotation_file='json/trainval_merged.json',
                 image_folder='VOCdevkit/VOC2010/JPEGImages',
                 phase='trainval'):
        """
        Constructor of Detail helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that has pascal JPEG images.
        :param phase (str): image set to look at: train, val, test, or any combination
                            of the three (trainval, trainvaltest)
        :return:
        """
        # load dataset
        self.phase = phase
        self.img_folder = image_folder

        print('loading annotations into memory...')
        tic = time.time()

        self.data = json.load(open(annotation_file, 'r'))
        assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
        print('Done (t={:0.2f}s)'.format(time.time()- tic))

        self.__createIndex()

    def __createIndex(self):
        # create index
        print('creating index...')

        # create class members
        self.cats,self.imgs,self.instances,self.semantic,self.occlusion,
            self.parts= {},{},{},{},{},{}

        phases = []
        if "train" in self.phase: splits.append("train")
        if "val" in self.phase: splits.append("val")
        if "test" in self.phase: splits.append("test")
        assert len(phases) > 0, 'Invalid phase, {}'.format(self.phase)

        # Filter images and annotations according to phase
        for img in data['images']:
            if img['phase'] not in phases:
                data['images'].remove(img)
            else:
                self.imgs[img['image_id']] = img

        imgIds = list(self.imgs.keys())
        for segm in data['annos_segmentation']:
            if segm['image_id'] not in imgIds:
                data['annos_segmentation'].remove(segm)
            elif not segm['iscrowd']:
                self.instances[segm['id']] = segm
            else:
                self.semantic[segm['id']] = segm

        for occl in data['annos_occlusion']:
            if occl['image_id'] not in imgIds:
                data['annos_occlusion'].remove(occl)
            else:
                self.occlusion[occl['id']] = occl

        # Follow references
        for cat in self.data['categories']:
            cat['images'] = []
            cat['annotations'] = []
            cats[cat['category_id']] = cat

        for img in self.data['images']:
            img['annotations'] = []
            img['categories'] = []

        for instance_id, instance in self.instances:
            img = self.imgs[instance['image_id']]
            cat = self.cats[instance['category_id']]
            img['annotations'].append(instance_id)
            cat['annotations'].append(instance_id)

            if cat['category_id'] not in img['categories']:
                img['categories'].apend(cat['category_id'])
            if img['image_id'] not in cat['images']:
                cat['images'].append(img['image_id'])

        for semantic_id, semantic in self.semantic:
            img = self.imgs[semantic['image_id']]
            cat = self.cats[semantic['category_id']]
            img['annotations'].append(semantic_id)
            cat['annotations'].append(semantic_id)

            if cat['category_id'] not in img['categories']:
                img['categories'].append(cat['category_id'])
            if img['image_id'] not in cat['images']:
                cat['images'].append(img['image_id'])

        for occl_id, occl in self.occlusion:
            img = self.imgs[occl['image_id']]
            cat = self.cats[occl['category_id']]
            img['annotations'].append(occl_id)
            cat['annotations'].append(occl_id)

            if cat['category_id'] not in img['categories']:
                img['categories'].append(cat['category_id'])
            if img['image_id'] not in cat['images']:
                cat['images'].append(img['image_id'])

        print('index created!')

    def info(self):
        """
        Print information about the annotation file.
        :return:
        """
        for key, value in self.dataset['info'].items():
            print('{}: {}'.format(key, value))

    def getAnns(self, annIds=[], imgs=[], cats=[], areaRng=[]):
        """
        Get annotations that satisfy given filter conditions. default is no filter
        :param annIds (int array) : get anns with the given IDs
               imgs  (image array)     : get anns in the given imgs
               cats  (category array)     : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
        :return: anns  (annotation array)       : array of annotations
        """
        imgs = imgs if type(imgs) == list else [imgs]
        cats = cats if type(cats) == list else [cats]
        annIds = annIds if type(annIds) == list else [annIds]

        # Convert imgs and cats to simple IDs
        for i in range(len(imgs)):
            imgs[i] = self.__getImgId(imgs[i])
        for i in range(len(cats)):
            cats[i] = self.__getCatId(cats[i])

        # initialize anns from index
        if len(annIds) > 0:
            try: anns = [self.anns[annId] for annId in annIds]
            except IndexError: assert False, 'Annotation with the given IDs not found: %s' % annIds
        elif len(imgs) > 0:
            lists = [self.imgs[imgId]['annotations'] for imgId in imgs]
            ids = list(itertools.chain.from_iterable(lists))
            anns = [self.anns[id] for id in ids]
        elif len(cats) > 0:
            lists = [self.cats[catId]['annotations'] for catId in cats]
            ids = list(itertools.chain.from_iterable(lists))
            anns = [self.anns[id] for id in ids]
        else:
            anns = self.anns[:] # shallow copy self.anns

        # filter according to params
#        if len(annIds) > 0:
#            anns = [ann for ann in anns if ann['id'] in annIds]
        if len(imgs) > 0:
            anns = [ann for ann in anns if ann['image_id'] in imgs]
        if len(cats) > 0:
            anns = [ann for ann in anns if ann['category_id'] in cats]
        if len(areaRng) > 0:
            anns = [ann for ann in anns if ann['area'] > areaRng[0] \
                    and ann['area'] < areaRng[1]]

        return anns


    def getMask(self, img, cat=None, instance=None, part=None, show=False):
        """
        Get mask for a particular level of segmentation (can be an instance, class, or part mask).

        If instance-level semgntation is requested (img and cat provided), the result is an image whose pixel
        values are the instance IDs for that class and 0 everywhere else.

        If part-level segmentation is requested (img, cat, and instance provided),
        the result is an image whose pixel values are the part IDs for that instance
        and 0 everywhere else.

        Lastly, if semantic segmentation is requested (cat=instance=part=None), the result
        is an image whose pixel values are the class IDs for that image.

        :param img (string/int/dict) : image that mask describes
               cat (string/int/dict) : category that mask describes
               instance (string/int/dict) : instance that the mask describes (default
                                            is all instances). If integer, interpreted
                                            as id of an "annotation" object in JSON. If
                                            string starting with #, e.g. '#0', interpreted as 0-based index
                                            of instance within the image (cat is None)
                                            or of instance within the given class (cat not None).
               part (string or int) :  part that mask describes (None means all parts)
               show (boolean) :  whether to pass the mask to self.showMask() before returning.
        :return: mask (numpy 2D array) : a mask describing the requested annotation.
        """

        # Validate params and convert them to dicts
        img = self.getImgs(imgs=self.__getImgId(img))[0]
        if cat is not None:
            cat = self.getCats(cats=self.__getCatId(cat))[0]
        if part is not None:
            if type(part) is str:
                assert cat is not None, \
                    "Couldn't infer part ID from part name - please specify a category"
                # infer part from name using `cat` param
                part = [p for p in cat['parts'] if p['name'] == part][0]
            elif type(part) is int:
                # get part as dict
                for c in self.getCats(cats=img['categories']):
                    for p in c['parts']:
                        if p['part_id'] == part:
                            part = p
                            break
                    if type(part) is not int:
                        break
        if instance is not None:
            if type(instance) is str:
                if instance.startswith('#'):
                    # If instance is set to '#N' where N is an integer,
                    # get the Nth (0-indexed) instance of the given category.
                    instance = self.getAnns(imgs=img, cats=([] if cat is None else cat))[int(instance[1:])]
                else:
                    instance = int(instance)

            if type(instance) is int:
                instance = self.getAnns(instance)[0]


        anns = self.getAnns(annIds=img['annotations'])
        mask = np.zeros((img['height'], img['width']))

        # Generate mask based on params
        if cat is instance is part is None:
            # Generate class mask
            for ann in anns:
                m = self.decodeMask(ann['segmentation'])
                mask[np.nonzero(m)] = ann['category_id']
        elif cat is not None and instance is part is None:
            # Generate instance mask
            for ann in anns:
                if ann['category_id'] == cat['category_id']:
                    m = self.decodeMask(ann['segmentation'])
                    mask[np.nonzero(m)] = ann['id']
        elif instance is not None and part is None:
            # Generate part mask
            for p in instance['parts']:
                m = self.decodeMask(p['segmentation'])
                mask[np.nonzero(m)] = p['part_id']
        elif instance is not None and part is not None:
            # Generate single-part mask
            partMask = [p['segmentation'] for p in instance['parts'] \
                        if p['part_id'] == part['part_id']]
            assert len(partMask) > 0, 'Coudn\'t find a part mask for the given part and instance'
            partMask = partMask[0]
            m = self.decodeMask(partMask)
            mask[np.nonzero(m)] = part['part_id']
        elif cat is not None and part is not None:
            # Generate single-part mask
            ann = self.getAnns(imgs=img, cats=cat)
            assert len(ann) == 1, 'The given category must have one instance, or part should be provided.'
            ann = ann[0]
            partMask = [p['segmentation'] for p in ann['parts'] \
                        if p['part_id'] == part['part_id']]
            assert len(partMask) > 0, 'Couldn\'t find a part mask for the given part and category'
            partMask = partMask[0]
            m = self.decodeMask(partMask)
            mask[np.nonzero(m)] = part['part_id']
        else:
            assert False, 'Invalid parameters'

        if show:
            self.showMask(mask)

        return mask

    def showMask(self, mask):
        """
        Display given mask (numpy 2D array) as a colormapped image.
        """
        nonzero = mask[np.nonzero(mask)]
        plt.imshow(mask, clim=[np.min(nonzero) - 1, np.max(nonzero) + 1])
        plt.axis('off')
        plt.show()

    def __getImgId(self, img):
        if type(img) == int: return img
        if type(img) == dict: return img['image_id']
        img = img.split('.')[0] # cut off .jpg extension
        return img[:4] + img[5:] # '2008_000002' --> 2008000002

    def __getCatId(self, cat):
        if type(cat) == int: return cat
        if type(cat) == dict: return cat['category_id']
        return [c for c in self.cats if self.cats[c]['name'] == cat][0]

    def getCats(self, cats=[], imgs=[]):
        """
        Get categories abiding by the given filters. default is no filter.
        :param cats (int/string/dict array)  : get cats for given cat names/ids/dicts
        :param imgs (int/string/dict array)  : get cats common across image names/ids
        :return: categories (dict array)   : array of category dicts
        """
        categories = []

        cats = cats if type(cats) == list else [cats]
        imgs = imgs if type(imgs) == list else [imgs]

        for i in range(len(imgs)):
            imgs[i] = self.__getImgId(imgs[i])
        for i in range(len(cats)):
            cats[i] = self.__getCatId(cats[i])

        if len(imgs) == 0 and len(cats) == 0:
            categories = list(self.cats.values())
        elif len(cats) > 0:
            categories = [self.cats[catId] for catId in cats]
            if len(imgs) > 0:
                for img in imgs:
                    categories = [category for category in categories if img in category['images']]
        else: # only images given
            lists = [self.imgs[imgId]['categories'] for imgId in imgs]
            catIds = set(lists[0])
            for l in range(1, len(lists)):
                catIds = catIds.intersect(set(l))
            categories = [self.cats[catId] for catId in catIds]



        return categories

    def getParts(self, cat):
        """
        Get parts of a particular category.
        :param cat (int, string, or dict) : category to get parts for
        :return: parts (dict array) : array of part dicts, e.g.
        [{"name": "mouth", "superpart": "head", "part_id": 110},...]
        """
        cat = self.getCats(cat)
        assert len(cat) > 0, 'Invalid category'
        cat = cat[0]
        return list(cat['parts'])

    def getImgs(self, imgs=[], cats=[]):
        '''
        Get images that satisfy given filter conditions.
        :param imgs (int array) : get imgs with given ids
        :param cats (int array) : get imgs with all given cats
        :return: images (dict array)  :  array of image dicts
        '''
        images = []

        imgs = imgs if type(imgs) == list else [imgs]
        cats = cats if type(cats) == list else [cats]

        for i in range(len(imgs)):
            imgs[i] = self.__getImgId(imgs[i])
        for i in range(len(cats)):
            cats[i] = self.__getCatId(cats[i])

        if len(imgs) == 0 and len(cats) == 0:
            images = list(self.imgs.values())
        elif len(imgs) > 0:
            images = [self.imgs[imgId] for imgId in imgs]
            if len(cats) > 0:
                for cat in cats:
                    images = [image for image in images if cat in image['categories']]
        else: # only cats given
            lists = [self.cats[catId]['images'] for catId in cats]
            imgIds = set(lists[0])
            for l in range(1,len(lists)):
                imgIds = imgIds.intersect(set(l))
            images = [self.imgs[imgId] for imgId in imgIds]

        return images

    # From COCO API...not implemented for PASCAL in Detail
#     def showAnns(self, anns):
#         """
#         Display the specified annotations.
#         :param anns (array of object): annotations to display
#         :return: None
#         """
#         if len(anns) == 0:
#             return 0
#         if 'segmentation' in anns[0] or 'keypoints' in anns[0]:
#             datasetType = 'instances'
#         elif 'caption' in anns[0]:
#             datasetType = 'captions'
#         else:
#             raise Exception('datasetType not supported')
#         if datasetType == 'instances':
#             ax = plt.gca()
#             ax.set_autoscale_on(False)
#             polygons = []
#             color = []
#             for ann in anns:
#                 c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
#                 if 'segmentation' in ann:
#                     if type(ann['segmentation']) == list:
#                         # polygon
#                         for seg in ann['segmentation']:
#                             poly = np.array(seg).reshape((int(len(seg)/2), 2))
#                             polygons.append(Polygon(poly))
#                             color.append(c)
#                     else:
#                         # mask
#                         t = self.imgs[ann['image_id']]
#                         if type(ann['segmentation']['counts']) == list:
#                             rle = maskUtils.frPyObjects([ann['segmentation']], t['height'], t['width'])
#                         else:
#                             rle = [ann['segmentation']]
#                         m = maskUtils.decode(rle)
#                         img = np.ones( (m.shape[0], m.shape[1], 3) )
#                         if ann['iscrowd'] == 1:
#                             color_mask = np.array([2.0,166.0,101.0])/255
#                         if ann['iscrowd'] == 0:
#                             color_mask = np.random.random((1, 3)).tolist()[0]
#                         for i in range(3):
#                             img[:,:,i] = color_mask[i]
#                         ax.imshow(np.dstack( (img, m*0.5) ))
#                 if 'keypoints' in ann and type(ann['keypoints']) == list:
#                     # turn skeleton into zero-based index
#                     sks = np.array(self.loadCats(ann['category_id'])[0]['skeleton'])-1
#                     kp = np.array(ann['keypoints'])
#                     x = kp[0::3]
#                     y = kp[1::3]
#                     v = kp[2::3]
#                     for sk in sks:
#                         if np.all(v[sk]>0):
#                             plt.plot(x[sk],y[sk], linewidth=3, color=c)
#                     plt.plot(x[v>0], y[v>0],'o',markersize=8, markerfacecolor=c, markeredgecolor='k',markeredgewidth=2)
#                     plt.plot(x[v>1], y[v>1],'o',markersize=8, markerfacecolor=c, markeredgecolor=c, markeredgewidth=2)
#             p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
#             ax.add_collection(p)
#             p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
#             ax.add_collection(p)
#         elif datasetType == 'captions':
#             for ann in anns:
#                 print(ann['caption'])


    # From COCO API...not implemented for PASCAL in Detail
#     def loadRes(self, resFile):
#         """
#         Load result file and return a result api object.
#         :param   resFile (str)     : file name of result file
#         :return: res (obj)         : result api object
#         """
#         res = Detail()
#         res.dataset['images'] = [img for img in self.dataset['images']]
#
#         print('Loading and preparing results...')
#         tic = time.time()
#         if type(resFile) == str or type(resFile) == unicode:
#             anns = json.load(open(resFile))
#         elif type(resFile) == np.ndarray:
#             anns = self.loadNumpyAnnotations(resFile)
#         else:
#             anns = resFile
#         assert type(anns) == list, 'results in not an array of objects'
#         annsImgIds = [ann['image_id'] for ann in anns]
#         assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), \
#                'Results do not correspond to current Detail set'
#         if 'caption' in anns[0]:
#             imgIds = set([img['image_id'] for img in res.dataset['images']]) & set([ann['image_id'] for ann in anns])
#             res.dataset['images'] = [img for img in res.dataset['images'] if img['image_id'] in imgIds]
#             for id, ann in enumerate(anns):
#                 ann['id'] = id+1
#         elif 'bbox' in anns[0] and not anns[0]['bbox'] == []:
#             res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
#             for id, ann in enumerate(anns):
#                 bb = ann['bbox']
#                 x1, x2, y1, y2 = [bb[0], bb[0]+bb[2], bb[1], bb[1]+bb[3]]
#                 if not 'segmentation' in ann:
#                     ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
#                 ann['area'] = bb[2]*bb[3]
#                 ann['id'] = id+1
#                 ann['iscrowd'] = 0
#         elif 'segmentation' in anns[0]:
#             res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
#             for id, ann in enumerate(anns):
#                 # now only support compressed RLE format as segmentation results
#                 ann['area'] = maskUtils.area(ann['segmentation'])
#                 if not 'bbox' in ann:
#                     ann['bbox'] = maskUtils.toBbox(ann['segmentation'])
#                 ann['id'] = id+1
#                 ann['iscrowd'] = 0
#         elif 'keypoints' in anns[0]:
#             res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
#             for id, ann in enumerate(anns):
#                 s = ann['keypoints']
#                 x = s[0::3]
#                 y = s[1::3]
#                 x0,x1,y0,y1 = np.min(x), np.max(x), np.min(y), np.max(y)
#                 ann['area'] = (x1-x0)*(y1-y0)
#                 ann['id'] = id + 1
#                 ann['bbox'] = [x0,y0,x1-x0,y1-y0]
#         print('DONE (t={:0.2f}s)'.format(time.time()- tic))
#
#         res.dataset['annotations'] = anns
#         res.createIndex()
#         return res


    # From COCO API...not implemented for PASCAL in Detail
#     def loadNumpyAnnotations(self, data):
#         """
#         Convert result data from a numpy array [Nx7] where each row contains {imageID,x1,y1,w,h,score,class}
#         :param  data (numpy.ndarray)
#         :return: annotations (python nested list)
#         """
#         print('Converting ndarray to lists...')
#         assert(type(data) == np.ndarray)
#         print(data.shape)
#         assert(data.shape[1] == 7)
#         N = data.shape[0]
#         ann = []
#         for i in range(N):
#             if i % 1000000 == 0:
#                 print('{}/{}'.format(i,N))
#             ann += [{
#                 'image_id'  : int(data[i, 0]),
#                 'bbox'  : [ data[i, 1], data[i, 2], data[i, 3], data[i, 4] ],
#                 'score' : data[i, 5],
#                 'category_id': int(data[i, 6]),
#                 }]
#         return ann

    def decodeMask(self, json):
        """
        Convert json mask to binary mask.
        :return: binary mask (numpy 2D array)
        """
        return maskUtils.decode(json)
