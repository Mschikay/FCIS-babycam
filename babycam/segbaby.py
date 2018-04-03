"""
Created on Thursday Mar 29 2018

@author: ziqi Tang
"""
import _init_paths
import os
import sys
import cv2
from config.config import config, update_config
from utils.image import resize, transform
import numpy as np
import copy
from skimage import measure


# get config
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
cur_path = os.path.abspath(os.path.dirname(__file__))
update_config(cur_path + config.TEST_YAML)

sys.path.insert(0, os.path.join(cur_path, config.EX_MXNET, config.MXNET_VERSION))
import mxnet as mx
print "use mxnet at", mx.__file__
from core.tester import im_detect, Predictor
from symbols import *
from utils.load_model import load_param
from utils.show_masks import show_masks
from utils.tictoc import tic, toc
from nms.nms import py_nms_wrapper
from mask.mask_transform import gpu_mask_voting, cpu_mask_voting
import time


class segBaby(object):

    def __init__(self):
        self.ctx_id, self.data_names, self.predictor = self.loadModel()
        self.classes = config.TEST_CLASS_NAME
        self.num_classes = 81
        self.data = None
        self.im = np.array([])
        self.fg = np.array([])
        self.bg = np.array([])
        self.photo = np.array([])
        self.result = np.array([])


    #loading pretrained model
    def loadModel(self):
        # get symbol
        ctx_id = [int(i) for i in config.gpus.split(',')]
        # pprint.pprint(config)
        sym_instance = eval(config.symbol)()
        sym = sym_instance.get_symbol(config, is_train=False)

        # get predictor
        data_names = ['data', 'im_info']
        label_names = []
        max_data_shape = [[('data', (1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]]
        provide_data = [[max_data_shape[0][0], ('im_info', (1L, 3L))]]
        provide_label = [None]
        arg_params, aux_params = load_param(cur_path + config.PREMODEL, 0, process=True)
        predictor = Predictor(sym, data_names, label_names,
                              context=[mx.gpu(ctx_id[0])], max_data_shapes=max_data_shape,
                              provide_data=provide_data, provide_label=provide_label,
                              arg_params=arg_params, aux_params=aux_params)

        return ctx_id, data_names, predictor


    def loadImage(self, im, bg):
        self.im = im
        self.fg = im
        self.bg = bg

        target_size = config.SCALES[0][0]
        max_size = config.SCALES[0][1]
        self.im, im_scale = resize(self.im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tensor = transform(self.im, config.network.PIXEL_MEANS)
        im_info = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)
        self.data = {'data': im_tensor, 'im_info': im_info}
        self.data = [mx.nd.array(self.data[name]) for name in self.data_names]


    def Seg(self):
        for i in xrange(2):
            data_batch = mx.io.DataBatch(data=[self.data], label=[], pad=0, index=0,
                                         provide_data=[[(k, v.shape) for k, v in zip(self.data_names, self.data)]],
                                         provide_label=[None])
            scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]
            _, _, _, _ = im_detect(self.predictor, data_batch, self.data_names, scales, config)

        data_batch = mx.io.DataBatch(data=[self.data], label=[], pad=0, index=0,
                                     provide_data=[[(k, v.shape) for k, v in zip(self.data_names, self.data)]],
                                     provide_label=[None])
        scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]

        tic()
        scores, boxes, masks, data_dict = im_detect(self.predictor, data_batch, self.data_names, scales, config)
        #print masks #right
        im_shapes = [data_batch.data[i][0].shape[2:4] for i in xrange(len(data_batch.data))]

        if not config.TEST.USE_MASK_MERGE:
            all_boxes = [[] for _ in xrange(self.num_classes)]
            all_masks = [[] for _ in xrange(self.num_classes)]
            nms = py_nms_wrapper(config.TEST.NMS)
            for j in range(1, self.num_classes):
                indexes = np.where(scores[0][:, j] > 0.7)[0]
                cls_scores = scores[0][indexes, j, np.newaxis]
                cls_masks = masks[0][indexes, 1, :, :]
                try:
                    if config.CLASS_AGNOSTIC:
                        cls_boxes = boxes[0][indexes, :]
                    else:
                        raise Exception()
                except:
                    cls_boxes = boxes[0][indexes, j * 4:(j + 1) * 4]

                cls_dets = np.hstack((cls_boxes, cls_scores))
                keep = nms(cls_dets)
                all_boxes[j] = cls_dets[keep, :]
                all_masks[j] = cls_masks[keep, :]
            dets = [all_boxes[j] for j in range(1, self.num_classes)]
            masks = [all_masks[j] for j in range(1, self.num_classes)]
        else:
            masks = masks[0][:, 1:, :, :]
            im_height = np.round(im_shapes[0][0] / scales[0]).astype('int')
            im_width = np.round(im_shapes[0][1] / scales[0]).astype('int')
            print(im_height, im_width)
            boxes = clip_boxes(boxes[0], (im_height, im_width))
            result_masks, result_dets = gpu_mask_voting(masks, boxes, scores[0], self.num_classes,
                                                        100, im_width, im_height,
                                                        config.TEST.NMS, config.TEST.MASK_MERGE_THRESH,
                                                        config.BINARY_THRESH, self.ctx_id[0])

            dets = [result_dets[j] for j in range(1, self.num_classes)]
            masks = [result_masks[j][:, 0, :, :] for j in range(1, self.num_classes)]

        for i in xrange(1, len(dets)):
            keep = np.where(dets[i][:, -1] > 1)
            dets[i] = dets[i][keep]
            masks[i] = masks[i][keep]

        keep = np.where(dets[0][:, -1] > 0.8)
        dets[0] = dets[0][keep]
        masks[0] = masks[0][keep]
        
        newmask = show_masks(self.fg, dets, masks, self.classes, config) #!!!!!!!! wrong mask
        self.result = newmask
        return newmask


    def addBg(self, newmask):
        '''
        1. load and resize the background
        2. mix background and foreground to a photo
        3. save photo
        '''
        # @newmask: mask(type:float in [0,1]) from orginal segmentation code
        ksize=45
        binary_t = 0.05
        if self.bg.shape != self.fg.shape:
            self.bg = cv2.resize(self.bg, (self.fg.shape[1], self.fg.shape[0]))

        # mask filtering
        newmask = np.where(newmask>binary_t, 255, 0)
        newmask = np.array(newmask, dtype=np.uint8) # change the type to make it an image(uint8)
        edge = np.zeros((newmask.shape[0], newmask.shape[1]), dtype=np.uint8)
        mask = copy.deepcopy(edge)

        edge = self.drawEdge(newmask, edge, width=ksize)
        mask = edge
        mask = cv2.blur(mask, (ksize, ksize))
        weight = np.array(mask*1.0/255) # mask is int type
    
        # mix
        weight = cv2.merge([weight, weight, weight])
        newfg = np.multiply(weight, self.fg)
        newbg = np.multiply(pow(1-weight,1), self.bg)
        self.photo = np.add(newfg, newbg)
        self.photo = np.array(self.photo, dtype = np.uint8)
        return 

     
    def drawEdge(self, newmask, edge, width=30, degree=7):
        '''
        This function realizes: 
        1. expanding sementation area 
        2. smoothing the edge
        3. transition from foreground to background based on the mask
        '''
        # @edge: return the drawn image
        # @width: paint the edge with a width of @width
        # @degree: to what degree that the polylines become curve
        i, contours, h = cv2.findContours(newmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        longest = 0 # find the largest contour(which contains the baby)
        for i in range(len(contours)):
            if len(contours[i])>longest:
                longest = i
        maxcontour = contours[longest]
        hull = cv2.convexHull(maxcontour) # smoothing: find the min encircling convex based on max contours
        l = len(hull)
        hull = hull.reshape(l, 2)
        hull = np.row_stack((hull,[hull[0][0],hull[0][1]])) # loop-locked points
        for _ in range(1):
            curve = measure.subdivide_polygon(hull, degree=degree, preserve_ends=True) # polyline to curve

        lcurve = len(curve)
        curve = (curve.reshape(lcurve, 1, 2)).astype(int)
        cv2.drawContours(edge, [curve], 0, (255, 255, 255), -1) # paint inside area
        cv2.drawContours(edge, [curve], -1, (255, 255, 255), width) # paint outside edge
        return edge


    def run(self, im, bg, ):
        self.loadImage(im, bg)
        newmask = self.Seg()
        self.addBg(newmask)

if __name__ == '__main__':
    im = cv2.imread(config.FG)
    bg = cv2.imread(config.BG)

    segbaby = segBaby()
    segbaby.run(im, bg)

    cv2.imshow("", segbaby.photo)
    cv2.waitKey(0)

