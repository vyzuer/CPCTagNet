from PIL import Image
import numpy as np
import lmdb
import caffe
import sys
import math
import argparse
import random
from scipy import sparse, io
import os
import time
import sys 
import os
import matplotlib.pyplot as plt
# from pymongo import MongoClient

import numpy as np
import os.path as osp

from copy import copy

caffe_root = '/home/yogesh/work/caffe/'
sys.path.append(caffe_root + 'python')
import caffe # If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.

from caffe import layers as L, params as P # Shortcuts to define the net prototxt.

sys.path.append("pycaffe/layers") # the datalayers we will use are in this directory.
sys.path.append("pycaffe") # the tools file is in this folder

import tools #this contains some tools that we need

sys.path.append('/home/yogesh/work/code/ftags/')

batch_size = 100
n_samples = 46700
# n_samples = 1000
clasess = None
photo_ids = None
dump_dir = None
num_asamples = 0

# these variables are for measuring individual tag accuracy
tp_ptag = None # true positive rate
ap_ptag = None # prediction rate
total_ptag = None # total present

n_tags = None

import common.globals as gv

models = {0:'tagnet', 1:'contagnet', 2:'uctagnet', 3:'cuctagnet'}
prediction_layers = {0:'fc8_tag', 1:'fc9_flickr_con', 2:'fc10_flickr_con', 3:'fc10_flickr_con'}

def _init():

    global classes
    global photo_ids
    global dump_dir
    global n_tags

    caffe.set_mode_gpu()
    caffe.set_device(2)

    photo_ids = np.loadtxt('/home/yogesh/work/data/DataSets/ftags/test.list', dtype=str)

    dump_dir = gv.__dump_dir

    n_tags = gv.__NUM_TAGS_1540
    classes = np.loadtxt('/home/yogesh/work/data/DataSets/ftags/tags_1540_list.txt', dtype=str)


def evaluate_k(gt, est, k=5):
    """
        the est are the estimated labels
        """
    global tp_ptag 
    global ap_ptag 
    global total_ptag

    acc = 0.0
    prec = 0.0
    rec = 0.0
    tp = 0.0

    tag_ids = est.argsort()[::-1]

    for i in range(k):
        _id = tag_ids[i]
        if gt[_id] == 1:
            acc = 1.0
            tp += 1.0

    prec = tp/k
    rec = tp/np.sum(gt)

    if k == 5:
        num_tags = np.sum(gt)
        for i in range(num_tags):
            _id = tag_ids[i]
            ap_ptag[_id] += 1
            if gt[_id] == 1:
                tp_ptag[_id] += 1
            
        total_ptag += gt

    return acc, prec, rec
    

def check_baseline_accuracy(net, num_batches):
    acc = 0.0
    for t in range(num_batches):
        net.forward()
        gts = net.blobs['label'].data[:,0,0,:]
        ests = np.zeros(gts.shape)
        for gt, est in zip(gts, ests): #for each ground truth and estimated label vector
            acc += hamming_distance(gt, est)
    return acc / (1. * num_batches * batch_size)


def hamming_distance(gt, est):
    return sum([1 for (g, e) in zip(gt, est) if g == e]) / float(len(gt))

def check_accuracy_nonc(net, k=5):
    global num_asamples
    num_asamples = 0
    num_batches = n_samples/batch_size
    acc = 0.0
    prec = 0.0
    rec = 0.0
    acc1 = 0.0
    for t in range(num_batches):
        net.forward()
        gts = net.blobs['label'].data[:,0,0,:]
        ests = net.blobs['fc8_flickr'].data[:,:]
        for gt, est in zip(gts, ests): #for each ground truth and estimated label vector
            if np.sum(gt) > 4:
                num_asamples += 1
                acc += hamming_distance(gt, est > 0)
                a, p, r = evaluate_k(gt, est, k)
                acc1 += a
                prec += p
                rec += r

    acc1 /= (1. * num_asamples)
    prec /= (1. * num_asamples)
    rec  /= (1. * num_asamples)
    print 'acc: %.6f, prec: %.6f, rec: %.6f' %(acc1, prec, rec)

    return np.array([acc1, prec, rec])

def check_accuracy(net, k=5, context=1):
    global num_asamples
    pred_layer = prediction_layers[context]

    fname = dump_dir + 'prediction_val_' + str(k) + '_' + str(context) + '.tsv'
    fp = open(fname, 'w')
    num_asamples = 0
    num_batches = n_samples/batch_size
    acc = 0.0
    prec = 0.0
    rec = 0.0
    acc1 = 0.0
    for t in range(num_batches):
        net.forward()
        gts = net.blobs['label'].data[:,0,0,:]
        ests = net.blobs[pred_layer].data[:,:]
        i = 0
        for gt, est in zip(gts, ests): #for each ground truth and estimated label vector
            if np.sum(gt) > 0:
                num_asamples += 1
                a, p, r = evaluate_k(gt, est, k)
                acc1 += a
                prec += p
                rec += r

                estlist = est.argsort()[::-1][:5]
                pred = [classes[estlist]]
                ppath = photo_ids[t*batch_size + i]
                # print ppath
                photo_id = os.path.splitext(os.path.split(ppath)[1])[0]
                fp.write('%s\t%.3f\t%.3f\t%.3f\t' %(photo_id, a, p, r))
                np.savetxt(fp, pred, fmt='%s', delimiter=',')
            else:
                print photo_ids[t*batch_size + i]

            i += 1

    acc1 /= (1. * num_asamples)
    prec /= (1. * num_asamples)
    rec  /= (1. * num_asamples)
    print 'acc: %.6f, prec: %.6f, rec: %.6f' %(acc1, prec, rec)
    print 'total samples: ', num_asamples

    fp.close()

    return np.array([acc1, prec, rec])

def predict_tags(net, context, k=5):
    fname = dump_dir + 'prediction_' + models[context] + '_' + str(k) + '.tsv'
    fp = open(fname, 'w')
    num_batches = n_samples/batch_size
    for t in range(num_batches):
        net.forward()
        # gts = net.blobs['label'].data[0,0,:]
        for i in range(batch_size):
            # pred_label = net.blobs['fc8_flickr'].data[i, ...]
            layer = prediction_layers[context]
            pred_label = net.blobs[layer].data[i, ...]
            estlist = pred_label.argsort()[::-1][:k]
            pred = [classes[estlist]]
            ppath = photo_ids[t*batch_size + i]
            # print ppath
            photo_id = os.path.splitext(os.path.split(ppath)[1])[0]
            fp.write('%s\t' %(photo_id))
            np.savetxt(fp, pred, fmt='%s', delimiter=',')

    fp.close()


def predict_accuracy(net, context):

    global tp_ptag 
    global ap_ptag 
    global total_ptag

    tp_ptag = np.zeros(n_tags)
    ap_ptag = np.zeros(n_tags)
    total_ptag = np.zeros(n_tags)

    fname = dump_dir + 'results_' + str(context) + '.list'
    fp = open(fname, 'w')
    # for k in (1, 3, 5, 10):
    for k in ([5]):
        res = check_accuracy(net, k, context)

        fp.write('k: %d\n' %(k))
        fp.write('accuracy precision recall\n')
        np.savetxt(fp, res, fmt='%.6f')

    fp.close()

    # dump the per tag accuracy results
    prec = [x/y if y else 0 for x,y in zip(tp_ptag,ap_ptag)]
    rec = [x/y if y else 0 for x,y in zip(tp_ptag,total_ptag)]

    pt_acc = np.vstack([prec, rec]).transpose()
    fname_pt = dump_dir + 'per_tag_' + str(context) + '.list'
    np.savetxt(fname_pt, pt_acc, fmt='%.6f')


def master(model_dir, context, viz = False):

    model_def = model_dir + models[context] + '/test.prototxt'    
    model_weights = None
    model_weights = caffe_root + 'models/' + models[context] + '/alexnet_' + models[context] + '_0_iter_1000000.caffemodel'
    net = caffe.Net(model_def,      # defines the structure of the model
                    model_weights,  # contains the trained weights
                    caffe.TEST)     # use test mode (e.g., don't perform dropout)

    # predict_tags(net, context)

    predict_accuracy(net, context)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print 'usage: python ', sys.argv[0], ' model_dir', 'context_mode'
        exit(0)

    model_path = sys.argv[1]
    context = int(sys.argv[2])

    _init()

    master(model_path, context=context, viz=False)

