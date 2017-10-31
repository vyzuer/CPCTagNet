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
import shutil

sys.path.append('/home/vyzuer/work/code/ftags/')

data_base = None
batch_size = 10000

import common.globals as gv

def _init():
    global data_base

    data_base = gv.__image_data

def resize(img, maxPx, minPx):
    try:
        width = img.size[0]
        height = img.size[1]
        smallest = min(width, height)
        largest = max(width, height)
        k = 1
        if largest > maxPx:
            k = maxPx / float(largest)
            smallest *= k
            largest *= k
        if smallest < minPx:
            k *= minPx / float(smallest)
        size = int(math.ceil(width * k)), int(math.ceil(height * k))
        img = img.resize(size, Image.ANTIALIAS)
        return img
    except IOError as e:
        print "I/O error({0}): {1}".format(e.errno, e.strerror)

def fillLmdb(labels_file, images, images_new, labels):
    cnt = 0

    base_dir = os.path.split(labels_file)[0]
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    labels_db = lmdb.open(labels_file, map_size=int(1e12))

    labels_txn = labels_db.begin(write=True)

    max_cnt = images_new.size

    examples = zip(images, labels)
    for in_idx, (image, label) in enumerate(examples):
        if cnt < max_cnt and image == images_new[cnt]:
            #save label
            label = np.array(label).astype(float).reshape(1,1,len(label))
            label_dat = caffe.io.array_to_datum(label)
            labels_txn.put('{:0>10d}'.format(in_idx), label_dat.SerializeToString())
            
            cnt += 1

            # write batch
            if cnt%batch_size == 0:
                labels_txn.commit()
                labels_txn = labels_db.begin(write=True)
                print 'saved batch: ', cnt

        if in_idx%500 == 0:
            string_ = str(in_idx+1) + ' / ' + str(len(images))
            sys.stdout.write("%s\r" % string_)
            sys.stdout.flush()

    if cnt%batch_size != 0:
        labels_txn.commit()
        print 'saved last batch: ', cnt

    labels_db.close()

    print "\nFilling lmdb completed"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create lmdb for caffe'
    )
    parser.add_argument(
        '--images',
        type=str,
        help='Images file',
        required=True
    )
    parser.add_argument(
        '--images_dst',
        type=str,
        help='Images file',
        required=True
    )
    parser.add_argument(
        '--labels',
        type=str,
        help='Labels npy file',
        required=True
    )
    parser.add_argument(
        '--labelsOut',
        type=str,
        help='Labels lmdb',
        required=True
    )

    args = parser.parse_args()

    # print images_new
    images = np.loadtxt(args.images, str)
    images_new = np.loadtxt(args.images_dst, str)
    labels = io.mmread(args.labels).tocsr().toarray()

    _init()

    print "Creating train set..."

    fillLmdb(
        labels_file=args.labelsOut,
        images=images[:],
        images_new=images_new[:],
        labels=labels[:])

    print "Completed."

