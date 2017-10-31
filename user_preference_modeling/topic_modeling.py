from pymongo import MongoClient
import numpy as np
import sys
import os
import urllib
import logging
import socket
import time
import random
from scipy import sparse, io
from dateutil import parser
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import h5py
from sklearn.decomposition import LatentDirichletAllocation

# debug flags
__DEBUG = True
_DEBUG = True

tag_db = None
yfcc100m = None
user_tag_matrix = None
data_dir = None
num_splits = None
num_tags = None
num_dim = 6
max_users = 300000
tag_mapping = {}
out_dir = None
min_user_tags = 20

sys.path.append('/home/vyzuer/work/code/ftags/')

import common.globals as gv

num_clusters = gv.__NUM_CLUSTERS
num_topics = gv.__NUM_TOPICS

def _init_tag_list():
    global tag_mapping
    
    tag_col = tag_db['tags_1540']
    cursor = tag_col.find(no_cursor_timeout=True)

    for doc in cursor:
        tag_mapping[doc['_id']] = doc['label']
        
    cursor.close()


def load_globals():
    global tag_db
    global yfcc100m
    global data_dir
    global num_splits
    global num_tags
    global user_tag_matrix
    global out_dir

    client = None

    host_name = socket.gethostname()
    if host_name == 'cassandra':
        client = MongoClient()
    else:
        client = MongoClient('172.29.35.126:27019')
        # client = MongoClient('localhost:27019')

    tag_db = client.flickr_tag
    yfcc100m = tag_db['yfcc100m']
    user_tag_matrix = tag_db['usertag_matrix_train']

    data_dir = gv.__dataset_path

    num_splits = gv.__num_splits
    num_tags = gv.__NUM_TAGS
    out_dir = gv.__base_dir

    _init_tag_list()


def get_collection(col_id):
    col_name = 'train_' + str(col_id)
    # load the dataset
    col = tag_db[col_name]

    return col

def get_labels(tags):

    labels = np.zeros(num_tags, dtype='int')

    for t in tags.split(','):
        _id = tag_mapping[t]

        labels[_id] = 1

    return labels


def _plot_histogram(x):

    num_bins = np.max(x)-np.min(x)+1
    hist, bins = np.histogram(x, num_bins)
    plt.axis([0, num_bins, 0, np.max(hist)])
    # width = 0.8 * (bins[1] - bins[0])
    # center = (bins[:-1] + bins[1:]) / 2
    # plt.bar(center, hist, align='center', width=width)
    plt.plot(hist)
    plt.title('Distribution of Tags Usage')
    f_name = out_dir + 'DB/data_analysis/tag_usage_dist.png'
    plt.savefig(f_name)
    plt.close()


def _dump_user_tag_matrix(utm, labels_count):
    # dump the compressed matrix 
    base_dir = out_dir + '/DB/data/'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    fname = base_dir + 'user_tag_matrix.mtx'
    if os.path.exists(fname):
        os.remove(fname)

    sparse_labels = sparse.csr_matrix(utm)
    io.mmwrite(fname, sparse_labels)

    # dump the data for later use
    spath = out_dir + '/DB/data_analysis/tag_usage.list'
    np.savetxt(spath, labels_count.reshape(1,-1), fmt='%d')


def _load_user_tag_matrix():
    utm = None
    labels_count = None

    # load the compressed matrix if exists
    fname = out_dir + '/DB/data/user_tag_matrix.mtx'
    if os.path.exists(fname):
        utm = io.mmread(fname).tocsr().toarray()

    # load the label count
    spath = out_dir + '/DB/data_analysis/tag_usage.list'
    if os.path.exists(spath):
        labels_count = np.loadtxt(spath, dtype=int)

    return utm, labels_count


def _get_user_tag_matrix():
    
    # check if user tag matrix is already in the disk
    # try to load from the disk
    utm, labels_count = _load_user_tag_matrix()

    # create matrix using mongodb
    if utm is None:
        print 'Not stored in disk, creating matrix...\n'

        users = user_tag_matrix.find(no_cursor_timeout=True).sort('_id', 1)
        num_users = users.count()

        utm = np.zeros(shape=(num_users, num_tags), dtype='int')
        labels_count = np.zeros(num_users, dtype='int')

        for i, user in enumerate(users):
            if i%100 == 0:
                sys.stdout.flush()
                stat = i*100./num_users
                print 'status: [%.2f%%]\r'%(stat),

            ulabels = np.array(user['labels'])
            
            utm[i,:] = ulabels
            labels_count[i] = np.count_nonzero(ulabels)

        print '\ndone!\n'

        users.close()

        # dump the utm and labels_count
        _dump_user_tag_matrix(utm, labels_count)

        _plot_histogram(labels_count)


    return utm, labels_count


def _dump_topic_distrib(topic_distrib):
    spath = out_dir + '/DB/topic_model/topic_distrib.h5'

    h5f = h5py.File(spath, 'w')
    h5f.create_dataset('dataset_1', data=topic_distrib)
    h5f.close()


def _dump_model(model, topic_dist):
    spath = out_dir + '/DB/topic_model/lda_model/'
    if not os.path.exists(spath):
        os.makedirs(spath)

    fname = spath + 'model.pkl'

    joblib.dump(model, fname)

    # dump the topic distribution for selected data for clustering
    spath = out_dir + '/DB/topic_model/topic_distrib_lda.h5'

    h5f = h5py.File(spath, 'w')
    h5f.create_dataset('dataset_1', data=topic_dist)
    h5f.close()


def _perform_topic_modeling(data):

    if __DEBUG:
        n_samples, n_features = data.shape
        print("Fitting LDA models with tf features")
        print("n_samples=%d and n_features=%d..." % (n_samples, n_features))

    lda = LatentDirichletAllocation(n_topics=num_topics, max_iter=100,
                                    learning_method='online', learning_offset=10.,
                                    random_state=0, batch_size=8192, verbose=1,n_jobs=-1)
    t0 = time.time()
    topic_distrib = lda.fit_transform(data)
    print("done in %0.3fs." % (time.time() - t0))

    return lda, topic_distrib


def master():
    # load the user tag matrix for topic modeling
    if _DEBUG:
        print 'Loading user tag matrix...\n'
    utm, labels_count = _get_user_tag_matrix()

    # remove users with less than min_tags
    data = []
    for i, n_tags in enumerate(labels_count):
        if n_tags >= min_user_tags:
            data.append(utm[i,:])

    data = np.array(data)

    # perform topic modeling
    if _DEBUG:
        print 'performing topic modeling...'
    model, topic_dist = _perform_topic_modeling(data)

    # dump the lda model
    if _DEBUG:
        print 'dumping topic-model...'
    _dump_model(model, topic_dist)

    if _DEBUG:
        print 'dumping topic distribution...'
    topic_distrib = model.transform(utm)
    _dump_topic_distrib(topic_distrib)

    if _DEBUG:
        print 'topic modeling done.'

    

if __name__ == "__main__":
    if len(sys.argv) != 1:
        print "Usage: python sys.argv[0]"
        exit(0)

    load_globals()

    master()

