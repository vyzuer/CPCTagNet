from time import time
import numpy as np
import sys
import os
import random
from pymongo import MongoClient
import pickle
import inspect
import h5py
from sklearn.externals import joblib
import socket


"""
Output
---------
mpoi_profiling/random_samples.h5
mpoi_profiling/cluster_model/model.pkl
"""

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import MiniBatchKMeans, KMeans

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

n_clusters = gv.__NUM_CLUSTERS
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

    # num_tags = gv.__NUM_TAGS
    out_dir = gv.__base_dir

    _init_tag_list()


def clustering(X):

    # normalize data before clustering
    X = normalize(X)
    
    if _DEBUG:
        print 'performing clustering...'
    # Compute clustering with MiniBatchKMeans.
    mbk = KMeans(init='k-means++', n_clusters=n_clusters, max_iter=500,
                          n_init=12, verbose=1, tol=0.00000001, n_jobs=-1)
    # mbk = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, batch_size=4096, max_iter=1000,
    #                       n_init=5, max_no_improvement=100, verbose=1, init_size=2*n_clusters,
    #                       reassignment_ratio=0.1)

    t0 = 0
    if __DEBUG:
        t0 = time()
    mbk.fit(X)
    if __DEBUG:
        t_mini_batch = time() - t0
        print("Time taken to run KMeans %0.2f seconds" % t_mini_batch)

    return mbk


def load_data():

    X = None

    spath = out_dir + '/DB/topic_model/topic_distrib_lda.h5'

    if os.path.exists(spath):
        if _DEBUG:
            print 'loading data samples from disk...'
        h5f = h5py.File(spath, 'r')
        X = h5f['dataset_1'][:]
        h5f.close()
    else:
        print 'Error loading data...'
        exit(0)
    
    return X


def load_cluster_model():
    model = None

    spath = out_dir + '/DB/cluster_model/model.pkl'

    if os.path.exists(spath):
        if _DEBUG:
            print 'loading cluster model from disk...'
        model = joblib.load(spath)
    
    return model


def dump_model(model):
    spath = out_dir + '/DB/cluster_model/'
    if not os.path.exists(spath):
        os.makedirs(spath)

    fname = spath + 'model.pkl'

    joblib.dump(model, fname)


def get_cluster_model(data, clean=False):
    model = None
    # if present in disk load, otherwise perfor clustering
    if not clean:
        model = load_cluster_model()

    if model is None:
        model = clustering(data)
        dump_model(model)

    return model


def load_topic_distrbution():

    X = None

    spath = out_dir + '/DB/topic_model/topic_distrib.h5'

    if os.path.exists(spath):
        if _DEBUG:
            print 'loading topic distribution from disk...'
        h5f = h5py.File(spath, 'r')
        X = h5f['dataset_1'][:]
        h5f.close()
    else:
        print 'Error loading topic distribution data...'
        exit(0)
    
    return X


def perform_standardization(X):
    # perform feature scaling and dump the model for later use
    scaler = StandardScaler().fit(X)
    Y = scaler.transform(X)

    # dump the model
    spath = out_dir + '/DB/topic_model/scaler/'
    if not os.path.exists(spath):
        os.makedirs(spath)
    scaler_path = spath + '/scaler.pkl'
    joblib.dump(scaler, scaler_path)


    return Y

def _predict_and_update(model):
    """
        predict cluster id for each of the user
        and create mongodb for each user with cluster id and 
        topic distribution
    """

    # load data from disk for topic distribution
    data = load_topic_distrbution()

    # normalize data for cluster prediction
    data = normalize(data)

    # predic the cluster ids for each user
    cluster_ids = model.predict(data)

    # perform standardization
    data = perform_standardization(data)

    # iterate over all the users and update topic distribution and cluster ids
    users = user_tag_matrix.find(no_cursor_timeout=True).sort('_id', 1)

    for i, doc in enumerate(users):
        # update the topic distribution and cluster id
        user_tag_matrix.update_one({'_id': doc['_id']}, {'$set': {'cluster_id': int(cluster_ids[i]), 'userpref': data[i].tolist()}})
        
    users.close()

    
def master(clean=False):
    # load the data
    if _DEBUG:
        print 'loading data...'
    data = load_data()
    # print data[10,:]

    # get the cluster model 
    if _DEBUG:
        print 'getting cluster model...'
    model = get_cluster_model(data, clean)

    # predict cluster for each of the segment
    # and update the mongodb database
    if _DEBUG:
        print 'predicting user cluster and creating mogodb'
    _predict_and_update(model)


if __name__ == "__main__":
    if len(sys.argv) != 1:
        print "Usage: python sys.argv[0]"

    np.set_printoptions(precision=4, suppress=True)

    load_globals()

    master(clean=False)

