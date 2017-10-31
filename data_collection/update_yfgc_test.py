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
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.externals import joblib

tag_db = None
yfcc100m = None
data_dir = None
num_splits = None
num_tags = None
num_dim = 6
yfcc100m_count = 100000000
tag_mapping = {}
user_tag_matrix = None
out_dir = None

sys.path.append('/home/vyzuer/work/code/ftags/')

import common.globals as gv

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
    # tag_db.drop_collection('usertag_matrix_test')
    user_tag_matrix = tag_db['usertag_matrix_test']


    data_dir = gv.__database_path

    num_splits = gv.__num_splits
    num_tags = gv.__NUM_TAGS
    out_dir = gv.__base_dir

    _init_tag_list()


def get_context(doc):
    lat = doc['Latitude']
    lon = doc['Longitude']
    date = doc['Date_taken']
    if date is u'' or date == 'null':
        date = datetime.today()
    else:
        date = parser.parse(date)

    if lat is u'':
        lat = 0.0
    if lon is u'':
        lon = 0.0
    
    context = [lat, lon, date.minute + date.hour*60, date.day, date.month, date.weekday()]
    return context


def get_details(pid):

    doc = yfcc100m.find_one({'_id': pid})

    lat = doc['Latitude']
    lon = doc['Longitude']
    date = doc['Date_taken']
    tags = doc['User_tags']

    return lat, lon, date, tags


def normalize_context(col):
    cursor = col.find(no_cursor_timeout=True).sort('_id', 1)

    # store the labels in an array
    num_samples = cursor.count()
    context = np.zeros(shape=(num_samples, num_dim))
    for idx, doc in enumerate(cursor):
        # get the context
        context[idx,:] = get_context(doc)

    cursor.close()

    base_dir = data_dir + '/train/'
    spath = base_dir + '/scaler/'
    scaler_path = spath + '/scaler.pkl'
    scaler = joblib.load(scaler_path)

    # perform standard scaling
    context = scaler.transform(context)

    cursor = col.find(no_cursor_timeout=True).sort('_id', 1)
    for idx, doc in enumerate(cursor):
        # update the normalized context for later use
        col.update_one({'_id': doc['_id']}, {'$set':{'norm_context': context[idx,:].tolist()}})


    cursor.close()

def update_database(col):
    cursor = col.find(no_cursor_timeout=True).sort('_id', 1)

    for i, doc in enumerate(cursor):
        pid = doc['_id']

        lat, lon, date, tags = get_details(pid)

        # update the database
        col.update_one({'_id': doc['_id']}, {'$set':{"tags": tags, "Latitude": lat, "Longitude": lon, "Date_taken": date}})

    cursor.close()



def get_labels(tags):

    labels = np.zeros(num_tags, dtype='int')

    for t in tags.split(','):
        try:
            _id = tag_mapping[t]
            labels[_id] = 1
        except Exception as e:
            pass

    return labels


def predict_td(data):
    spath = out_dir + '/DB/topic_model/lda_model/model.pkl' 
    model = joblib.load(spath)

    X = model.transform(data)

    X = normalize(X)

    return X


def predict_cid(data):
    spath = out_dir + '/DB/cluster_model/model.pkl'
    model = joblib.load(spath)

    cids = model.predict(data)

    return cids

def perform_standardization(data):
    spath = out_dir + '/DB/topic_model/scaler/scaler.pkl'
    model = joblib.load(spath)

    X = model.transform(data)

    return X


def create_utm(col):

    cursor = col.find(no_cursor_timeout=True)

    max_users = col.count()

    # first create a database of all the users
    # then iterate over the yfcc100m to find the users
    # it will be less expensive as compared to perfrom a find on yfcc100m
    k = 0
    for i, doc in enumerate(cursor):
        if i%100 == 0:
            sys.stdout.flush()
            stat = i*100./max_users
            print 'status: [%.2f%%]\r'%(stat),

        uid = doc['uid']
        user_id = k

        user = user_tag_matrix.find_one({'_id': uid})
        
        if user is None:
            user_tag_matrix.insert_one({'_id': uid, 'uid': k})
            k += 1
        else:
            pass

    cursor.close()

def create_user_tag_matrix(col):
    # create user_tag_matrix
    create_utm(col)

    # iterate over yfcc100m
    u_photos = yfcc100m.find(no_cursor_timeout=True)

    num_users = user_tag_matrix.count()
    labels = np.zeros(shape=(num_users, num_tags), dtype='int')
    
    # iterate through these photographs and skip those which are in test set
    for i, pic in enumerate(u_photos):
        pid = pic['_id']
        uid = pic['User_NSID']

        if i%1000 == 0:
            sys.stdout.flush()
            stat = i*100./yfcc100m_count
            print 'status: [%.4f%%]\r'%(stat),

        # check if we have this user in our test set
        user = user_tag_matrix.find_one({'_id':uid})
        
        # if user is present ensure this photo is not in the test set
        if user is not None and col.find_one({'_id':pid}) is None:
            tags = pic['User_tags']
            if type(tags) is unicode:
                # print tags
                labels[user['uid'],:] += get_labels(tags)


    u_photos.close()

    print '\ndone.\n'

    return labels

def _dump_user_tag_matrix(utm):
    # dump the compressed matrix 
    base_dir = out_dir + '/DB/data/'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    fname = base_dir + 'user_tag_matrix_test.mtx'
    if os.path.exists(fname):
        os.remove(fname)

    sparse_labels = sparse.csr_matrix(utm)
    io.mmwrite(fname, sparse_labels)


def _load_user_tag_matrix():
    utm = None

    # load the compressed matrix if exists
    fname = out_dir + '/DB/data/user_tag_matrix_test.mtx'
    if os.path.exists(fname):
        utm = io.mmread(fname).tocsr().toarray()

    return utm


def get_user_tag_matrix(col):
    # if exists in disk load
    utm = _load_user_tag_matrix()

    # otherwise create from database
    if utm is None:
        utm = create_user_tag_matrix(col)
        _dump_user_tag_matrix(utm)

    return utm


def update_td_cids(labels):

    # predict the topic distribution
    topic_distrib = predict_td(labels)

    # predict cluster ids
    cluster_ids = predict_cid(topic_distrib)

    # standardize
    userpref = perform_standardization(topic_distrib)

    # update the database for topic distribution and cluster id
    # iterate through all the users and update database for labels
    cursor = user_tag_matrix.find(no_cursor_timeout=True)

    for doc in cursor:

        uid = doc['uid']
        user_tag_matrix.update_one({'_id': doc['_id']}, {'$set':{'labels': labels[uid,:].tolist(), 'cluster_id': int(cluster_ids[uid]), 'userpref': userpref[uid].tolist()}})

    cursor.close()


def master():

    col = tag_db['test_data']

    # first update the database with tags and context data
    print 'updating test database...'
    # update_database(col)
    print 'done.'

    # normalize the context
    print 'normalizing and updating context...'
    # normalize_context(col)
    print 'done.'

    # create user tag matrix for users in test set
    print 'creating user tag matrix...'
    labels = get_user_tag_matrix(col)
    print 'done.'

    # update the user preferences
    print 'updating user preferences...'
    update_td_cids(labels)
    print 'done.'


if __name__ == "__main__":
    if len(sys.argv) != 1:
        print "Usage: python sys.argv[0]"
        exit(0)

    load_globals()

    master()

