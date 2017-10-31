#!/bin/csh

# shuffle and split the image list
# python split_shuffle_train_data.py

# dump image list and labels
# python dump_train_list.py

# dump test list and labels
# python dump_test_list.py

# base directory for storing all the media data
# set base_dir = /home/vyzuer/work/data/DataSets/tags/data/
# set base_dir = /hdfs/masl/tag_data/

# perform topic modeling and dump the topic distribution for later use
python topic_modeling.py

# perform clustering on extracted topics
python perform_clustering.py

