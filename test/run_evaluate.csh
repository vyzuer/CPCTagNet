#! /bin/csh

# 0 - no context
# 1 - contagnet
# 2 - ucontagnet
# 3 - cucontagnet
set context = 3
set model_path = "/home/vyzuer/work/caffe/models/"
# set model = "/home/vyzuer/work/caffe/models/contagnet/"
# set model = "/home/vyzuer/work/caffe/models/cuctagnet/"

# set context = 0
# set model = "/home/vyzuer/work/caffe/models/finetune_flickr_tag/"

# python train.py ${model}

python evaluate.py ${model_path} ${context}

# foreach i (0 1 2 3)
#     python evaluate.py ${model_path} ${i}
# end
