#!/usr/bin/env python


from fcn.models import FCN32s
import chainer.serializers as S
import os.path as osp
from skimage.io import imread
import numpy as np
from chainer import Variable


chainermodel = '/home/leus/ros/indigo/src/start-jsk/jsk_apc/jsk_apc2016_common/data/fcn32s_8000.chainermodel'

model = FCN32s(n_class=40)
S.load_hdf5(chainermodel, model)


data_dir = '/home/leus/projects/image_segmentation/fcn/examples/apc/dataset/APC2016/single_item_labeled'
img_path = osp.join(data_dir, 'woods_extension_cord_20160619171112_bin_f.jpg')

img = imread(img_path, mode='RGB')
print(img.shape)


# BGR
datum = img.astype(np.float32)
datum = datum[:, :, ::-1]  # RGB -> BGR
datum -= np.array((104.00698793, 116.66876762, 122.67891434))
datum = datum.transpose((2, 0, 1))


x_data = np.array([datum], dtype=np.float32)
x = Variable(x_data, volatile=False)
model(x)
pred = model.score
pred_datum = pred.data[0]





