import tensorflow.compat.v1 as tf
import os
from utils import DataLoader, load, save, psnr_error, diff_gt, objectness_rgb_estimation
from constant import const
import matplotlib
import matplotlib.pyplot as py
import cv2
matplotlib.use('TkAgg')
# from edge_boxes_python import edge_boxes_python
tf.disable_v2_behavior()
#tf.enable_eager_execution()
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = const.GPU

dataset_name = const.DATASET
test_folder = const.TEST_FOLDER

batch_size = const.BATCH_SIZE
num_his = const.NUM_HIS
height, width = 256, 256
flow_height, flow_width = const.FLOW_HEIGHT, const.FLOW_WIDTH

l_num = const.L_NUM
alpha_num = const.ALPHA_NUM
lam_lp = const.LAM_LP
lam_gdl = const.LAM_GDL
lam_adv = const.LAM_ADV
lam_flow = const.LAM_FLOW
adversarial = (lam_adv != 0)

summary_dir = const.SUMMARY_DIR
snapshot_dir = const.SNAPSHOT_DIR
with tf.name_scope('dataset'):
    test_loader = DataLoader(test_folder, resize_height=height, resize_width=width)
    test_dataset = test_loader(batch_size=batch_size, time_steps=num_his, num_pred=1)
    test_it = tf.compat.v1.data.make_one_shot_iterator(test_dataset)
    test_videos_clips_tensor = test_it.get_next()
    test_videos_clips_tensor.set_shape([batch_size, height, width, 3*(num_his + 1)])
    test_inputs = test_videos_clips_tensor[..., 0:num_his*3]
    test_gt = test_videos_clips_tensor[..., -3:]
    #objectness_rgb_map_test = objectness_rgb_estimation(test_gt)
    objectness_rgb_map_test = objectness_rgb_estimation(test_gt)
print(f'array is {objectness_rgb_map_test[0,:,:,:]}')
print(f'shape is {test_gt.shape}')
proto_tensor = objectness_rgb_map_test[0,:,:,0].eval(session=tf.compat.v1.Session())
proto_tensor1 = test_gt[0,:,:,:].eval(session=tf.compat.v1.Session())
#proto_tensor=tf.linalg.normalize(objectness_rgb_map_test[0,:,:,:],axis=-1)[0]
#proto_tensor = proto_tensor.eval(session=tf.compat.v1.Session())
print(f'attention map is {proto_tensor}\n')
print(f'Ground truth is {proto_tensor1}')
py.imshow(proto_tensor)
py.show()
cv2.waitKey(0)
cv2.imshow('Ground Truth',proto_tensor1)
cv2.waitKey(0)
cv2.destroyAllWindows()
