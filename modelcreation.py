import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from common.nets.module import BackboneNet, PoseNet
from common.nets.loss import JointHeatmapLoss, HandTypeLoss, RelRootDepthLoss
from main.config import cfg
import math
from PIL import Image
from torchvision import transforms
from main.model import Model 
from common.utils.vis import vis_keypoints, vis_3d_keypoints
from common.utils.preprocessing import load_img, load_skeleton, process_bbox, generate_patch_image, transform_input_to_output_space, trans_point2d


# Load the checkpoint
checkpoint = torch.load(r"C:\\Users\\user\\Desktop\\Codes\\SignLanguageProject\\InterHand2.6M\\snapshot.tar", map_location='cpu')

network = checkpoint['network']

# Create an instance of the model architecture
# You should replace 'ModelClass' with the actual class of your model
backbone_net = BackboneNet()
pose_net = PoseNet(21)
model_instance = Model(backbone_net, pose_net)

# Load the model's state_dict from the 'network' key
model_instance.load_state_dict(network,strict=False)

# Move the model to the appropriate device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_instance.to(device)

# Now you can use the 'model_instance' for inference or further training
imsize = 256
loader = transforms.Compose([transforms.ToTensor()])

img_path = r"C:\\Users\\user\\Desktop\\Codes\\SignLanguageProject\\InterHand2.6M\\hello4.jpeg"
image = load_img(img_path)
original_img_height, original_img_width = image.shape[:2]

joint_num = 21 # single hand
root_joint_idx = {'right': 20, 'left': 41}
joint_type = {'right': np.arange(0,joint_num), 'left': np.arange(joint_num,joint_num*2)}
skeleton = load_skeleton(r"C:\Users\user\Desktop\Codes\SignLanguageProject\InterHand2.6M\skeleton.txt", joint_num*2)

# prepare bbox
bbox = [69, 137, 165, 153] # xmin, ymin, width, height
bbox = process_bbox(bbox, (original_img_height, original_img_width, original_img_height))
img, trans, inv_trans = generate_patch_image(image, bbox, False, 1.0, 0.0, cfg.input_img_shape)
transform = transforms.ToTensor()
img = transform(img.astype(np.float32))/255
img = img[None,:,:,:]
model_instance.eval()
inputs = {'img': img}
targets = {}
meta_info = {}
with torch.no_grad():
    out = model_instance(inputs, targets, meta_info, 'test')
img = img[0].cpu().numpy().transpose(1,2,0) # cfg.input_img_shape[1], cfg.input_img_shape[0], 3
joint_coord = out['joint_coord'][0].cpu().numpy() # x,y pixel, z root-relative discretized depth
rel_root_depth = out['rel_root_depth'][0].cpu().numpy() # discretized depth
hand_type = out['hand_type'][0].cpu().numpy() # handedness probability

# restore joint coord to original image space and continuous depth space
joint_coord[:,0] = joint_coord[:,0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
joint_coord[:,1] = joint_coord[:,1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
joint_coord[:,:2] = np.dot(inv_trans, np.concatenate((joint_coord[:,:2], np.ones_like(joint_coord[:,:1])),1).transpose(1,0)).transpose(1,0)
joint_coord[:,2] = (joint_coord[:,2]/cfg.output_hm_shape[0] * 2 - 1) * (cfg.bbox_3d_size/2)

# restore right hand-relative left hand depth to continuous depth space
rel_root_depth = (rel_root_depth/cfg.output_root_hm_shape * 2 - 1) * (cfg.bbox_3d_size_root/2)

# right hand root depth == 0, left hand root depth == rel_root_depth
joint_coord[joint_type['left'],2] += rel_root_depth
# handedness
joint_valid = np.zeros((joint_num*2), dtype=np.float32)
right_exist = False
if hand_type[0] > 0.5: 
    right_exist = True
    joint_valid[joint_type['right']] = 1
left_exist = False
if hand_type[1] > 0.5:
    left_exist = True
    joint_valid[joint_type['left']] = 1

print('Right hand exist: ' + str(right_exist) + ' Left hand exist: ' + str(left_exist))

# visualize joint coord in 2D space
filename = 'result_2d.jpg'
vis_img = image.copy()[:,:,::-1].transpose(2,0,1)
vis_img = vis_keypoints(vis_img, joint_coord, joint_valid, skeleton, filename, save_path='.')
# visualize joint coord in 3D space
# The 3D coordinate in here consists of x,y pixel and z root-relative depth.
# To make x,y, and z in real unit (e.g., mm), you need to know camera intrincis and root depth.
# The root depth can be obtained from RootNet (https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE)
filename = 'result_3d.jpg'
vis_3d_keypoints(joint_coord, joint_valid, skeleton, filename)
print(len(skeleton))