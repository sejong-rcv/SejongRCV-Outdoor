from __future__ import print_function
import argparse
from math import log10, ceil
import random, shutil, json
import os
from os.path import join, exists, isfile, realpath, dirname
from os import makedirs, remove, chdir, environ
import ujson as json
import pylab as plt
import glob
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pylab as plt
from torch.autograd import Variable
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.dataset import Subset
from sklearn.neighbors import NearestNeighbors
import torchvision.transforms as transforms
from PIL import Image
from datetime import datetime
import torchvision.datasets as datasets
import torchvision.models as models
import h5py
import pdb
import numpy as np
import netvlad
import faiss
import sys
import time
import pickle
import cv2
from tqdm import tqdm
from PIL import Image
from numpy.linalg import inv
from scipy import spatial
from Lidar_yeouido import Lidar_parse
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform
from skimage.transform import AffineTransform
from scipy.spatial.transform import Rotation as R
from naver import Naver_Datasets_yeouido_IMG, yeouido_Naver_Datasets, Test_Naver_Datasets_IMG, collate_fn
from pyquaternion import Quaternion
from models.matching import Matching
from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)

parser = argparse.ArgumentParser(description='pytorch-NetVlad')
parser.add_argument('--checkpoint', type=str, default = None)

DB_ROOT = './data/naver'
cacheRefreshRate = 100
cacheBatchSize = 20
batchSize = 8
workers = 8
pretrained = True
cuda = True
resume = ''
num_clusters = 64
optima_str = "SGD"
savePath = './netvlad_checkpoints'
evalEvery = 10
seed = 9
lr = 0.0001
momentum = 0.9
weightDecay = 0.001 
lrStep = 5
lrGamma = 0.5
start_epoch = 0
nEpochs = 30
margin = 0.1
################### PNP
images_path = os.path.join(DB_ROOT, 'yeouido', 'train', 'images')
lcam_poses = np.loadtxt(os.path.join(images_path, 'poses.txt')).reshape(-1, 4, 4)
calib = json.load(open(os.path.join(DB_ROOT, 'calibration.json'), 'r'))
LCam_K = np.array(calib['Intrinsic']['LCam']['K'])[:,:3]
LCam_D = np.array(calib['Intrinsic']['LCam']['Distortion'])
LCam_RT = np.array(calib['Extrinsic']['LCam'])
pose = np.load('./yeouido_pose_total.npy')
position = np.load('./yeouido_position_total.npy')

###################
#Super glue
resize_outdoor = [-1]
resize_float = True
rot0, rot1 = 0,0

####################
root = "data/naver"
img_dir = 'yeouido/train/images/left'
und_img_dir = 'yeouido/train/images/left_undistort'
img_list_txt = "yeouido_images_list_total.txt"

pickle_name = 'Total_yeouido_knn_pickle'
DB_cache_name = 'Total_yeouido_DB_cache.hdf5'
top_k = 50

file_path = "data/naver/submit_example.json"
out_path = "data/naver/submit_yeouido_pretrained_odometry.json"
    
file_list = sorted(glob.glob("./data/naver/yeouido/test/yeouido*/"))
odmetry_list = sorted(glob.glob("./data/naver/yeouido/test/**/odometry.txt"))

#############################################


# # Naver_yeouido
# RGB_mean = [0.368, 0.378, 0.383]
# RGB_std  = [0.28,0.294,0.311]

Lidar = Lidar_parse()

def make_odometry(pose, odometry, start=0):
    if start == 49 :
        return pose
    last_odometry=odometry[start]
    for odo in odometry[start+1:]:
        last_odometry=last_odometry@odo
    return pose@last_odometry

def pnp(points_3D, points_2D, index, odometry, start):

    distCoeffs = np.zeros((4, 1), dtype='float32')
    #distCoeffs = LCam_D

    assert points_3D.shape[0] == points_2D.shape[0], 'points 3D and points 2D must have same number of vertices'
    
    if points_3D.shape[0] <= 4:
        GT_qwxyz = Quaternion(matrix=pose[index][:3,:3]).elements
        GT_xyz = pose[index][:3,3]
        print("fail")
        return GT_qwxyz, GT_xyz, None
    
    _,solvR,solvt,inlierR = cv2.solvePnPRansac(points_3D.astype("float64"), \
                                                points_2D.astype("float64"), \
                                                LCam_K.astype("float64"), \
                                                distCoeffs.astype("float64"), \
                                                iterationsCount=100000, \
                                                useExtrinsicGuess = True, \
                                                confidence = 0.999, \
                                                reprojectionError = 8, \
                                                flags = cv2.SOLVEPNP_AP3P)
    
    solvRR,_ = cv2.Rodrigues(solvR)
    solvRR_inv = np.linalg.inv(solvRR)
    solvtt = -np.matmul(solvRR_inv,solvt)
    rot = cv2.Rodrigues(solvRR_inv)[0].squeeze(1)
    predict_pose = np.r_[np.c_[solvRR_inv,solvtt],np.array([[0,0,0,1]])]
    
    predict_pose = make_odometry(predict_pose,odometry,start=start)@inv(LCam_RT)
    #@inv(LCam_RT)
    ro = R.from_matrix(predict_pose[:3,:3])
    
    #Quaternion(matrix=make_odometry(predict_pose,odometry)[:3,:3])
    query_qwxyz = np.r_[ro.as_quat()[-1], ro.as_quat()[:-1]]
    #query_qwxyz = Quaternion(matrix=predict_pose[:3,:3]).elements
    query_xyz = predict_pose[:3,3]
    
    return query_qwxyz, query_xyz, inlierR

def input_transform():
    return transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.368, 0.378, 0.383],
                               std=[0.28,0.294,0.311]),
    ])

def load_pretrained_layers(model,path) :

    state_dict = model.state_dict()
    param_names = list(state_dict.keys())  

    # load checkpoint
    pretrained_base_state_dict = torch.load(path)['state_dict']
    pretrained_base_state_dict_name = list(pretrained_base_state_dict.keys())

    # Transfer conv. parameters from pretrained model to current model
    for i, param in enumerate(param_names[:]):
        state_dict[param] = pretrained_base_state_dict[pretrained_base_state_dict_name[i]]

    model.load_state_dict(state_dict)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)


if __name__ == "__main__":

    start = time.time()

    json_data = {}
    with open(file_path, "r") as json_file:
        json_data = json.load(json_file)
    
    img_list = join(root, img_list_txt)
    images = [e.strip() for e in open(img_list)]
    len_images = len(images)

    opt = parser.parse_args()

    if opt.checkpoint == None :
        print("Need to checkpoint")
        sys.exit(1)

    else :
        
        input_tr = input_transform()
        device = torch.device("cuda" if cuda else "cpu")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if cuda:
            torch.cuda.manual_seed(seed)

        encoder_dim = 512
        encoder = models.vgg16(pretrained=pretrained)
        # capture only feature part and remove last relu and maxpool
        layers = list(encoder.features.children())[:-2]

        if pretrained:
            # if using pretrained then only train conv5_1, conv5_2, and conv5_3
            for l in layers[:-5]: 
                for p in l.parameters():
                    p.requires_grad = False

        encoder = nn.Sequential(*layers)
        model = nn.Module() 
        model.add_module('encoder', encoder)
        net_vlad = netvlad.NetVLAD(num_clusters=num_clusters, dim=encoder_dim)
        model.add_module('pool', net_vlad)
        load_pretrained_layers(model,opt.checkpoint)
        model = model.to(device)
        
        model.eval()
        
        print("Model load Done")

        print("===> Loading Dataset")
        dataset = yeouido_Naver_Datasets(input_transform=input_transform)
        dataset_img = Naver_Datasets_yeouido_IMG(input_transform=input_transform)
        imgDataLoader = DataLoader(dataset_img, num_workers=workers, batch_size=cacheBatchSize, shuffle=False, pin_memory=True)

        print("Done")

        if not exists(join(DB_ROOT, 'centroids', pickle_name)) :

            print("===> Loading Cache")
            if not exists(join(DB_ROOT, 'centroids', DB_cache_name)) :
                print("===> Making Cache")
                dataset.cache = join(DB_ROOT, 'centroids', DB_cache_name)
                with h5py.File(dataset.cache, mode='w') as h5: 
                    pool_size = encoder_dim * num_clusters
                    DBfeature = h5.create_dataset("features", 
                            [len_images, pool_size], 
                            dtype=np.float32)
                    
                    with torch.no_grad():
                        for iteration, (input, indices) in enumerate(tqdm(imgDataLoader), 1):
                            input = input.to(device)
                            image_encoding = model.encoder(input)
                            vlad_encoding = model.pool(image_encoding) 
                            DBfeature[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()
                            del input, image_encoding, vlad_encoding

                h5 =  h5py.File(join(DB_ROOT, 'centroids', DB_cache_name), mode='r')
                DBfeature = h5.get('features')
                del h5

            else :
                h5 =  h5py.File(join(DB_ROOT, 'centroids', DB_cache_name), mode='r')
                DBfeature = h5.get('features')
                del h5
            print("Done")      
                    
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(DBfeature)
            knnPickle = open(join(DB_ROOT, 'centroids', pickle_name),'wb')
            pickle.dump(knn, knnPickle, protocol=4)  

            print("Restart for using pickle")
            sys.exit(0)
        
        else :
            knn = pickle.load(open(join(DB_ROOT, 'centroids', pickle_name), 'rb'))
            print("Load Done")
        
        
        print("===>Predicting")
        
        config = {
            'superpoint': {
                'nms_radius': 3,
                'keypoint_threshold': 0.005,
                'max_keypoints': 2048
            },
            'superglue': {
                'weights': 'outdoor',
                'sinkhorn_iterations': 20,
                'match_threshold': 0.2,
            }
        }
        matching = Matching(config).eval().to(device)
        
        for i in tqdm(range(len(file_list))):
            max_match = 0
            max_index = 0
            max_query = 0
            sub_file_list = sorted(glob.glob(file_list[i]+"*_L.png"))
            
            for tt in range(len(sub_file_list)) :
                print(sub_file_list[tt])
                odmetry = np.loadtxt(os.path.join(odmetry_list[i]),delimiter = ' ').reshape(-1,4,4)
                
                query_path = sub_file_list[tt]

                img = Image.open(query_path)
                img = input_tr(img)
                img = img.to(device)
                test_vlad_encoding = model.pool(model.encoder(img.unsqueeze(0))).cpu().detach().numpy()
                distances, candidates = knn.kneighbors(test_vlad_encoding, top_k, return_distance=True)
            
                for indexx, candidate in enumerate(candidates) :

                    for ii in range(len(candidate)) :
                    
                        match_id = candidate[ii]


                        img_candidate_path = os.path.join(root,img_dir,images[match_id])

                        image0, inp0, scales0 = read_image(
                            query_path, device, resize_outdoor, rot0, resize_float)
                        image1, inp1, scales1 = read_image(
                           img_candidate_path, device, resize_outdoor, rot1, resize_float)

                        if image0 is None or image1 is None:
                            print('Problem reading image pair: {} {}'.format(
                                input_dir/name0, input_dir/name1))
                            exit(1)

                        if image0.shape != image1.shape :
                            print('Size of image is not same')
                            exit(1)

                        pred = matching({'image0': inp0, 'image1': inp1})
                        pred = {k: v[0].cpu().detach().numpy() for k, v in pred.items()}
                        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
                        matches, conf = pred['matches0'], pred['matching_scores0']

                        valid = matches > -1
                        mkpts0 = kpts0[valid]
                        mkpts1 = kpts1[matches[valid]]
                    
                        if len(mkpts0) > max_match :
                            max_match = len(mkpts0)
                            max_index = match_id
                            max_qeury = tt
                        elif len(mkpts0) == max_match and max_query < tt :
                            max_match = len(mkpts0)
                            max_index = match_id
                            max_qeury = tt
                        
            query_path = sub_file_list[max_qeury]
            img_candidate_path = os.path.join(root,img_dir,images[max_index])

            image0, inp0, scales0 = read_image(
                query_path, device, resize_outdoor, rot0, resize_float)
            image1, inp1, scales1 = read_image(
               img_candidate_path, device, resize_outdoor, rot1, resize_float)

            if image0 is None or image1 is None:
                print('Problem reading image pair')
                exit(1)

            if image0.shape != image1.shape :
                print('Size of image is not same')
                exit(1)

            pred = matching({'image0': inp0, 'image1': inp1})
            pred = {k: v[0].cpu().detach().numpy() for k, v in pred.items()}
            kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
            matches, conf = pred['matches0'], pred['matching_scores0']

            # Keep the matching keypoints.
            valid = matches > -1
            mkpts0 = kpts0[valid]
            mkpts1 = kpts1[matches[valid]]
            mconf = conf[valid]

            query_xy = mkpts0*scales0
            db_xy = mkpts1*scales1

            uv, cam_pnts, veh_pose, merge_pnts = Lidar.parsing(int(images[max_index][:-4]))
            tree = spatial.KDTree(uv[:,:2])

            out_index = []
            fileter_db_xy = []
            fileter_query_xy = []

            for idx in range(len(query_xy)) :
                distance_tree, index = tree.query(db_xy[idx])
                if distance_tree < 1 :
                    out_index.append(index)
                    fileter_db_xy.append(db_xy[idx])
                    fileter_query_xy.append(query_xy[idx])

            fileter_db_xy = np.asarray(fileter_db_xy)
            fileter_query_xy = np.asarray(fileter_query_xy)

            good_match_3d_point = merge_pnts[out_index,:][:,:3]

            pred_query_qwxyz,pred_query_xyz,inlier = pnp(good_match_3d_point, fileter_query_xy,int(images[max_index][:-4]),odmetry,max_qeury)

            print(max_qeury)

            set_name = file_list[i][26:35]
            json_data['yeouido'][set_name]['qw'] = pred_query_qwxyz[0]
            json_data['yeouido'][set_name]['qx'] = pred_query_qwxyz[1]
            json_data['yeouido'][set_name]['qy'] = pred_query_qwxyz[2]
            json_data['yeouido'][set_name]['qz'] = pred_query_qwxyz[3]
            json_data['yeouido'][set_name]['x'] = pred_query_xyz[0]
            json_data['yeouido'][set_name]['y'] = pred_query_xyz[1]
            json_data['yeouido'][set_name]['z'] = pred_query_xyz[2]
        
with open(out_path, 'w') as outfile:
    json.dump(json_data, outfile, indent=4)
    
print("Done")
