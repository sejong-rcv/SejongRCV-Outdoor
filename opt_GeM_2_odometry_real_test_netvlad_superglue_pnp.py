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
import matplotlib.cm as cm
from tqdm import tqdm
from PIL import Image
from numpy.linalg import inv
from scipy import spatial
from Lidar import Lidar_parse
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform
from skimage.transform import AffineTransform
from naver import Naver_Datasets_IMG, Naver_Datasets, Test_Naver_Datasets_IMG, collate_fn
from pyquaternion import Quaternion
from models.matching import Matching
from scipy.spatial.transform import Rotation as R
from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)


import dirtorch.nets as nets

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
images_path = os.path.join(DB_ROOT, 'pangyo', 'train', 'images')
lcam_poses = np.loadtxt(os.path.join(images_path, 'poses.txt')).reshape(-1, 4, 4)
calib = json.load(open(os.path.join(DB_ROOT, 'calibration.json'), 'r'))
LCam_K = np.array(calib['Intrinsic']['LCam']['K'])[:,:3]
LCam_D = np.array(calib['Intrinsic']['LCam']['Distortion'])
LCam_RT = np.array(calib['Extrinsic']['LCam'])
pose = np.load('./pose_total.npy')

###################
#Super glue
resize_outdoor = [-1]
resize_float = True
rot0, rot1 = 0,0

###########################

root = "data/naver"
img_dir = 'pangyo/train/images/left'
und_img_dir = 'pangyo/train/images/left_undistort'
img_list_txt = "image_list_total.txt"

position_db = np.load('./pangyo_position_DB.npy')
position_test = np.load('./pangyo_position_test.npy')
pose_db = np.load('./pangyo_pose_DB.npy')
pose_test = np.load('./pangyo_pose_Test.npy')

pickle_name = 'GeM_2_Pretrained_Total_knn_pickle'
DB_cache_name = 'GeM_2_Pretrained_Total_DB_cache.hdf5'
top_k = 20

file_list = sorted(glob.glob("./data/naver/pangyo/test/pangyo*/"))
odmetry_list = sorted(glob.glob("./data/naver/pangyo/test/**/odometry.txt"))

file_path = "data/naver/submit_example.json"
out_path = "data/naver/submit_pangyo_Pretrained_odometry_top20_full_GeM_2_2048_opt_2_easy_Lidar.json"
#[15,16,18,19,36,38,39,43]


# # Naver
# RGB_mean = [0.389, 0.393, 0.392]
# RGB_std  = [0.280, 0.285, 0.291]

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
    
    success,solvR,solvt,inlierR = cv2.solvePnPRansac(points_3D.astype("float64"), \
                                                points_2D.astype("float64"), \
                                                LCam_K.astype("float64"), \
                                                distCoeffs.astype("float64"), \
                                                iterationsCount=100000, \
                                                useExtrinsicGuess = True, \
                                                confidence = 0.999, \
                                                reprojectionError = 8, \
                                                flags = cv2.SOLVEPNP_AP3P)

    if inlierR is None or inlierR.shape[0] <= 4:
        GT_qwxyz = Quaternion(matrix=pose[index][:3,:3]).elements
        GT_xyz = pose[index][:3,3]
        print("fail")
        return GT_qwxyz, GT_xyz, None
    
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
        transforms.Normalize(mean=[0.389, 0.393, 0.392],
                               std=[0.280, 0.285, 0.291]),
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
        
    with torch.no_grad():
        
        #match_txt = open("./match_check_mscore.txt",'w')

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

            print("NetVLAD Model Load Done")

            ################################################ GeM
            GeM = nets.create_model("resnet101_rmac",pretrained='./save/Resnet101-AP-GeM-LM18.pt',without_fc=False)
            GeM = GeM.to(device)
            GeM.eval()

            GeM_2 = nets.create_model("resnet101_rmac",pretrained='./save/Resnet-101-AP-GeM.pt',without_fc=False)
            GeM_2 = GeM_2.to(device)
            GeM_2.eval()

            print("GeM Model load Done")
            ################################################

            print("===> Loading Dataset")
            dataset = Naver_Datasets(input_transform=input_transform)
            dataset_img = Naver_Datasets_IMG(input_transform=input_transform)
            test_dataset_img = Naver_Datasets_IMG(input_transform=input_transform)
            imgDataLoader = DataLoader(dataset_img, num_workers=workers, batch_size=cacheBatchSize, shuffle=False, pin_memory=True)
            test_imgDataLoader = DataLoader(test_dataset_img, num_workers=workers, batch_size=cacheBatchSize, shuffle=False, pin_memory=True)
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
                                GeM_encoding = GeM(input)
                                GeM_2_encoding = GeM_2(input)
                                image_encoding = model.encoder(input)
                                vlad_encoding = model.pool(image_encoding)
                                total_encoding = torch.cat((vlad_encoding, GeM_encoding,GeM_2_encoding), dim=1)
                                total_encoding = F.normalize(total_encoding, p=2, dim=1)
                                DBfeature[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()

                                del input, image_encoding, vlad_encoding, total_encoding

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
                    'sinkhorn_iterations': 30,
                    'match_threshold': 0.2,
                }
            }
            matching = Matching(config).eval().to(device)
            
            for i in tqdm(range(len(file_list))):
                
                # good retrival
                if i in [5,6,8,11,15,16,17,18,19,23,26,28,29,32,36,37,38,39,41,42,43,44,45,46,49]:
                   continue
                
                # if i not in [15,16,18,19,36,38,39,43] :
                #     continue
                # if i not in wwwww :
                #     continue
                # check 
                # 
                
                #good retrival
                # [0,1,2,3,4,7,9,10,12,13,14,20,21,22,24,25,27,30,31,33,34,35,40,47,48]
                
                # boder retrival
                # [15,16,18,19,36,38,39,43]
                
                # bad retrival
                # [5,6,8,11,15,17,23,26,28,29,32,37,41,42,45,46,49]
                
                pnp_flag = 0
                max_match = 0
                max_index = 0
                max_query = 0
                max_inliner = 0
                sub_file_list = sorted(glob.glob(file_list[i]+"0*_L.png"))

                for tt in range(len(sub_file_list)) :
                    
                    if pnp_flag == 1 :
                        break
                        
                    print(sub_file_list[49-tt])
                    odmetry = np.loadtxt(os.path.join(odmetry_list[i]),delimiter = ' ').reshape(-1,4,4)
                    query_path = sub_file_list[49-tt]
                    img = Image.open(query_path)
                    img = input_tr(img)
                    img = img.to(device)
                    test_vlad_encoding = model.pool(model.encoder(img.unsqueeze(0)))
                    test_GeM_encoding = GeM(img.unsqueeze(0)).unsqueeze(0)
                    test_GeM_2_encoding = GeM_2(img.unsqueeze(0)).unsqueeze(0)

                    test_total_encoding = torch.cat((test_vlad_encoding, test_GeM_encoding,test_GeM_2_encoding), dim=1)
                    test_total_encoding = F.normalize(test_total_encoding, p=2, dim=1)

                    distances, candidates = knn.kneighbors(test_vlad_encoding.detach().cpu(), top_k, return_distance=True)

                    for indexx, candidate in enumerate(candidates) :

                        for ii in range(len(candidate)) :
                            match_id = candidate[ii]

                            img_candidate_path = os.path.join(root,img_dir,images[match_id])

                            image0, inp0, scales0 = read_image(query_path, device, resize_outdoor, rot0, resize_float)
                            image1, inp1, scales1 = read_image(img_candidate_path, device, resize_outdoor, rot1, resize_float)

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
                            mconf = conf[valid]
                            
                            query_xy = mkpts0*scales0
                            db_xy = mkpts1*scales1
                            uv, cam_pnts, veh_pose, merge_pnts = Lidar.parsing(int(images[match_id][:-4]))
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

                            pred_query_qwxyz,pred_query_xyz,inlier = pnp(good_match_3d_point, fileter_query_xy,int(images[match_id][:-4]),odmetry,49-tt)
                            
                            if inlier is not None :
                                # if len(inlier) > 100 :
                                #     set_name = file_list[i][25:33]
                                #     json_data['pangyo'][set_name]['qw'] = pred_query_qwxyz[0]
                                #     json_data['pangyo'][set_name]['qx'] = pred_query_qwxyz[1]
                                #     json_data['pangyo'][set_name]['qy'] = pred_query_qwxyz[2]
                                #     json_data['pangyo'][set_name]['qz'] = pred_query_qwxyz[3]
                                #     json_data['pangyo'][set_name]['x'] = pred_query_xyz[0]
                                #     json_data['pangyo'][set_name]['y'] = pred_query_xyz[1]
                                #     json_data['pangyo'][set_name]['z'] = pred_query_xyz[2]
                                #     pnp_flag = 1
                                #     break
                                if max_inliner < len(inlier) :
                                    max_inliner = len(inlier)
                                    max_match = pred_query_qwxyz
                                    max_index = pred_query_xyz
                                    
                            if ii >= len(candidate)-1 :
                                if max_inliner > 100 :
                                    set_name = file_list[i][25:33]
                                    json_data['pangyo'][set_name]['qw'] = max_match[0]
                                    json_data['pangyo'][set_name]['qx'] = max_match[1]
                                    json_data['pangyo'][set_name]['qy'] = max_match[2]
                                    json_data['pangyo'][set_name]['qz'] = max_match[3]
                                    json_data['pangyo'][set_name]['x'] = max_index[0]
                                    json_data['pangyo'][set_name]['y'] = max_index[1]
                                    json_data['pangyo'][set_name]['z'] = max_index[2]
                                    pnp_flag = 1
                                    break
                            
                    if tt >= 49 :
                        set_name = file_list[i][25:33]
                        json_data['pangyo'][set_name]['qw'] = max_match[0]
                        json_data['pangyo'][set_name]['qx'] = max_match[1]
                        json_data['pangyo'][set_name]['qy'] = max_match[2]
                        json_data['pangyo'][set_name]['qz'] = max_match[3]
                        json_data['pangyo'][set_name]['x'] = max_index[0]
                        json_data['pangyo'][set_name]['y'] = max_index[1]
                        json_data['pangyo'][set_name]['z'] = max_index[2]
                                

with open(out_path, 'w') as outfile:
    json.dump(json_data, outfile, indent=4)
    
print("Done")
