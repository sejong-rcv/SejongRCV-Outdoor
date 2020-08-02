import pdb
import numpy as np
import os
import cv2 
import ujson as json
import pylab as plt

from PIL import Image
from numpy.linalg import inv
from scipy import spatial

#####################################################################################

class Lidar_parse :

    def __init__(self) :

        # set the dataset root directory
        DATA_ROOT = 'data/naver'
        calib = json.load(open(os.path.join(DATA_ROOT, 'calibration.json'), 'r'))

        self.images_path = os.path.join(DATA_ROOT, 'pangyo', 'train', 'images')
        self.lcam_poses = np.loadtxt(os.path.join(self.images_path, 'poses.txt')).reshape(-1, 4, 4)
        self.lcam_timestamps = np.loadtxt(os.path.join(self.images_path,'timestamps.txt'))
        #print('We have %d left images w/ poses'%(len(self.lcam_poses)))

        self.lidar_path = os.path.join(DATA_ROOT, 'pangyo', 'train', 'lidars')
        self.lidar_poses = np.loadtxt(os.path.join(self.lidar_path, 'poses.txt')).reshape(-1, 4, 4)
        self.lidar_timestamps = np.loadtxt(os.path.join(self.lidar_path, 'timestamps.txt'))
        #print('We have %d lidar w/ poses'%(len(self.lidar_poses)))

        self.LCam_RT = np.array(calib['Extrinsic']['LCam'])
        self.LCam_K = np.array(calib['Intrinsic']['LCam']['K'])
        self.LiDAR_RT = np.array(calib['Extrinsic']['Lidar'])

    def get_image(self, idx, sensor='left'):
        return cv2.imread(os.path.join(self.images_path, sensor, '%06d.png'%idx))[:,:,::-1]

    def get_lidar(self,idx):
        pnts = np.load(os.path.join(self.lidar_path, '%05d.npy'%idx))
        invalid = np.logical_and(np.logical_and(pnts[:,0] == 0, pnts[:,1] == 0), pnts[:,2] == 0)
        return pnts[~invalid]

    def slow_and_naive_hidden_point_removal(self,in_uv, in_cam_pnts, in_merged_pnts):
        depth = in_cam_pnts[:,2]
        depth_order = np.argsort(depth)
        uv_int = in_uv.astype(int)[:,:2]
        nearest_idx = dict()
        for pnt_idx in depth_order:
            pnt_uv = tuple(uv_int[pnt_idx])
            if nearest_idx.get(pnt_uv) is None:
                nearest_idx[pnt_uv] = pnt_idx
        nearest_idx = np.array(list(nearest_idx.values()))
        cam_pnts_visible = in_cam_pnts[nearest_idx]
        uv_visible = in_uv[nearest_idx]
        merged_pnts_visible = in_merged_pnts[nearest_idx]

        return uv_visible, cam_pnts_visible, merged_pnts_visible

    def parsing(self, idx) :

        img_idx = idx

        img = self.get_image(img_idx)
        img_ts = self.lcam_timestamps[img_idx]
        veh_pose = self.lcam_poses[img_idx]

        merged_pnts = []
        # LiDAR indices captured within {stack_range} secs from image timestamp
        stack_ts_range = 3.0
        lidar_indices = np.where(np.logical_and(self.lidar_timestamps > img_ts-stack_ts_range, self.lidar_timestamps < img_ts+stack_ts_range))[0]
        #print('Stacking %d consecutive LiDAR frames around %f..'%(len(lidar_indices), img_ts))

        for lidar_idx in lidar_indices:
            pnts = self.get_lidar(lidar_idx)
            lidar_pose = self.lidar_poses[lidar_idx]
            pnts_m = pnts[:,:3]/100.0
            pnts_m = np.c_[pnts_m, np.ones([len(pnts_m), 1])]
            # transform LiDAR frame points to world frame
            transformed_pnts = (lidar_pose @ self.LiDAR_RT @ pnts_m.T).T
            merged_pnts.append(transformed_pnts)

        merged_pnts = np.concatenate(merged_pnts, axis=0)

        #print('Projecting world frame points onto the image..')

        cam_pnts = (inv(self.LCam_RT) @ inv(veh_pose) @ merged_pnts.T).T
        prj_pnts = (self.LCam_K @ cam_pnts.T).T
        uv = prj_pnts / prj_pnts[:,2][:,None]

        # ignore points closer than 5 meters
        front = cam_pnts[:,2] > 5
        uv = uv[front]
        cam_pnts=cam_pnts[front]
        merged_pnts= merged_pnts[front]

        # ignore points further than 50 meters
        further = cam_pnts[:,2] < 150
        uv = uv[further]
        cam_pnts=cam_pnts[further]
        merged_pnts= merged_pnts[further]
        
        # ignore points outside of image frame
        inframe = np.logical_and(np.logical_and(uv[:,0] > 0, uv[:,1] > 0),
                                np.logical_and(uv[:,0] < img.shape[1], uv[:,1] < img.shape[0]))
        uv = uv[inframe]
        cam_pnts=cam_pnts[inframe]
        merged_pnts=merged_pnts[inframe]
        #print('Now we have %d points projected on the image (note that we are ignoring visibility).'%len(uv))

        uv, cam_pnts, merged_pnts = self.slow_and_naive_hidden_point_removal(uv, cam_pnts, merged_pnts)

        return uv, cam_pnts, veh_pose, merged_pnts

