from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
from numpy.linalg import inv
import numpy as np
import cv2


def make_odometry(pose, odometry, start=0):
    if start == 49 :
        return pose
    last_odometry=odometry[start]
    for odo in odometry[start+1:]:
        last_odometry=last_odometry@odo
    return pose@last_odometry

# 쿼터니언, cv2, 
def pnp(points_3D, points_2D, pose, index, odometry, start, LCam_K, LCam_RT):

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
    
    predict_pose = make_odometry(predict_pose,odometry, start=start)@inv(LCam_RT)
    #@inv(LCam_RT)
    ro = R.from_matrix(predict_pose[:3,:3])
    
    #Quaternion(matrix=make_odometry(predict_pose,odometry)[:3,:3])
    query_qwxyz = np.r_[ro.as_quat()[-1], ro.as_quat()[:-1]]
    #query_qwxyz = Quaternion(matrix=predict_pose[:3,:3]).elements
    query_xyz = predict_pose[:3,3]
    
    return query_qwxyz, query_xyz, inlierR