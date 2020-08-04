import os
import json
import numpy as np
import torchvision.transforms as transforms
import torch
import torch.nn as nn



from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)



def matching_2d(query_path, img_candidate_path, matching, device, scale=False):

    resize_outdoor = [-1]
    resize_float = True
    rot0, rot1 = 0,0

    image0, inp0, scales0 = read_image(
        query_path, device, resize_outdoor, rot0, resize_float)
    image1, inp1, scales1 = read_image(
        img_candidate_path, device, resize_outdoor, rot1, resize_float)

    # if image0 is None or image1 is None:
    #     print('Problem reading image pair: {} {}'.format(
    #         input_dir/name0, input_dir/name1))
    #     exit(1)

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

    if scale:
        query_xy = mkpts0*scales0
        db_xy = mkpts1*scales1

        

        return query_xy, db_xy, mconf

    

    return mkpts0, mkpts1


def dump_submit(submit, out_path,  place, test_number, query_qwxyz=None, query_xyz=None, replace=False, quick=False):
    # submit json 파일, query_qwxyz, query_xyz

    #import pdb;pdb.set_trace()
    #test_number='pangyo{}'.format('%02d'%test_number)

    # qwxyz=np.array([float(best_result[place][test_number]['qw']), float(best_result[place][test_number]['qx']), \
    #     float(best_result[place][test_number]['qy']), float(best_result[place][test_number]['qz'])])
    # result_xyz = np.array([float(best_result[place][test_number]['x']), float(best_result[place][test_number]['y']), float(best_result[place][test_number]['z'])])
    

    # if replace:
    #     print("\n!!!!!!!replace!!!!!!!!\n")
    #     submit[place][test_number]['qw']=qwxyz[0]
    #     submit[place][test_number]['qx']=qwxyz[1]
    #     submit[place][test_number]['qy']=qwxyz[2]
    #     submit[place][test_number]['qz']=qwxyz[3]
    #     submit[place][test_number]['x']=result_xyz[0]
    #     submit[place][test_number]['y']=result_xyz[1]
    #     submit[place][test_number]['z']=result_xyz[2]

    #     if test_number=='{}49'.format(place):

    #         json.dumps(submit, ensure_ascii=False, indent="\t")

    #         with open(out_path, 'w') as make_file:
    #             json.dump(submit, make_file, ensure_ascii=False, indent="\t")

    #     return submit


    submit[place][test_number]['qw']=query_qwxyz[0]
    submit[place][test_number]['qx']=query_qwxyz[1]
    submit[place][test_number]['qy']=query_qwxyz[2]
    submit[place][test_number]['qz']=query_qwxyz[3]
    submit[place][test_number]['x']=query_xyz[0]
    submit[place][test_number]['y']=query_xyz[1]
    submit[place][test_number]['z']=query_xyz[2]

    
    #print( test_number, ' error qua : ', query_qwxyz - qwxyz, 'error xyz', query_xyz - result_xyz, '\n' )

    if test_number=='{}49'.format(place) or quick:

        json.dumps(submit, ensure_ascii=False, indent="\t")

        with open(out_path, 'w') as make_file:
            json.dump(submit, make_file, ensure_ascii=False, indent="\t")

    return submit


def CalibParam(DB_ROOT, place='yeuido'):

    images_path = os.path.join(DB_ROOT, place, 'train', 'images')
    lcam_poses = np.loadtxt(os.path.join(images_path, 'poses.txt')).reshape(-1, 4, 4)
    calib = json.load(open(os.path.join(DB_ROOT, 'calibration.json'), 'r'))
    LCam_K = np.array(calib['Intrinsic']['LCam']['K'])[:,:3]
    LCam_D = np.array(calib['Intrinsic']['LCam']['Distortion'])
    LCam_RT = np.array(calib['Extrinsic']['LCam'])
    pose = np.load('./data/{}_pose_total.npy'.format(place))
    position = np.load('./data/{}_position_total.npy'.format(place))

    return LCam_K, LCam_D, LCam_RT, pose, position



def input_transform():
    return transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.368, 0.378, 0.383],
                               std=[0.28,0.294,0.311]),
    ])


def load_pretrained_layers(model, path) :

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

