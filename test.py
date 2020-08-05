from __future__ import print_function
import argparse, random, json, random, os, glob, sys, time, pickle
from os.path import join, exists
import pylab as plt
import torch
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
from tqdm import tqdm
from scipy import spatial

from utils.Lidar import Lidar_parse
from utils.naver import Naver_Datasets_yeouido_IMG, Naver_Pangyo_IMG, collate_fn
from utils.utils import *

from utils import retrieval
from utils.pose_estimation import pnp
from models import netvlad
from models.matching import Matching

parser = argparse.ArgumentParser(description='pytorch-NetVlad')
parser.add_argument('--checkpoint', type=str, default = None)
parser.add_argument('--place', type=str, default = 'yeouido')
parser.add_argument('--DB_ROOT', type=str, default = './data/naver')
parser.add_argument('--savePath', type=str, default = './netvlad_checkpoints')
parser.add_argument('--output_json', type=str, default = 'submit_yeouido_pretrained_odometry')
parser.add_argument('--defualt_json', type=str, default = 'submit_example')
parser.add_argument('--top_k', type=int, default = 10 )
parser.add_argument('--pretrained', type=int, default = 1)
parser.add_argument('--cuda', type=int, default = 1)
parser.add_argument('--ensemble', type=int, default = 0)

opt = parser.parse_args()

DB_ROOT = opt.DB_ROOT
place=opt.place
pretrained = opt.pretrained
cuda = opt.pretrained
savePath = opt.savePath

img_dir = '{}/train/images/left'.format(place)
img_list_txt = "{}_images_list_total.txt".format(place)

if opt.ensemble:
    pickle_name = 'Total_{}_knn_pickle_ensemble'.format(place)
    DB_cache_name = 'Total_{}_DB_cache_ensemble.hdf5'.format(place)
else:
    pickle_name = 'Total_{}_knn_pickle'.format(place)
    DB_cache_name = 'Total_{}_DB_cache.hdf5'.format(place)

file_path = join(DB_ROOT, "{}.json".format(opt.defualt_json))
out_path = join(DB_ROOT, "{}.json".format(opt.output_json))

file_list = sorted(glob.glob(join( DB_ROOT, place, 'test/{}*/'.format(place) )))
odmetry_list = sorted(glob.glob(join( DB_ROOT, place, 'test/**/odometry.txt' )))



if __name__ == "__main__":

    start = time.time()

    # load submit file
    json_data = {}

    with open(file_path, "r") as json_file:
        json_data = json.load(json_file)

    with open('config.json', "r") as configfile:
        config = json.load(configfile)

    ret=config['NetVLAD']

    # read camera parameter
    LCam_K, LCam_D, LCam_RT, pose, position = CalibParam(DB_ROOT, place)

    # init Lidar
    Lidar = Lidar_parse(place=place)

    #````````````````````````````````     
    img_list = join(DB_ROOT, img_list_txt)
    images = [e.strip() for e in open(img_list)]
    len_images = len(images)
    #``````````````````````````````````

    gdes = retrieval.global_descriptor(DB_ROOT, file_path, img_list_txt, pretrained=pretrained)

    #``````````````````````````````````

    print("===> Loading Dataset")
    
    if place == 'pangyo' :
        dataset_img = Naver_Pangyo_IMG(input_transform=input_transform)
    elif place == 'yeouido' :
        dataset_img = Naver_Datasets_yeouido_IMG(input_transform=input_transform)
    else :
        print("Dataset Error")
        sys.exit(1)
        
    imgDataLoader = DataLoader(dataset_img, num_workers=ret['workers'], batch_size=ret['cacheBatchSize'], shuffle=False, pin_memory=True)
    print("Done")
    
    device = torch.device("cuda" if cuda else "cpu")

    if opt.checkpoint == None :
        print("Need to checkpoint")
        sys.exit(1)

    else :
        with torch.no_grad():
            model = gdes.NetVLAD(ret, opt)

            if opt.ensemble:
                model1 = gdes.APGeM(config["APGeM"]["Path1"])
                model2 = gdes.APGeM(config["APGeM"]["Path2"])
            
            print("Model load Done")
            
            if opt.ensemble:
                knn = gdes.make_cache( DB_cache_name, pickle_name, model, imgDataLoader, model1, model2 )
            else:   
                knn = gdes.make_cache( DB_cache_name, pickle_name, model, imgDataLoader )
            
            print("===>Predicting")
            
            matching = Matching(config["SuperGlue"]["config"]).eval().to(device)
            
            for i in tqdm(range(len(file_list))):
                
                max_match = 0
                max_index = 0
                max_query = 0
                sub_file_list = sorted(glob.glob(file_list[i]+"*_L.png"))
                
                for tt in range(len(sub_file_list)) :
                    if tt != 0 :
                        continue
                        
                    print(sub_file_list[tt])
                    odmetry = np.loadtxt(os.path.join(odmetry_list[i]),delimiter = ' ').reshape(-1,4,4)
                    
                    query_path = sub_file_list[tt]
                    #import pdb;pdb.set_trace()
                    if opt.ensemble:
                        test_vlad_encoding = gdes.retreival(query_path, model, model1, model2)    
                    else:
                        test_vlad_encoding = gdes.retreival(query_path, model)

                    distances, candidates = knn.kneighbors(test_vlad_encoding.detach().cpu(), opt.top_k, return_distance=True)
                
                    for indexx, candidate in enumerate(candidates) :

                        for ii in range(len(candidate)) :
                            
                            if ii != 0 :
                                continue
                                
                            match_id = candidate[ii]
                            
                            img_candidate_path = os.path.join(DB_ROOT, img_dir, images[match_id])
         
                            query_xy, db_xy, mconf = matching_2d(query_path, img_candidate_path, matching, device, scale=True)
                            
                            if len(query_xy[mconf>0.7]) > max_match :
                                max_match = len(query_xy)
                                max_index = match_id
                                max_qeury = tt
                            elif len(query_xy[mconf>0.7]) == max_match and max_query < tt :
                                max_match = len(query_xy)
                                max_index = match_id
                                max_qeury = tt
                
                query_path = sub_file_list[max_qeury]
                img_candidate_path = os.path.join(DB_ROOT,img_dir,images[max_index])
                query_xy, db_xy, mconf = matching_2d(query_path, img_candidate_path, matching, device, scale=True)
                good_match_3d_point, fileter_query_xy = Lidar.matching_2d_3d(int(images[max_index][:-4]), query_xy, db_xy)

                pred_query_qwxyz, pred_query_xyz, inlier = pnp(good_match_3d_point, fileter_query_xy,pose, int(images[max_index][:-4]),odmetry, max_qeury, LCam_K, LCam_RT)

                if place == 'pangyo' :
                    set_name = file_list[i][25:33]
                elif place == 'yeouido' :
                    set_name = file_list[i][26:35]
                else :
                    print("Dataset Error")
                    sys.exit(1)
            
            
            json_data = dump_submit(json_data, out_path, place, set_name, \
                        query_qwxyz=pred_query_qwxyz, query_xyz=pred_query_xyz)

    
print("Done")
