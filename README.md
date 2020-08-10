# SejongRCV-Outdoor
[NAVER LABS Mapping &amp; Localization Challenge](challenge.naverlabs.com/leaderboard)

![image](https://user-images.githubusercontent.com/44772344/89431605-203c6680-d77b-11ea-9107-1093d311e3d4.png)




### How to run

##### 해당 코드는 추가적인 학습 코드를 제공하지 않으며, [NetVLAD](https://github.com/Nanne/pytorch-NetVlad)에서 제공하는 피츠버그 데이터셋 Pretrained Model을 사용하고 있습니다. 

#### test
```
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=(GPU_NUM) python test.py --checkpoint (checkpoint_path) --place [pangyo/yeouido] --top_k 10 --DB_ROOT (DB_ROOT)
```

### Library
> numpy </br>
> pytorch </br>
> sickitlearn </br>
> pyquternion </br>

### Config.json
> NetVLAD, SuperGlue를 설정할 수 있으며, 추가로 APGeM이 앙상블된 모델도 옵션으로 수정 가능합니다. (챌린지에서는 NetVLAD와 SueprGlue만 사용)
```
{   
    "SuperGlue":
    {
        
        "config":
        {   
            "superpoint":
            {   
                "nms_radius": 3,
                "keypoint_threshold": 0.005,
                "max_keypoints": 2048
            },
            
            "superglue":
            {   
                "weights": "outdoor",
                "sinkhorn_iterations": 20,
                "match_threshold": 0.2
            }
        }
    },
    
    "NetVLAD" :
    {   
        "cacheRefreshRate" : 100,
        "cacheBatchSize" : 20,
        "batchSize" : 8,
        "workers" : 8,
        
        "resume" : "", 
        "num_clusters" : 64,
        "optima_str" : "SGD",
        
        "encoder_dim" : 512,
        
        "evalEvery" : 10,
        "seed" : 9,
        "lr" : 0.0001,
        "momentum" : 0.9,
        "weightDecay" : 0.001 ,
        "lrStep" : 5,
        "lrGamma" : 0.5,
        "start_epoch" : 0,
        "nEpochs" : 30,
        "margin" : 0.1
    },
    
    "APGeM":
    {   
        "Path1" : "../jobs/weight/Resnet101-AP-GeM-LM18.pt",
        "Path2" : "../jobs/weight/Resnet-101-AP-GeM.pt"
    }

}

```
### Data Tree
```
+-- data
|   +-- pangyo_pose_total.npy  pangyo_position_total.npy  yeouido_pose_total.npy  yeouido_position_total.npy
|   +-- naver
|       +-- submit_json.json
|       +-- pangyo_images_list_total.txt yeouido_images_list_total.txt
|       +-- centriods
|           +-- Total_pangyo_DB_cache.hdf5
|           +-- Total_pangyo_knn_pickle
|           +-- Total_yeouido_DB_cache.hdf5
|           +-- Total_yeouido_knn_pickle
|       +-- yeouido
|           +-- train
|               +-- images
|                   +-- left
|                       +-- 000000.png
|                       +-- ...
|                   +-- right
|                       +-- 000000.png
|                       +-- ...
|                   +-- poses.txt
|                   +-- timestamps.txt
|               +-- lidars
|           +-- test
|               +-- yeouido00
|                   +-- 00_L.png
|                   +-- 00_R.png
|                   +-- ...
|               +-- yeouido**
|       +-- pangyo
|           +-- train
|               +-- images
|                   +-- left
|                       +-- 000000.png
|                       +-- ...
|                   +-- right
|                       +-- 000000.png
|                       +-- ...
|                   +-- poses.txt
|                   +-- timestamps.txt
|               +-- lidars
|           +-- test
|               +-- pangyo00
|                   +-- 000_L.png
|                   +-- 000_R.png
|                   +-- ...
|               +-- pangyo**
```

### Reference

[NAVER LABS DATA](https://challenge.naverlabs.com/) </br>
[Preprocessing](https://github.com/naverlabs/mapping-and-localization-challenge/blob/master/labs_outdoor_dataset_tutorial.ipynb) </br>
[NetVLAD](https://github.com/Nanne/pytorch-NetVlad) </br>
[SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork) </br>

