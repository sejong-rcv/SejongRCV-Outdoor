# SejongRCV-Outdoor
NAVER LABS Mapping &amp; Localization Challenge

## How to run

### test
```
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=(GPU_NUM) python test.py --checkpoint (checkpoint_path) --place [pangyo/yeuido] --top_k 10 --DB_ROOT (DB_ROOT)
```

## Config.json
> NetVLAD, SuperGlue, APGeM을 설정할 수 있습니다.
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
## Data Tree
```
+-- data
|   +-- pangyo_pose_total.npy  pangyo_position_total.npy  yeouido_pose_total.npy  yeouido_position_total.npy
|   +-- naver
|       +-- submit_json.json
|       +-- centriods
|           +-- GeM_2_Pretrained_Total_DB_cache.hdf5
|           +-- GeM_2_Pretrained_Total_knn_pickle
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
