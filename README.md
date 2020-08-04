# SejongRCV-Outdoor
NAVER LABS Mapping &amp; Localization Challenge

## How to run

### test
```
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=(GPU_NUM) python test.py --checkpoint (checkpoint_path) --place [pangyo/yeuido] --top_k 10 --DB_ROOT (DB_ROOT)
```

## Config.json
> NetVLAD, SuperGlue, APGeM을 설정할 수 있습니다.

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
