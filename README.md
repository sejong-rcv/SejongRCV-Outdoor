# SejongRCV-Outdoor
NAVER LABS Mapping &amp; Localization Challenge

## How to run

### pangyo
```
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=(GPU_NUM) python opt_GeM_2_odometry_real_test_netvlad_superglue_pnp.py --checkpoint (checkpoint_path)
```

### yeouido
```
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=(GPU_NUM) python odometry_real_yeouido_test_netvlad_superglue_pnp.py --checkpoint (checkpoint_path)
```


## Data Tree
```
+-- data
|   +-- naver
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
