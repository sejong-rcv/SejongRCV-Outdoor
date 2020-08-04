import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import pandas as pd

from os.path import join, exists
from scipy.io import loadmat
import numpy as np
from random import randint, random
from collections import namedtuple
from PIL import Image
from utils.Lidar import Lidar_parse
from sklearn.neighbors import NearestNeighbors
import h5py
import pdb
import faiss

root = "data/naver"

#yeouido
yeouido_img_dir = 'yeouido/train/images/left'
yeouido_img_list_txt = "yeouido_images_list_total.txt"
yeouido_position_npy = "./data/yeouido_position_total.npy"

#pangyo 
img_dir = 'pangyo/train/images/left' 
img_list_txt = "pangyo_list_total.txt"
position_npy = "./data/position_total.npy"


class kNN_GPU():
    def __init__(self, d=64, GPU=False, GPU_Number=0): #default dimension=64
        self.idx = faiss.IndexFlatL2( d )   # build the index
        self.GPU = GPU
        if self.GPU:
            self.res = faiss.StandardGpuResources()  # use a single GPU
            self.res.noTempMemory()
            gpu = faiss.index_cpu_to_gpu(self.res, GPU_Number, self.idx)
            self.idx = gpu
    def train(self, index):
        if self.idx.is_trained:
            self.idx.add(index)
            return
        else:
            raise ValueError('kNN GPU Error')
    def predict(self, query, k=1, distance=False):
        D, I = self.idx.search(query, k)     # actual search

        if distance == True :
            return D, I
        else :
          return I

    def delete(self):
        del self.idx
        if self.GPU:
            del self.res
        return
    
class Naver_Yeouido_IMG(data.Dataset):
    def __init__(self, input_transform=None):

        super().__init__()

        self.img_list = join(root, "yeouido_test_list_Left.txt")
        self.images = [e.strip() for e in open(self.img_list)]
        self.input_transform = input_transform()

    def __getitem__(self, index):
        
        path = self.images[index]
        img = Image.open(path)

        if self.input_transform:
            img = self.input_transform(img)

        return img, path, index

    def __len__(self):
        return len(self.images)

class Naver_Indoor_B1_IMG(data.Dataset):
    def __init__(self, input_transform=None):

        super().__init__()
        
        img_train_list_path = join("./data/indoor/b1/train/csv/v2/train_val/train_b1.csv")
        img_val_list_path = join("./data/indoor/b1/train/csv/v2/train_val/val_b1.csv")
        img_train_list = pd.read_csv(img_train_list_path)
        img_val_list = pd.read_csv(img_val_list_path)
        
        self.images = np.array(img_train_list['id']).tolist()
        self.date = np.array(img_train_list['date']).tolist()
        self.input_transform = input_transform()

    def __getitem__(self, index):
        
        img_name = self.images[index] +'.jpg'
        path = "./data/indoor/b1/train/" + self.date[index] + '/images/' + img_name
        img = Image.open(path)

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)
    
class New_idea_Naver_Datasets_IMG(data.Dataset):
    def __init__(self, input_transform=None):

        super().__init__()

        self.img_list = join(root, img_list_txt)

        self.images = [e.strip() for e in open(self.img_list)]
        self.input_transform = input_transform()

    def __getitem__(self, index):

        img = Image.open(join(root,img_dir,self.images[index]))
        
        w = img.size[0]
        h = img.size[1]
        
        half_top = (0,0,w,h/2)
        half_top_img = img.crop(half_top)
        tr_1 = (0,0,w/3,h)
        tr_1_img = img.crop(tr_1)
        tr_3 = (2*w/3,0,w,h)
        tr_3_img = img.crop(tr_3)

        if self.input_transform:
            img = self.input_transform(img)
            half_top_img = self.input_transform(half_top_img)
            tr_1_img = self.input_transform(tr_1_img)
            tr_3_img = self.input_transform(tr_3_img)
        
        
        return img,half_top_img,tr_1_img,tr_3_img, index

    def __len__(self):
        return len(self.images)

class Naver_Pangyo_IMG(data.Dataset):
    def __init__(self, input_transform=None):

        super().__init__()

        self.img_list = join(root, "pangyo_test_list_Left.txt")
        self.images = [e.strip() for e in open(self.img_list)]
        self.input_transform = input_transform()

    def __getitem__(self, index):
        
        path = self.images[index]
        img = Image.open(path)

        if self.input_transform:
            img = self.input_transform(img)

        return img, path, index

    def __len__(self):
        return len(self.images)
    
class Naver_Datasets_IMG(data.Dataset):
    def __init__(self, input_transform=None):

        super().__init__()

        self.img_list = join(root, img_list_txt)

        self.images = [e.strip() for e in open(self.img_list)]
        self.input_transform = input_transform()

    def __getitem__(self, index):

        img = Image.open(join(root,img_dir,self.images[index]))

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)

class Naver_Datasets_IMG_Segment(data.Dataset):
    def __init__(self, input_transform=None):

        super().__init__()
        
        self.seg_list ='/raid4/outdoor/seg_train_pangyo.txt'
        self.segment = [e.strip() for e in open(self.seg_list)]

        self.img_list = join(root, img_list_txt)
        self.images = [e.strip() for e in open(self.img_list)]
        
        self.input_transform = input_transform()

    def __getitem__(self, index):

        seg = np.load(join(self.segment[index]))['img'].transpose(1,2,0)
            
        mask = np.ones((seg.shape[0], seg.shape[1], seg.shape[2]), dtype="i8")
        mask[seg > 10] = 0
        img = np.array(Image.open(join(root,img_dir,self.images[index])))
        img = np.stack([mask[:,:,0] * img[:,:,0], mask[:,:,0] * img[:,:,1], mask[:,:,0] * img[:,:,2]], axis=2)
        img = Image.fromarray(img.astype(np.uint8))

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)
    
class Naver_Datasets_yeouido_IMG(data.Dataset):
    def __init__(self, input_transform=None):

        super().__init__()

        self.img_list = join(root, yeouido_img_list_txt)

        self.images = [e.strip() for e in open(self.img_list)]
        self.input_transform = input_transform()

    def __getitem__(self, index):
        
        try :
            img = Image.open(join(root,yeouido_img_dir,self.images[index]))
            
            if self.input_transform:
                img = self.input_transform(img)
        except :
            import pdb;pdb.set_trace()
            print(self.images[index])
            

        return img, index

    def __len__(self):
        return len(self.images)
    
class Test_Naver_Datasets_IMG(data.Dataset):
    def __init__(self, input_transform=None):

        super().__init__()

        self.img_list = join(root, test_img_list)
        self.images = [e.strip() for e in open(self.img_list)]
        self.input_transform = input_transform()

    def __getitem__(self, index):

        img = Image.open(join(root,img_dir,self.images[index]))

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)

def collate_fn(batch):
    """
    Creates mini-batch tensors from the list of tuples (query, positive, negatives).
    
    Args:
        data: list of tuple (query, positive, negatives). 
            - query: torch tensor of shape (3, h, w).
            - positive: torch tensor of shape (3, h, w).
            - negative: torch tensor of shape (n, 3, h, w).
    Returns:
        query: torch tensor of shape (batch_size, 3, h, w).
        positive: torch tensor of shape (batch_size, 3, h, w).
        negatives: torch tensor of shape (batch_size, n, 3, h, w).
    """

    batch = list(filter (lambda x:x is not None, batch))
    if len(batch) == 0: return None, None, None, None, None

    query, positive, negatives, indices = zip(*batch)

    query = data.dataloader.default_collate(query)
    positive = data.dataloader.default_collate(positive)
    negCounts = data.dataloader.default_collate([x.shape[0] for x in negatives])
    negatives = torch.cat(negatives, 0)
    import itertools
    indices = list(itertools.chain(*indices))

    return query, positive, negatives, negCounts, indices

def get_multiple_elements(in_list, in_indices) :
    return [in_list[i] for i in in_indices]

class Naver_Datasets(data.Dataset):
    def __init__(self, nNegSample=1000, nNeg=10, margin=0.6, input_transform=None):
        super().__init__()
        
        self.img_list = join(root, img_list_txt)
        self.images = [e.strip() for e in open(self.img_list)]
        self.input_transform = input_transform()

        self.nNegSample = nNegSample # number of negatives to randomly sample
        self.nNeg = nNeg # number of negatives used for training
        self.margin = margin

        self.position = np.load(position_npy)
        self.positive_thres = 5
        self.negative_thres = 10

        self.DBidx = np.arange(int(len(self.images)/2)) *2
        self.Qidx = self.DBidx +1

        np.random.shuffle(self.DBidx)
        np.random.shuffle(self.Qidx)

        # potential positives are those within nontrivial threshold range
        #fit NN to find them, search by radius
        #knn = kNN_GPU(d=len(get_multiple_elements(self.position,self.DBidx)[0]), GPU = True, GPU_Number=torch.cuda.current_device())
        #knn.train(np.array(get_multiple_elements(self.position,self.DBidx)).astype("float32"))
        knn_cpu = NearestNeighbors(n_jobs=-1,metric='euclidean')
        knn_cpu.fit(get_multiple_elements(self.position,self.DBidx))
        
        # potential negatives are those outside of posDistThr range
        self.potential_positives = knn_cpu.radius_neighbors(get_multiple_elements(self.position,self.Qidx),
                radius=self.positive_thres, 
                return_distance=False)
                
        #self.potential_positives = knn.predict(np.asarray(get_multiple_elements(self.position,self.Qidx)).astype("float32"), 10)
        
        # sort indecies of potential positives
        for i, positive_indices in enumerate(self.potential_positives) :
            self.potential_positives[i] = np.sort(positive_indices)
        
        # it's possible some queries don't have any non trivial potential positives
        self.queries = np.where(np.array([len(x) for x in self.potential_positives])>0)[0]
        
        # for potential negatives
        potential_unnegatives = knn_cpu.radius_neighbors(get_multiple_elements(self.position,self.Qidx), radius=self.negative_thres, return_distance=False)
        #potential_unnegatives = knn.predict(np.asarray(get_multiple_elements(self.position,self.Qidx)).astype("float32"), 20)

        # potential negatives' indices of DBidx away then 25 meters
        self.potential_negatives = []
        for pos in potential_unnegatives :
            self.potential_negatives.append(np.setdiff1d(np.arange(self.DBidx.shape[0]), pos, assume_unique=True))

        self.cache = None # filepath of HDF5 containing feature vectors for images
        self.negCache = [np.empty((0,)) for _ in range(self.Qidx.shape[0])]

    def __getitem__(self, index):
        index = self.queries[index] # re-map index to match dataset
        
        with h5py.File(self.cache, mode='r') as h5: 
            h5feat = h5.get("features")

            qFeat = h5feat[self.Qidx[index]]

            posFeat = h5feat[sorted(self.DBidx[self.potential_positives[index]].tolist())]
            
            knn = kNN_GPU(d=len(posFeat[0]), GPU = True, GPU_Number=torch.cuda.current_device())
            knn.train(posFeat.astype("float32"))        
                
            #knn = NearestNeighbors(n_jobs=-1,metric='euclidean')
            #knn.fit(posFeat)

            # positive's index of DBidx closest to query vlad vector
            dPos, posNN = knn.predict(qFeat.reshape(1,-1),1,distance=True)
            knn.delete()
            del knn
            #dPos, posNN = knn.kneighbors(qFeat.reshape(1,-1), 1)
            dPos = dPos.item()
            posIndex = self.potential_positives[index][posNN[0]].item()
            
            # choose number of nNegSample potiential negatives' indices of DBidx
            negSample = np.random.choice(self.potential_negatives[index], self.nNegSample)
            negSample = np.unique(np.concatenate([self.negCache[index], negSample])).astype(np.int)

            negFeat = h5feat[sorted(self.DBidx[negSample].tolist())]


            # choose number of nNeg x 10 negatives whose vlad vector is closest to query vlad vector
            
            try :
                knn = kNN_GPU(d=len(posFeat[0]), GPU = True, GPU_Number=torch.cuda.current_device())
                knn.train(negFeat)             
            #knn.fit(negFeat)
            except :
                import pdb;pdb.set_trace()
                
            dNeg, negNN = knn.predict(qFeat.reshape(1,-1), self.nNeg*10, distance=True)
            #dNeg, negNN = knn.kneighbors(qFeat.reshape(1,-1), self.nNeg*10)
            knn.delete()
            del knn
            
            dNeg = dNeg.reshape(-1)
            negNN = negNN.reshape(-1)

            # dNeg is vlad vector of potential negatives
            # dPos is vlad vector of positive
           
            violatingNeg = dNeg < dPos + self.margin**0.5
            
            if np.sum(violatingNeg) < 1:
                #if none are violating then skip this query
                return None
            
            # choose number of nNeg negatives in violatingNeg
            negNN = negNN[violatingNeg][:self.nNeg]
            # nNeg number of negative's indices of DBidx 
            negIndices = negSample[negNN].astype(np.int32)

            self.negCache[index] = negIndices

        
        query = Image.open(join(root,img_dir, get_multiple_elements(self.images,self.Qidx)[index]))
        positive = Image.open(join(root,img_dir, get_multiple_elements(self.images,self.DBidx)[posIndex]))
        
        if self.input_transform:
            query = self.input_transform(query)
            positive = self.input_transform(positive)

        negatives = []
        for negIndex in negIndices :
            negative = Image.open(join(root,img_dir, get_multiple_elements(self.images,self.DBidx)[negIndex]))

            if self.input_transform :
                negative = self.input_transform(negative)
            
            negatives.append(negative)

        negatives = torch.stack(negatives, 0)
        
        return query, positive, negatives, [index, posIndex] + negIndices.tolist()

    def __len__(self):
        return len(self.queries)

    
class yeouido_Naver_Datasets(data.Dataset):
    def __init__(self, nNegSample=1000, nNeg=10, margin=0.1, input_transform=None):
        super().__init__()
        
        self.img_list = join(root, yeouido_img_list_txt)
        self.images = [e.strip() for e in open(self.img_list)]
        self.input_transform = input_transform()

        self.nNegSample = nNegSample # number of negatives to randomly sample
        self.nNeg = nNeg # number of negatives used for training
        self.margin = margin

        self.position = np.load(yeouido_position_npy)
        self.positive_thres = 5
        self.negative_thres = 20

        All_idx = np.arange(0,len(self.images),step=1)
        self.Qidx = np.arange(0,len(self.images),step=4)
        self.DBidx = np.setdiff1d(All_idx, self.Qidx)
        #self.DBidx = np.arange(int(len(self.images)/2)) *2
        #self.Qidx = self.DBidx +1

        np.random.shuffle(self.DBidx)
        np.random.shuffle(self.Qidx)

        # potential positives are those within nontrivial threshold range
        #fit NN to find them, search by radius
        #knn = kNN_GPU(d=len(get_multiple_elements(self.position,self.DBidx)[0]), GPU = True, GPU_Number=torch.cuda.current_device())
        #knn.train(np.array(get_multiple_elements(self.position,self.DBidx)).astype("float32"))
        knn_cpu = NearestNeighbors(n_jobs=-1,metric='euclidean')
        knn_cpu.fit(get_multiple_elements(self.position,self.DBidx))
        
        # potential negatives are those outside of posDistThr range
        self.potential_positives = knn_cpu.radius_neighbors(get_multiple_elements(self.position,self.Qidx),
                radius=self.positive_thres, 
                return_distance=False)
                
        #self.potential_positives = knn.predict(np.asarray(get_multiple_elements(self.position,self.Qidx)).astype("float32"), 10)
        
        # sort indecies of potential positives
        for i, positive_indices in enumerate(self.potential_positives) :
            self.potential_positives[i] = np.sort(positive_indices)
        
        # it's possible some queries don't have any non trivial potential positives
        self.queries = np.where(np.array([len(x) for x in self.potential_positives])>0)[0]
        
        # for potential negatives
        potential_unnegatives = knn_cpu.radius_neighbors(get_multiple_elements(self.position,self.Qidx), radius=self.negative_thres, return_distance=False)
        #potential_unnegatives = knn.predict(np.asarray(get_multiple_elements(self.position,self.Qidx)).astype("float32"), 20)

        # potential negatives' indices of DBidx away then 25 meters
        self.potential_negatives = []
        for pos in potential_unnegatives :
            self.potential_negatives.append(np.setdiff1d(np.arange(self.DBidx.shape[0]), pos, assume_unique=True))

        self.cache = None # filepath of HDF5 containing feature vectors for images
        self.negCache = [np.empty((0,)) for _ in range(self.Qidx.shape[0])]

    def __getitem__(self, index):
        index = self.queries[index] # re-map index to match dataset
        
        with h5py.File(self.cache, mode='r') as h5: 
            h5feat = h5.get("features")

            qFeat = h5feat[self.Qidx[index]]

            posFeat = h5feat[sorted(self.DBidx[self.potential_positives[index]].tolist())]
            
            knn = kNN_GPU(d=len(posFeat[0]), GPU = True, GPU_Number=torch.cuda.current_device())
            knn.train(posFeat.astype("float32"))        
                
            #knn = NearestNeighbors(n_jobs=-1,metric='euclidean')
            #knn.fit(posFeat)

            # positive's index of DBidx closest to query vlad vector
            dPos, posNN = knn.predict(qFeat.reshape(1,-1),1,distance=True)
            knn.delete()
            del knn
            #dPos, posNN = knn.kneighbors(qFeat.reshape(1,-1), 1)
            dPos = dPos.item()
            posIndex = self.potential_positives[index][posNN[0]].item()
            
            # choose number of nNegSample potiential negatives' indices of DBidx
            negSample = np.random.choice(self.potential_negatives[index], self.nNegSample)
            negSample = np.unique(np.concatenate([self.negCache[index], negSample])).astype(np.int)

            negFeat = h5feat[sorted(self.DBidx[negSample].tolist())]


            # choose number of nNeg x 10 negatives whose vlad vector is closest to query vlad vector
            
            try :
                knn = kNN_GPU(d=len(posFeat[0]), GPU = True, GPU_Number=torch.cuda.current_device())
                knn.train(negFeat)             
            #knn.fit(negFeat)
            except :
                import pdb;pdb.set_trace()
                
            dNeg, negNN = knn.predict(qFeat.reshape(1,-1), self.nNeg*10, distance=True)
            #dNeg, negNN = knn.kneighbors(qFeat.reshape(1,-1), self.nNeg*10)
            knn.delete()
            del knn
            
            dNeg = dNeg.reshape(-1)
            negNN = negNN.reshape(-1)

            # dNeg is vlad vector of potential negatives
            # dPos is vlad vector of positive
           
            violatingNeg = dNeg < dPos + self.margin**0.5
            
            if np.sum(violatingNeg) < 1:
                #if none are violating then skip this query
                return None

            # choose number of nNeg negatives in violatingNeg
            negNN = negNN[violatingNeg][:self.nNeg]
            # nNeg number of negative's indices of DBidx 
            negIndices = negSample[negNN].astype(np.int32)

            self.negCache[index] = negIndices

        
        query = Image.open(join(root,yeouido_img_dir, get_multiple_elements(self.images,self.Qidx)[index]))
        positive = Image.open(join(root,yeouido_img_dir, get_multiple_elements(self.images,self.DBidx)[posIndex]))
        
        if self.input_transform:
            query = self.input_transform(query)
            positive = self.input_transform(positive)

        negatives = []
        for negIndex in negIndices :
            negative = Image.open(join(root,yeouido_img_dir, get_multiple_elements(self.images,self.DBidx)[negIndex]))

            if self.input_transform :
                negative = self.input_transform(negative)
            
            negatives.append(negative)

        negatives = torch.stack(negatives, 0)

        return query, positive, negatives, [index, posIndex] + negIndices.tolist()

    def __len__(self):
        return len(self.queries)
