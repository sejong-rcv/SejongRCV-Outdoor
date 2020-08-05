import h5py, pickle, sys, random
from os.path import join, exists
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from PIL import Image

from models import netvlad
from utils.utils import input_transform, load_pretrained_layers
import dirtorch.nets as nets



class global_descriptor:
    def __init__(self, DB_ROOT, file_path, img_list_txt, pretrained=None):
        self.DB_ROOT=DB_ROOT
        self.file_path=file_path
        self.img_list_txt=img_list_txt
        self.img_list = join(DB_ROOT, img_list_txt)
        self.images = [e.strip() for e in open(self.img_list)]
        self.len_images = len(self.images)
        self.encoder_dim = 512
        self.num_clusters = 64
        self.pretrained=pretrained
        self.cuda=True
        self.device = torch.device("cuda" if self.cuda else "cpu")
        self.input_tr = input_transform()
        


    def retreival(self, query_path, model, model1=None, model2=None):


        
        img = Image.open(query_path)
        img = self.input_tr(img)
        img = img.to(self.device)

        vlad_encoding = model.pool(model.encoder(img.unsqueeze(0)))

        if model1 is not None and model2 is not None:
            model1_encoding = model1(img.unsqueeze(0)).unsqueeze(0)
            model2_encoding = model2(img.unsqueeze(0)).unsqueeze(0)

            test_total_encoding = torch.cat((vlad_encoding, model1_encoding, model2_encoding), dim=1)
            test_total_encoding = F.normalize(test_total_encoding, p=2, dim=1)

        elif model1 is not None:
            model1_encoding = model1(img.unsqueeze(0)).unsqueeze(0)

            test_total_encoding = torch.cat((vlad_encoding, model1_encoding), dim=1)
            test_total_encoding = F.normalize(test_total_encoding, p=2, dim=1)
        
        else:
            test_total_encoding = vlad_encoding

        return test_total_encoding

    def NetVLAD(self, ret, opt):

            

            random.seed(ret['seed'])
            np.random.seed(ret['seed'])
            torch.manual_seed(ret['seed'])
            if self.cuda:
                torch.cuda.manual_seed(ret['seed'])

            encoder_dim = ret['encoder_dim']
            encoder = models.vgg16(pretrained=self.pretrained)
            # capture only feature part and remove last relu and maxpool
            layers = list(encoder.features.children())[:-2]

            if self.pretrained:
                # if using pretrained then only train conv5_1, conv5_2, and conv5_3
                for l in layers[:-5]: 
                    for p in l.parameters():
                        p.requires_grad = False

            encoder = nn.Sequential(*layers)
            model = nn.Module() 
            model.add_module('encoder', encoder)
            net_vlad = netvlad.NetVLAD(num_clusters=ret["num_clusters"], dim=ret["encoder_dim"])
            model.add_module('pool', net_vlad)
            load_pretrained_layers(model, opt.checkpoint)
            model = model.to(self.device)

            model.eval()

            return model

    def APGeM(self, checkpoint):
        GeM = nets.create_model("resnet101_rmac",pretrained=checkpoint, without_fc=False)
        GeM = GeM.to(self.device)
        GeM.eval()
        return GeM

    def make_cache(self, DB_cache_name, pickle_name, model, imgDataLoader, model1=None, model2=None):

        
        if not exists(join(self.DB_ROOT, 'centroids', pickle_name)) :
            print("===> Loading Cache")
            if not exists(join(self.DB_ROOT, 'centroids', DB_cache_name)) :
                
                print("===> Making Cache")
                dataset_cache = join(self.DB_ROOT, 'centroids', DB_cache_name)
                with h5py.File(dataset_cache, mode='w') as h5: 
                    if model1 is not None and model2 is not None:
                        # pool_size = self.encoder_dim * self.num_clusters + 1024*2
                        pool_size=36864
                    else:
                        pool_size = self.encoder_dim * self.num_clusters
                    DBfeature = h5.create_dataset("features", 
                            [self.len_images, pool_size], 
                            dtype=np.float32)
                    
                    with torch.no_grad():
                        for iteration, (input, indices) in enumerate(tqdm(imgDataLoader), 1):
                            input = input.to(self.device)
                            image_encoding = model.encoder(input)
                            vlad_encoding = model.pool(image_encoding) 

                            if model1 is not None and model2 is not None:
                                model1_encoding=model1(input)
                                model2_encoding=model2(input)
                                total_encoding = torch.cat((vlad_encoding, model1_encoding ,model2_encoding), dim=1)
                                total_encoding = F.normalize(total_encoding, p=2, dim=1)

                                DBfeature[indices.detach().numpy(), :] = total_encoding.detach().cpu().numpy()

                                del total_encoding
                                
                            elif model1 is not None:
                                model1_encoding=model1(input)
                                total_encoding = torch.cat((vlad_encoding, model1_encoding ,model2_encoding), dim=1)
                                total_encoding = F.normalize(total_encoding, p=2, dim=1)

                                DBfeature[indices.detach().numpy(), :] = total_encoding.detach().cpu().numpy()

                                del total_encoding

                            else:
                                DBfeature[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()
                            
                            del input, image_encoding, vlad_encoding

                h5 =  h5py.File(join(self.DB_ROOT, 'centroids', DB_cache_name), mode='r')
                DBfeature = h5.get('features')
                del h5

            else :
                h5 =  h5py.File(join(self.DB_ROOT, 'centroids', DB_cache_name), mode='r')
                DBfeature = h5.get('features')
                del h5
            print("Done")      
                    
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(DBfeature)
            knnPickle = open(join(self.DB_ROOT, 'centroids', pickle_name),'wb')
            pickle.dump(knn, knnPickle, protocol=4)  

            print("Restart for using pickle")
            sys.exit(0)

            return knn

        else :
            knn = pickle.load(open(join(self.DB_ROOT, 'centroids', pickle_name), 'rb'))
            print("Load Done")


            return knn
