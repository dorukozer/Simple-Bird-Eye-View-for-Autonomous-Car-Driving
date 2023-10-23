"""
code adapted from https://github.com/nv-tlabs/lift-splat-shoot
and also https://github.com/wayveai/fiery/blob/master/fiery/data.py
"""

import torch
import os
import numpy as np
from PIL import Image
import cv2
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box
from glob import glob

import torchvision
from functools import reduce
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.data_classes import PointCloud
from nuscenes.utils.geometry_utils import transform_matrix
import time

import utils.py
import utils.geom
import itertools
import matplotlib.pyplot as plt




def get_batch_iou(preds, binimgs):
    """Assumes preds has NOT been sigmoided yet
    """
    with torch.no_grad():
        pred = (preds > 0)
        tgt = binimgs.bool()
        intersect = (pred & tgt).sum().float().item()
        union = (pred | tgt).sum().float().item()
    return intersect, union, intersect / union if (union > 0) else 1.0


def get_val_info(model, valloader, loss_fn, device, use_tqdm=False, max_iters=None, use_lidar=False):
    model.eval()
    total_loss = 0.0
    total_intersect = 0.0
    total_union = 0
    print('running eval...')

    t0 = time()
    
    loader = tqdm(valloader) if use_tqdm else valloader

    if max_iters is not None:
        counter = 0
    with torch.no_grad():
        for batch in loader:

            if max_iters is not None:
                counter += 1
                if counter > max_iters:
                    break

            if use_lidar:
                allimgs, rots, trans, intrins, pts, binimgs = batch
            else:
                allimgs, rots, trans, intrins, binimgs = batch
                
            preds = model(allimgs.to(device), rots.to(device),
                          trans.to(device), intrins.to(device))
            binimgs = binimgs.to(device)

            # loss
            total_loss += loss_fn(preds[:,0:1], binimgs).item() * preds.shape[0]

            # iou
            intersect, union, _ = get_batch_iou(preds[:,0:1], binimgs)
            total_intersect += intersect
            total_union += union
    t1 = time()
    print('eval took %.2f seconds' % (t1-t0))

    model.train()

    if max_iters is not None:
        normalizer = counter
    else:
        normalizer = len(valloader.dataset)
        
    return {
        'total_loss': total_loss / normalizer,
        'iou': total_intersect / total_union,
    }



import os 
import glob
import json

import numpy as np
import torch

from PIL import Image
from torchvision import transforms
#from torchvision.datasets.folder import default_loader

from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

from typing import List, Optional, Sequence, Union, Callable

import matplotlib.pyplot as plt

from utils.bev_utils import * 


########## SENSOR CONFIGS FOR TRANSFUSER DATA #############

cam_config = {
            'width': 320,
            'height': 160,
            'fov': 60
        }

SENSOR_CONFIGS = {
        'CAM_RGB_FRONT_LEFT': {
                            'type': 'sensor.camera.rgb',
                            'x': 1.3, 'y': 0.0, 'z':2.3,
                            'roll': 0.0, 'pitch': 0.0, 'yaw': -60.0,
                            'width': cam_config['width'], 'height': cam_config['height'], 'fov': cam_config['fov'],
                            'id': 'rgb_left'
        },
        'CAM_RGB_FRONT': {  
                            'type': 'sensor.camera.rgb',
                            'x': 1.3, 'y': 0.0, 'z':2.3,
                            'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                            'width': cam_config['width'], 'height': cam_config['height'], 'fov': cam_config['fov'],
                            'id': 'rgb_front'
                            },

        'CAM_RGB_FRONT_RIGHT': {
                            'type': 'sensor.camera.rgb',
                            'x': 1.3, 'y': 0.0, 'z':2.3,
                            'roll': 0.0, 'pitch': 0.0, 'yaw': 60.0,
                            'width': cam_config['width'], 'height': cam_config['height'], 'fov': cam_config['fov'],
                            'id': 'rgb_right'
        }
}

class BEVDataset(Dataset):
    def __init__(self, 
                data_path,
                split, 
                image_folder,
                transform,
                zero_out_red_channel,
                data_aug_conf,
                seqlen,
                refcam_id,
                get_tids,
                res_3d,
                bounds,
                centroid,
                #cams,
                rgb_cam_configs = {},
                use_radar_filters = False,
                do_shuffle_cams = False,
            ):

        super(BEVDataset, self).__init__()


        self.data_path = data_path #'/home/mbarin/storage/data-transfuser'
        self.split = split
        self.image_folder = image_folder
        self.transform = transform
        self.zero_out_red_channel = zero_out_red_channel
        
        self.dataset = self.get_img_paths()

        #self.cams = cams
        self.rgb_cam_configs = SENSOR_CONFIGS # rgb_cam_configs
        self.use_radar_filters = use_radar_filters
        self.do_shuffle_cams = do_shuffle_cams
        self.res_3d = res_3d
        self.bounds = bounds
        self.centroid = centroid

        self.seqlen = 1
        self.refcam_id = 1 # TODO: check if it is the front camera


        XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX = self.bounds
        self.Z, self.Y, self.X = self.res_3d


        grid_conf = { # note the downstream util uses a different XYZ ordering
            'xbound': [XMIN, XMAX, (XMAX-XMIN)/float(self.X)],
            'ybound': [ZMIN, ZMAX, (ZMAX-ZMIN)/float(self.Z)],
            'zbound': [YMIN, YMAX, (YMAX-YMIN)/float(self.Y)],
        }

        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()



    def get_img_paths(self):

        self.main_folders = os.listdir(self.data_path)

        if self.split=="train":
            self.main_folders = [self.main_folders[2]]  # [:-2] # TODO: is it for rgb [2:-2]
        elif self.split=="val":
            self.main_folders = [self.main_folders[2]]   #[self.main_folders[-2]]
        elif self.split=="test":
            self.main_folders = [self.main_folders[-2]]

        #print('main folders ', self.main_folders)


        sub_folder_depth = '/*/*'

        self.sub_folders = []
        for folder in self.main_folders:
            sub_folders_path = self.data_path + folder + sub_folder_depth
            self.sub_folders += glob.glob(sub_folders_path)

        self.images = []
        dataset = []
        for folder in self.sub_folders:
            img_paths = folder + '/' + self.image_folder + '/*'

            img_paths = sorted(glob.glob(img_paths))

            # TODO: try passing multiple steps
            dataset += [path for path in img_paths]


        dataset = [dataset[0]] # YOU TRY OVERFITTING!!!!

        print('DATASET ' , dataset)


        print(f"Detected {len(dataset)} images in split {self.split}")
        print(f"Detected {len(dataset)} commands in split {self.split}")

        return dataset
    
    
    def get_file_path(self, rgb_path, folder_name, file_ext, is_encoded=False):

        if is_encoded:
            file_name = 'encoded_' + rgb_path.split('/')[-1][:-3]  + file_ext
        else:
            file_name = rgb_path.split('/')[-1][:-3] + file_ext

        # print('FOLDER NAME ', folder_name)
        # print('FILE NAME ', file_name)
        # print('PATH LIST ', rgb_path.split('/')[:-2])
        file_path = '/'.join(rgb_path.split('/')[:-2] + [folder_name] + [file_name])

        #print('fILE PATH ', file_path)

        return file_path
        

    def __len__(self):

        return len(self.dataset)


    def get_egopose_4x4matrix(self, rgb_path):

        f =  open(self.get_file_path(rgb_path,'measurements','json'))
        measurements = json.load(f)
        egopose = measurements['ego_matrix']

        return torch.Tensor(egopose)


    def get_binimg(self, rgb_path):
        '''
            get_binimg():
            For NuScenes case, they define polygons representing vehicles based on the egopose.
            Then form the topdown view of the scene.

            get_seg_bev():
            Also, they form bev segmentation of the scene using LIDAR data. 
            We consider topdown view as ground truth BEV. 

            For Transfuser data, we have 'topdown' and 'semantics' folders. The former corresponds to their binimg.
            The latter is the segmentic version of the rgb camera images. 960 x 160     

            For ncams = 3, we need to crop topdown images and get the upper half part of the image. 
            Since we are just predicting the front part.
        '''

        topdown_path = self.get_file_path(rgb_path,'topdown','png',is_encoded=True)
        topdown = Image.open(topdown_path)
        topdown = transforms.ToTensor()(topdown) # .unsqueeze_(0)
        
        # zero out red channel
        topdown[0,:,:] = 0

        # take the upper part of the topdown view 
        _, H, W = topdown.shape
        """ print('H ', H)
        print('w ', W) """
        topdown_t = topdown[:,:H//2,:] #[:,H//2:(H//2+self.Z),:]

        #print('topdowb shape ' , topdown_t.shape)

        """ plt.imshow(topdown_t.permute(1,2,0))
        plt.show() """

        # TODO: transform topdown img based on model input img. 
        # resize, crop? 500x500 is too big to handle
        topdown =  transforms.functional.center_crop(topdown_t,(self.X,self.Z))

        # TODO: HOW DOES IOU IS COMPUTED, LOOK AT THE COLORS OF THE MODEL PREDS
        topdown = transforms.functional.rgb_to_grayscale(topdown).squeeze()


        """ print('TOPDOWN SHAPE ', topdown.shape)
        plt.imshow(topdown)
        plt.show()
        exit() """
        """ print('MINMAXXXXX ' , torch.max(topdown), '   ' , torch.min(topdown), )
        plt.imshow(topdown)
        plt.show() """
        

        return topdown.unsqueeze(0)

    
    def get_image_data(self, rgb_path):

        rgbs = []
        rots = []
        trans = []
        intrins = []


        # RGB CAMERA IMAGE
        rgb = Image.open(rgb_path)

        #print(type(rgb), )

        convert_tensor = transforms.ToTensor()

        rgb = convert_tensor(rgb)
        
        """ if self.transform is not None:
            rgb = self.transform(rgb) """
        
        if self.zero_out_red_channel:
            rgb[0, :, :] = 0

        _, H, W = rgb.shape

        rgbs.append(rgb[:,:,:W//3])
        rgbs.append(rgb[:,:,W//3:2*W//3])
        rgbs.append(rgb[:,:,2*W//3:])


        # MEASUREMENTS

        import json
        #measurements = json.load(self.get_measurements_path(rgb_path))
        _, trans, rots, intrins = utils.bev_utils.get_trans_and_rot_from_sensor_list(SENSOR_CONFIGS)
  

        """ print('RGBs ', len(rgbs))
        print(rgbs[0].shape)
        print('TRNS ' , len(trans))
        print('TRNS ' , trans[0].shape)
        print('TOR ' , rots[0].shape)
        print('INSTRINS ', len(intrins))
        print('INTRINSS ', intrins[0].shape) """


        return [torch.stack(rgbs), torch.stack(rots), torch.stack(trans), torch.stack(intrins)]


    def get_single_item(self,rgb_path):


        imgs, rots, trans, intrins = self.get_image_data(rgb_path)  # rot , trans of the rgb cameras
        
        img_ref = imgs[self.refcam_id].clone()
        img_0 = imgs[0].clone()
        imgs[0] = img_ref
        imgs[self.refcam_id] = img_0

        rot_ref = rots[self.refcam_id].clone()
        rot_0 = rots[0].clone()
        rots[0] = rot_ref
        rots[self.refcam_id] = rot_0
        
        tran_ref = trans[self.refcam_id].clone()
        tran_0 = trans[0].clone()
        trans[0] = tran_ref
        trans[self.refcam_id] = tran_0

        intrin_ref = intrins[self.refcam_id].clone()
        intrin_0 = intrins[0].clone()
        intrins[0] = intrin_ref
        intrins[self.refcam_id] = intrin_0
        
        egopose = self.get_egopose_4x4matrix(rgb_path)
        binimg = self.get_binimg(rgb_path)

        seg_bev = self.get_binimg(rgb_path)
        # valid bev is for invisible annotations, we dont have it in the carla dataset
        valid_bev = 1 - seg_bev                 

        #seg_bev = torch.zeros((1, self.Z, self.X), dtype=torch.float32)
        #valid_bev = torch.ones((1, self.Z, self.X), dtype=torch.float32)
        center_bev = torch.zeros((1, self.Z, self.X), dtype=torch.float32)
        offset_bev = torch.zeros((2, self.Z, self.X), dtype=torch.float32)
        size_bev = torch.zeros((3, self.Z, self.X), dtype=torch.float32)
        ry_bev = torch.zeros((1, self.Z, self.X), dtype=torch.float32)
        ycoord_bev = torch.zeros((1, self.Z, self.X), dtype=torch.float32)
    
        N = 150 # i've seen n as high as 103 before, so 150 is probably safe (max number of objects)
        lrtlist = torch.zeros((N, 19), dtype=torch.float32)
        vislist = torch.zeros((N), dtype=torch.float32)
        scorelist = torch.zeros((N), dtype=torch.float32)
        #lrtlist[:N_] = lrtlist_
        #vislist[:N_] = vislist_
        #scorelist[:N_] = 1

        # WHAT IS BIN IMG
        #binimg = torch.zeros((1, self.Z, self.X))

        binimg = (binimg > 0).float()
        """ print('SEG BEV SHAPE ' , seg_bev.shape)
        print(torch.max(seg_bev), torch.min(seg_bev))

        exit() """
        #seg_bev = (seg_bev > 0).float()
        #valid_bev = (valid_bev > 0 ).float()


        """ print('SEG BEV SHAPE ', seg_bev.shape)
        print('seg bev ', torch.max(seg_bev), torch.max(valid_bev))
        print('seg bev ', torch.min(seg_bev), torch.min(valid_bev))
        import matplotlib.pyplot as plt
        plt.imshow(seg_bev[0])
        plt.show()

        plt.imshow(valid_bev[0])
        plt.show()
        """

        return imgs, rots, trans, intrins, lrtlist, vislist, scorelist, seg_bev, valid_bev, center_bev, offset_bev, size_bev, ry_bev, ycoord_bev, egopose #radar_data, egopose


    def __getitem__(self, idx):
        #bev, cmd, bev_next = default_loader()


        # FOR EACH TIME STEP ??

        rgb_path = self.dataset[idx]
        
        
        #print('RGB PATH ', rgb_path)
        all_imgs = []
        all_rots = []
        all_trans = []
        all_intrins = []
        all_lrtlist = []
        all_vislist = []
        all_tidlist = []
        all_scorelist = []
        all_seg_bev = []
        all_valid_bev = []
        all_center_bev = []
        all_offset_bev = []
        all_radar_data = []
        all_egopose = []


        imgs, rots, trans, intrins, lrtlist, vislist, scorelist, seg_bev, valid_bev, center_bev, offset_bev, size_bev, ry_bev, ycoord_bev, egopose = self.get_single_item(rgb_path)

        all_imgs.append(imgs)
        all_rots.append(rots)
        all_trans.append(trans)
        all_intrins.append(intrins)
        all_lrtlist.append(lrtlist)
        all_vislist.append(vislist)
        #all_tidlist.append(tidlist)
        all_scorelist.append(scorelist)
        all_seg_bev.append(seg_bev)
        all_valid_bev.append(valid_bev)
        all_center_bev.append(center_bev)
        all_offset_bev.append(offset_bev)
        #all_radar_data.append(radar_data)
        all_egopose.append(egopose)

        """ 
        print('all imgs   ', len(all_imgs))
        print('all imgs   ', type(all_imgs[0]), ) """
        all_imgs = torch.stack(all_imgs)
        #print('all_imgs shape ', all_imgs.shape) # 1 num_cam c h w 
        all_rots = torch.stack(all_rots)
        all_trans = torch.stack(all_trans)
        all_intrins = torch.stack(all_intrins)
        all_lrtlist = torch.stack(all_lrtlist)
        all_vislist = torch.stack(all_vislist)
        # all_tidlist = torch.stack(all_tidlist)
        all_scorelist = torch.stack(all_scorelist)
        all_seg_bev = torch.stack(all_seg_bev)
        all_valid_bev = torch.stack(all_valid_bev)
        all_center_bev = torch.stack(all_center_bev)
        all_offset_bev = torch.stack(all_offset_bev)
        # all_radar_data = torch.stack(all_radar_data)
        all_egopose = torch.stack(all_egopose)
        
        
        
        
        usable_tidlist = -1*torch.ones_like(all_scorelist).long()
        counter = 0
        for t in range(len(all_tidlist)):
            for i in range(len(all_tidlist[t])):
                if t==0:
                    usable_tidlist[t,i] = counter
                    counter += 1
                else:
                    st = all_tidlist[t][i]
                    if st in all_tidlist[0]:
                        usable_tidlist[t,i] = all_tidlist[0].index(st)
                    else:
                        usable_tidlist[t,i] = counter
                        counter += 1
        all_tidlist = usable_tidlist


        #print('ALLL IMGSSS SHAP ', all_imgs.shape)

        

        return all_imgs, all_rots, all_trans, all_intrins, all_lrtlist, all_vislist, all_tidlist, all_scorelist, all_seg_bev, all_valid_bev, all_center_bev, all_offset_bev, all_radar_data, all_egopose

 

    def sample_augmentation(self):
        fH, fW = self.data_aug_conf['final_dim']
        if self.is_train:
            if 'resize_lim' in self.data_aug_conf and self.data_aug_conf['resize_lim'] is not None:
                resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            else:
                resize = self.data_aug_conf['resize_scale']

            resize_dims = (int(fW*resize), int(fH*resize))

            newW, newH = resize_dims

            # center it
            crop_h = int((newH - fH)/2)
            crop_w = int((newW - fW)/2)

            crop_offset = self.data_aug_conf['crop_offset']
            crop_w = crop_w + int(np.random.uniform(-crop_offset, crop_offset))
            crop_h = crop_h + int(np.random.uniform(-crop_offset, crop_offset))

            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        else: # validation/test
            # do a perfect resize
            resize_dims = (fW, fH)
            crop_h = 0
            crop_w = 0
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        return resize_dims, crop



def worker_rnd_init(x):
    np.random.seed(13 + x)


def compile_data(version, data_dir, data_aug_conf, centroid, bounds, res_3d, bsz,
                 nworkers, shuffle=True, nsweeps=1, nworkers_val=1, seqlen=1, refcam_id=1, get_tids=False,
                 temporal_aug=False, use_radar_filters=False, do_shuffle_cams=True):

    
    print('loading bev dataset...')
    print('version ', version)
    print('dataroot ', data_dir)



    traindata = BEVDataset(data_path=data_dir,
                            split='train', 
                            image_folder='rgb',
                            transform = None,
                            zero_out_red_channel=True,
                            data_aug_conf=data_aug_conf,
                            #nsweeps=nsweeps,
                            centroid=centroid,
                            bounds=bounds,
                            res_3d=res_3d,
                            seqlen=seqlen,
                            refcam_id=refcam_id,
                            get_tids=get_tids,
                            #temporal_aug=temporal_aug,
                            use_radar_filters=use_radar_filters,
                            do_shuffle_cams=do_shuffle_cams
                            )
    valdata = BEVDataset(data_path=data_dir,
                            split='val', 
                            image_folder='rgb',
                            transform = None,
                            zero_out_red_channel=True,
                            data_aug_conf=data_aug_conf,
                            #nsweeps=nsweeps,
                            centroid=centroid,
                            bounds=bounds,
                            res_3d=res_3d,
                            seqlen=seqlen,
                            refcam_id=refcam_id,
                            get_tids=get_tids,
                            #temporal_aug=temporal_aug,
                            use_radar_filters=use_radar_filters,
                            do_shuffle_cams=do_shuffle_cams
                            )

    trainloader = torch.utils.data.DataLoader(
        traindata,
        batch_size=bsz,
        shuffle=shuffle,
        num_workers=nworkers,
        drop_last=True,
        worker_init_fn=worker_rnd_init,
        pin_memory=False)


    valloader = torch.utils.data.DataLoader(
        valdata,
        batch_size=bsz,
        shuffle=shuffle,
        num_workers=nworkers_val,
        drop_last=True,
        pin_memory=False)


    print('data ready')


    return trainloader, valloader
