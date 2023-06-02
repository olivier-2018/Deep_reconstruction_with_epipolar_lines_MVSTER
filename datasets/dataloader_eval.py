from torch.utils.data import Dataset
import numpy as np
import os, cv2, time
from PIL import Image
from datasets.data_io import *
import math


DEBUG = False  # local debugging

class MVSDataset(Dataset):
    def __init__(self, datapath, resolution, listfile, mode, nviews, interval_scale=1.06, **kwargs):
        super(MVSDataset, self).__init__()
        self.datapath = datapath
        self.resolution = resolution
        self.listfile = listfile
        self.mode = mode
        assert self.mode == "test"
        self.nviews = nviews
        self.interval_scale = interval_scale
        
        self.ndepths = 192  # Hardcode
        self.max_h, self.max_w = kwargs["max_h"], kwargs["max_w"]
        self.fix_res = kwargs.get("fix_res", False)  #whether to fix the resolution of input image.
        self.fix_wh = False
        self.pair_fname = kwargs.get("pair_fname", "pair.txt")
        self.lighting = kwargs.get("lighting", 3)     
        self.dsname = kwargs.get("dsname", "blender")
        
        self.metas = self.build_list()

        if self.dsname == "dtu":        
            self.img_folder = 'Rectified_raw/{}/rect_{:0>3}_3_r5000.png'
            self.cam_folder = 'Cameras/{:0>8}_cam.txt'
            # self.img_folder = 'Rectified/{}_train/rect_{:0>3}_3_r5000.png'
            # self.cam_folder = 'Cameras/train/{:0>8}_cam.txt'
        elif self.dsname == "blender":
            self.img_folder = 'Rectified'+self.resolution+'/{}/rect_C{:0>3}_L{:0>2}.png'
            self.cam_folder = 'Cameras'+self.resolution+'/{:0>8}_cam.txt'
        elif self.dsname == "bin":
            self.img_folder = 'Rectified/{}/{:0>8}.png'
            self.cam_folder = 'Cameras/{}/{:0>8}_cam.txt'
        

    def build_list(self):
        metas = []

        # scans         
        for scan in self.listfile:
            print ("[DATALOADER] Pair file: ", self.pair_fname)
            # read the pair file
            with open(os.path.join(self.datapath, self.pair_fname)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]                    
                    metas.append((scan, ref_view, src_views, scan))
        print("[DataLoader] Mode:{}, Ncams:{}, #metas:{} ".format(self.mode, num_viewpoint, len(metas)))

        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filename, interval_scale):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        # intrinsics[:2, :] /= 4.0 # FIXME - ONLY REQD for dtu_yao_eval.py 
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1])

        if len(lines[11].split()) >= 3:
            num_depth = lines[11].split()[2]
            depth_max = depth_min + int(float(num_depth)) * depth_interval
            depth_interval = (depth_max - depth_min) / self.ndepths

        depth_interval *= interval_scale

        return intrinsics, extrinsics, depth_min, depth_interval

    def read_depth(self, filename):
        # read pfm depth file
        return np.array(read_pfm(filename)[0], dtype=np.float32)

    def read_rescale_crop_img(self, img_fname, intrinsics, img_res=(512,640)):
        
        base_image_size = 64
        
        # Read image
        img = Image.open(img_fname) #  Colors=RGB
        w_src, h_src = img.size  # Warning: width first with pillow
        if DEBUG: print(f"[DATALOADER] img read dims: ({h_src},{w_src})")        
        
        # evaluate scale factor 
        h_target, w_target = img_res
        if DEBUG: print(f"[DATALOADER] img target dims: ({h_target},{w_target})")    
        
        h_scale = float(h_target) / h_src
        w_scale = float(w_target) / w_src
        if h_scale > 1 or w_scale > 1:
            print("[read_rescale_crop_img] max_h, max_w should < W and H (image resolution should only be reduced)!")
            exit()
        resize_scale = h_scale
        if w_scale > h_scale:
            resize_scale = w_scale
        if DEBUG: print(f"[DATALOADER] resize_scale: {resize_scale}")        
        
        # rescale image
        img_rescaled = img.resize(size=(int(w_src*resize_scale), int(h_src*resize_scale)), resample=Image.BILINEAR ) # Warning: width first with pillow
        w_rescaled, h_rescaled = img_rescaled.size        
        if DEBUG: print(f"[DATALOADER] img rescaled dims: ({h_rescaled}, {w_rescaled})")
        
        # rescale intrinsics
        intrinsics[:2,:] *= resize_scale
        if DEBUG: print("[DATALOADER] rescaled intrinsics:\n", intrinsics)
        
        # determine if cropping needed (dims must be compatible with base_image_size)
        final_h = h_rescaled
        final_w = w_rescaled
        
        if final_h > h_target:
            final_h = h_target
        else:
            final_h = int(math.floor(h_target / base_image_size) * base_image_size)
            
        if final_w > w_target:
            final_w = w_target
        else:
            final_w = int(math.floor(w_target / base_image_size) * base_image_size)

        # evaluate cropping parameters 
        start_h = int(math.floor((h_rescaled - final_h) / 2))
        start_w = int(math.floor((w_rescaled - final_w) / 2))
        finish_h = start_h + final_h
        finish_w = start_w + final_w
        
        # crop img and intrinsics
        # img_cropped = img_rescaled[start_h:finish_h, start_w:finish_w] # for numpy
        croping_dims = (start_w, start_h, finish_w, finish_h)
        img_cropped = img_rescaled.crop(croping_dims)  # for pillow
        if DEBUG: 
            print(f"[DATALOADER] croping dims: (left, top, right, bottom)={croping_dims}")
            print(f"[DATALOADER] cropped img dims: ({img_cropped.size[1]}, {img_cropped.size[0]})")
        
        # crop intrinsics
        intrinsics[0,-1] -= start_w
        intrinsics[1,-1] -= start_h
        if DEBUG: print("[DATALOADER] Final intrinsics:\n", intrinsics)
        
        # convert pillow image to numpy
        np_img = np.array(img_cropped, dtype=np.float32) / 255.
        
        # checks shape
        # assert np_img.shape[:2] == img_res
        
        # check image has 3 channels (RGB), stack if only 1 channel
        if len(np_img.shape) == 2:
            np_img = np.dstack((np_img, np_img, np_img))
            
        if DEBUG: print(f"[DATALOADER] final img shape: {np_img.shape}")

        return np_img, intrinsics


    def __getitem__(self, idx):
        
        global s_h, s_w
        meta = self.metas[idx]
        scan, ref_view, src_views, scene_name = meta
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.nviews - 1] # nviews is no of views to evaluate depth including the ref view
        if DEBUG: print ("[DATALOADER] view_ids: ", view_ids)

        imgs = []
        depth_values = None
        proj_matrices = []

        for i, vid in enumerate(view_ids):
            
            if DEBUG: print("[DATALOADER] ========== idx=",i," viewID: ",vid )
                
            # Define filenames (Nb: Blender image filenames go from 0 to N-1, not 1~N as in DTU)
            if self.dsname == "dtu":
                img_filename = os.path.join(self.datapath, self.img_folder.format(scan, vid+1, self.lighting))
            else:
                img_filename = os.path.join(self.datapath, self.img_folder.format(scan, vid+1, self.lighting))
            proj_mat_filename = os.path.join(self.datapath, self.cam_folder).format(vid)
            
            if DEBUG:
                print("[DATALOADER] img filenames:", img_filename)
                print("[DATALOADER] cam filenames:", proj_mat_filename)
            
            # Read camera
            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename, interval_scale=self.interval_scale)
            if DEBUG: print("[DATALOADER] intrinsics read:\n", intrinsics)
                
            # Read & rescale image input
            img, intrinsics = self.read_rescale_crop_img(img_filename, intrinsics, (self.max_h, self.max_w))                

            if self.dsname == "dtu" and self.cam_folder[:13] == 'Cameras/train':
                intrinsics[:2] *= 4
                
            # Append images, transpose from H,W,C to C,H,W ?
            imgs.append(img.transpose(2,0,1))
            
            # extrinsics, intrinsics
            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)  #
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics
            proj_matrices.append(proj_mat)

            if i == 0:  # reference view
                depth_values = np.arange(depth_min, depth_interval * (self.ndepths - 0.5) + depth_min, depth_interval, dtype=np.float32)
                # depth_values = np.arange(depth_min, depth_min + depth_interval * self.ndepths , depth_interval, dtype=np.float32)
                if DEBUG: print("[DATALOADER] Min/Max depth values:", depth_values.min(), depth_values.max())
            
        # evaluate proj_matrices for each stage
        proj_matrices = np.stack(proj_matrices)
        stage4_pjmats = proj_matrices.copy()
        stage3_pjmats = proj_matrices.copy() 
        stage3_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] / 2.0
        stage2_pjmats = proj_matrices.copy() 
        stage2_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] / 4.0
        stage1_pjmats = proj_matrices.copy() 
        stage1_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] / 8.0

        proj_matrices_ms = {
            "stage1": stage1_pjmats,
            "stage2": stage2_pjmats,
            "stage3": stage3_pjmats,
            "stage4": stage4_pjmats  
        }
        
        if DEBUG:
            print("[DATALOADER]  ref img dims:",imgs[0].shape) 
            print("[DATALOADER]  src img dims:",imgs[1].shape) 
            
            print("[DATALOADER] stg4 intrinsics:\n", proj_matrices_ms["stage4"][0][1])             
            print("[DATALOADER] stg1 intrinsics:\n", proj_matrices_ms["stage1"][0][1]) 

            
        return {"imgs": imgs,
                "proj_matrices": proj_matrices_ms,
                "depth_values": depth_values,
                "filename": scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}"}
