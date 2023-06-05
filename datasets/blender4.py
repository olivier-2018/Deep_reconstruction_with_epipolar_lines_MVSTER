from torch.utils.data import Dataset
import numpy as np
import os, cv2, time, math
from PIL import Image
from datasets.data_io import *
from torchvision import transforms


DEBUG = False # local debugging

class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, interval_scale=1.06, **kwargs):
        super(MVSDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = 192  # Hardcode
        self.interval_scale = interval_scale
        self.kwargs = kwargs
        self.rt = kwargs.get("rt", False)
        self.use_raw_train = kwargs.get("use_raw_train", False)
        self.color_augment = transforms.ColorJitter(saturation = 0.4 , contrast=0.5, brightness=0.6, hue=0.01)
        self.pair_fname = kwargs.get("pair_fname", "pair.txt")
        Nlights_str = kwargs.get("Nlights", "1:1")
        self.Nlights = int(Nlights_str.split(":")[0].replace("(","").replace(")",""))
        self.TotLights = int(Nlights_str.split(":")[1])
        
        assert self.mode in ["train", "val", "test"]
        
        self.metas = self.build_list()

    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]
        
        # Load pair data
        pair_file = os.path.join(self.datapath, self.pair_fname)
        
        if DEBUG: print ("[DATALOADER] (init) Pairfile: ", pair_file)
        for scan in scans:
            # read the pair file
            with open(pair_file) as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for _ in range(num_viewpoint):  
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    # light conditions 
                    if self.Nlights == 0:
                        metas.append((scan, 0, ref_view, src_views))
                    elif self.Nlights < 0:
                        metas.append((scan, -self.Nlights, ref_view, src_views))
                    else:                
                        if self.mode == "val":
                            assert self.Nlights >= 2, "Eval number of lights must be >2 " 
                            Nlights_val = random.sample(range(self.Nlights), k=2) # sample w/o replacements
                            for light_idx in Nlights_val:    
                                metas.append((scan, light_idx, ref_view, src_views))                            
                        else:
                            assert self.Nlights <= self.TotLights, "Training number of lights must be < total number of lights in dataset" 
                            Nlights_train = random.sample(range(self.TotLights), k=self.Nlights) # sample w/o replacements
                            for light_idx in Nlights_train:    
                                metas.append((scan, light_idx, ref_view, src_views))
                                
        print("[DataLoader] Mode:{}, Ncams:{}, #metas:{} ".format(self.mode, num_viewpoint, len(metas)))
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1]) * self.interval_scale
        return intrinsics, extrinsics, depth_min, depth_interval

    def read_img(self, filename):
        img = Image.open(filename) 
        if self.mode == 'train':
            img = self.color_augment(img)
            if random.random() < 0.1: img.convert('L')
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.
        return np_img

    def read_mask_hr(self, filename):        
        img = Image.open(filename)
        np_img = np.array(img, dtype=np.float32)
        np_img = (np_img > 10).astype(np.float32)        
        h, w = np_img.shape
        mask_ms = {
            "stage1": cv2.resize(np_img, (w//8, h//8), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(np_img, (w//4, h//4), interpolation=cv2.INTER_NEAREST),
            "stage3": cv2.resize(np_img, (w//2, h//2), interpolation=cv2.INTER_NEAREST),
            "stage4": np_img,
        }
        return mask_ms


    def read_depth_hr(self, filename, scale):
        # read pfm depth file        
        depth = np.array(read_pfm(filename)[0], dtype=np.float32) * scale            
        h, w = depth.shape
        depth_ms = {
            "stage1": cv2.resize(depth, (w//8, h//8), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(depth, (w//4, h//4), interpolation=cv2.INTER_NEAREST),
            "stage3": cv2.resize(depth, (w//2, h//2), interpolation=cv2.INTER_NEAREST),
            "stage4": depth,
        }
        return depth_ms

    def __getitem__(self, idx):
        meta = self.metas[idx]
        
        # use only the reference view and first nviews-1 source views
        scan, light_idx, ref_view, src_views = meta
        # if DEBUG: print("[DATALOADER] meta:",meta)
        
        # Select ref view and src_views (event. using random training)
        if self.mode == 'train' and self.rt:
            num_src_views = len(src_views)
            index = random.sample(range(num_src_views), self.nviews - 1)
            view_ids = [ref_view] + [src_views[i] for i in index]
            scale = random.uniform(0.8, 1.25)
            # scale = random.uniform(0.9, 1.1)
        else:
            view_ids = [ref_view] + src_views[:self.nviews - 1]
            scale = 1
        
        # Init
        imgs = []
        mask = None
        depth_values = None
        proj_matrices = []
        
        
        # Setting suffix for input folders, H & W are just to assert dimensions
        if self.use_raw_train:
            suffix = "_1024x1280"
            H, W = 1024, 1280
            print ("use_raw should not be used for training")
            exit()
        else:
            suffix = "_512x640"
            H, W = 512, 640
        
        # Read mask only once
        mask_filename_hr = os.path.join(self.datapath, 'Depths'+suffix, '{}/depth_mask_{:0>3}.png'.format(scan, ref_view))         
        mask_ms = self.read_mask_hr(mask_filename_hr)
        h, w = mask_ms["stage4"].shape[:2]
        assert (h, w) == (H, W), "Image dimension doubtful. Please generate masks with dims {}x{} !".format(H, W)
        if DEBUG:
            print("[DATALOADER] mask dims:", mask_ms["stage4"].shape) 
        
        # Read depth only once
        depth_filename_hr = os.path.join(self.datapath, 'Depths'+suffix, '{}/depth_map_{:0>3}.pfm'.format(scan, ref_view))
        depth_ms = self.read_depth_hr(depth_filename_hr, scale)
        h, w = depth_ms["stage4"].shape[:2]
        assert (h, w) == (H, W), "Image dimension doubtful. Please generate depthmaps with dims {}x{} !".format(H, W)
        if DEBUG:
            print("[DATALOADER] depth dims:", depth_ms["stage4"].shape)
            
        # Read all ref and associated src images as stated in "pair.txt" 
        for i, vid in enumerate(view_ids):                        
            
            if DEBUG: print("[DATALOADER] GET_ITEM: idx=",i)
                
            # Define filenames
            # NOTE: Blender image filenames are from 0 to N-1 (not 1~N)
            img_filename = os.path.join(self.datapath, 'Rectified'+suffix, '{}/rect_C{:0>3}_L{:0>2}.png'.format(scan, vid, light_idx))            
            proj_mat_filename = os.path.join(self.datapath, 'Cameras'+suffix, '{:0>8}_cam.txt').format(vid) 
            
            if DEBUG:
                print("[DATALOADER] GET_ITEM: filenames:")
                print (img_filename)
                print (mask_filename_hr)
                print (depth_filename_hr)
                print (proj_mat_filename)
            
            # Read image and process images if required
            img = self.read_img(img_filename)
            h, w = img.shape[:2]
            assert (h, w) == (H, W), "Image dimension doubtful. Please generate images with dims {}x{} !".format(H, W)
            if DEBUG: print("[DATALOADER] read img dims:", img.shape)
                
            # Read camera parameters
            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename)
            if DEBUG: print("[DATALOADER] read intrinsics:\n", intrinsics)
            
            # Adjust extrinsics translation vector if robust training  
            if self.rt:
                extrinsics[:3,3] *= scale

            # Read mask and depth for ref image
            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics
            proj_matrices.append(proj_mat)
            imgs.append(img.transpose(2,0,1))

        # Calculate depth virtual planes distances
        depth_max = depth_interval * self.ndepths + depth_min
        depth_values = np.array([depth_min * scale, depth_max * scale], dtype=np.float32) # Is this correct?
        # depth_values = np.arange(depth_min, depth_min + depth_interval * self.ndepths, depth_interval, dtype=np.float32) # trial OLI        
        
        # create proj_matrices_ms
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
            print("[DATALOADER] stg4 depth dims:",depth_ms["stage4"].shape) 
            print("[DATALOADER] stg4 mask dims:",mask_ms["stage4"].shape) 
            
            print("[DATALOADER] stg1 intrinsics:\n", proj_matrices_ms["stage1"][0][1]) 
            print("[DATALOADER] stg1 depth dims:",depth_ms["stage1"].shape) 
            print("[DATALOADER] stg1 mask dims:",mask_ms["stage1"].shape) 
            
        return {"imgs": imgs,                       # list of imgs: (Nv C H W)
                "proj_matrices": proj_matrices_ms,  # dict: 4 stages of (Nv 2 4 4)
                "depth": depth_ms,                  # dict: 4 stages of (H W)
                "depth_values": depth_values,       # array
                "mask": mask_ms }                      # dict: 4 stages of (H W)
        