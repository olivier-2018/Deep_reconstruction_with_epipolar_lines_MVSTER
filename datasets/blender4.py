from torch.utils.data import Dataset
import numpy as np
import os, cv2, time, math
from PIL import Image
from datasets.data_io import *
from torchvision import transforms

# the DTU dataset preprocessed by Yao Yao (only for training)
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
        self.lightings = kwargs.get("lightings", 7)
        self.debug = kwargs.get("debug", False)
        
        print ("[MVSDataset] INIT: pair file: ", self.pair_fname)

        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()

    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        # scans
        # pair_file = "pair.txt"
        pair_file = self.pair_fname
        if self.debug > 0:
            print ("[MVSDataset] BUILD_LIST: Pair file used: ", pair_file)
        
        # Manage multiple of single light sources
        if self.lightings > 0: # if positive number, then light_idx will iterate 
            light_srcs_qty = self.lightings
            light_offset = 0
        elif self.lightings < 0: # if negative number, then light_idx will be a fixed value
            light_srcs_qty = 1
            light_offset = -self.lightings
        elif self.lightings == 0: # if null, then light_idx is set to 3 
            light_srcs_qty = 1
            light_offset = 3
            
        for scan in scans:
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    # light conditions 0-6
                    for light_idx in range(light_srcs_qty):
                        metas.append((scan, light_idx+light_offset, ref_view, src_views))
        # print("dataset", self.mode, "metas:", len(metas))
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

    # def crop_img(self, img):
    #     raw_h, raw_w = img.shape[:2]
    #     start_h = (raw_h-1024)//2
    #     start_w = (raw_w-1280)//2
    #     return img[start_h:start_h+1024, start_w:start_w+1280, :]  # 1024, 1280, C

    # def prepare_img(self, hr_img):
    #     h, w = hr_img.shape         # image should arrive at 1024x1280
    #     if not self.use_raw_train:
    #         # w1600-h1200-> 800-600 ; crop -> 640, 512; downsample 1/4 -> 160, 128
    #         # downsample
    #         hr_img_ds = cv2.resize(hr_img, (w//2, h//2), interpolation=cv2.INTER_NEAREST)
    #         h, w = hr_img_ds.shape
    #         target_h, target_w = 512, 640
    #         start_h, start_w = (h - target_h)//2, (w - target_w)//2
    #         hr_img_crop = hr_img_ds[start_h: start_h + target_h, start_w: start_w + target_w]
    #     elif self.use_raw_train:
    #         hr_img_crop = hr_img[h//2-1024//2:h//2+1024//2, w//2-1280//2:w//2+1280//2]  # 1024, 1280, c
    #     return hr_img_crop

    def read_mask_hr(self, filename):
        
        img = Image.open(filename)
        np_img = np.array(img, dtype=np.float32)
        np_img = (np_img > 10).astype(np.float32)

        # if  self.use_raw_train:
        #     mask = np_img
        # else:
        #     mask = cv2.resize(np_img, (w//2, h//2), interpolation=cv2.INTER_NEAREST) # bilinear?
        
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
        #w1600-h1200-> 800-600 ; crop -> 640, 512; downsample 1/4 -> 160, 128
        
        depth = np.array(read_pfm(filename)[0], dtype=np.float32) * scale
        # if  self.use_raw_train:
        #     depth = depth_hr
        # else:
        #     depth = cv2.resize(depth_hr, (w//2, h//2), interpolation=cv2.INTER_NEAREST) # bilinear?
            
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
        if self.debug > 0:
                print("[MVSDataset] meta:",meta)
        
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
        
        # Read all ref and associated src images as stated in "pair.txt" 
        for i, vid in enumerate(view_ids):                        
            
            if self.debug > 0:
                print("[MVSDataset] GET_ITEM: idx=",i)
                
            # Setting suffix for input folders, H & W are just to assert dimensions
            if self.use_raw_train:
                suffix = "_1024x1280"
                H, W = 1024, 1280
            else:
                suffix = "_512x640"
                H, W = 512, 640
            
            # Define filenames
            # NOTE: Blender image filenames are from 0 to N-1 (not 1~N)
            img_filename = os.path.join(self.datapath, 'Rectified'+suffix, '{}/rect_C{:0>3}_L{:0>2}.png'.format(scan, vid, light_idx))
            mask_filename_hr = os.path.join(self.datapath, 'Depths'+suffix, '{}/depth_mask_{:0>3}.png'.format(scan, vid)) 
            depth_filename_hr = os.path.join(self.datapath, 'Depths'+suffix, '{}/depth_map_{:0>3}.pfm'.format(scan, vid))
            proj_mat_filename = os.path.join(self.datapath, 'Cameras'+suffix, '{:0>8}_cam.txt').format(vid) 
            
            if self.debug > 0:
                print("[MVSDataset] GET_ITEM: filenames:")
                print (img_filename)
                print (mask_filename_hr)
                print (depth_filename_hr)
                print (proj_mat_filename)
            
            # Read image and process images if required
            img = self.read_img(img_filename)
            h, w = img.shape[:2]
            assert (h, w) == (H, W), "Image dimension doubtful. Please generate images with dims {}x{} !".format(H, W)
            if self.debug > 0:
                print("[MVSDataset] GET_ITEM: img raw size:", img.shape)
           
            # if not self.use_raw_train:  # Not required since all inputs are read in a consistent with each others
            #     img = cv2.resize(img, (w//2, h//2), interpolation=cv2.INTER_NEAREST) # bilinear?
            #     h, w = img.shape
            #     assert (h, w) == (512, 640), "Image dimension doubtful after resizing !"
                
            # Read camera parameters
            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename)
            if self.debug > 0:
                print("[MVSDataset] GET_ITEM: intrinsics:", intrinsics[0,:])
            
            # Adjust extrinsics translation vector if robust training  
            if self.rt:
                extrinsics[:3,3] *= scale
            # if not self.use_raw_train:
            #     intrinsics[:2, :] /= 2.0 # camera intrinsic already generate for high res images  # no need, already read with the matching intrinsics

            # Read mask and depth only once
            if i == 0:
                # Read depth & mask
                mask_read_ms = self.read_mask_hr(mask_filename_hr)
                h, w = mask_read_ms["stage4"].shape[:2]
                assert (h, w) == (H, W), "Image dimension doubtful. Please generate masks with dims {}x{} !".format(H, W)
                    
                # get depth values
                depth_ms = self.read_depth_hr(depth_filename_hr, scale)
                h, w = depth_ms["stage4"].shape[:2]
                assert (h, w) == (H, W), "Image dimension doubtful. Please generate depthmaps with dims {}x{} !".format(H, W)
                if self.debug > 0:
                    print("[MVSDataset] GET_ITEM: mask:", mask_read_ms["stage4"].shape)
                    print("[MVSDataset] GET_ITEM: depth:", depth_ms["stage4"].shape)
                    
                # Calculate depth virtual planes distances
                depth_max = depth_interval * self.ndepths + depth_min
                depth_values = np.array([depth_min * scale, depth_max * scale], dtype=np.float32) # Is this correct?
                # depth_values = np.arange(depth_min, depth_min + depth_interval * self.ndepths, depth_interval, dtype=np.float32) # trial OLI
                mask = mask_read_ms

            # Read mask and depth for ref image
            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics
            proj_matrices.append(proj_mat)
            imgs.append(img.transpose(2,0,1))
            
            if self.debug > 0:
                print("[MVSDataset] GET_ITEM: proj_mat:\n", proj_mat)

        #all

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
        if self.debug > 0:
            print("[MVSDataset] GET_ITEM: proj_matrices_ms stg4 intrinsics:\n", proj_matrices_ms["stage4"].shape) # (Nv, camparams, 4, 4) with camparams = 2 for in-&extrinsics
            print("[MVSDataset] GET_ITEM: proj_matrices_ms stg4 intrinsics:\n", proj_matrices_ms["stage4"][0][1]) # refView, intrinsics, 4, 4
            
        return {"imgs": imgs,                       # list of imgs: (Nv C H W)
                "proj_matrices": proj_matrices_ms,  # dict: 4 stages of (Nv 2 4 4)
                "depth": depth_ms,                  # dict: 4 stages of (H W)
                "depth_values": depth_values,       # array
                "mask": mask }                      # dict: 4 stages of (H W)
        