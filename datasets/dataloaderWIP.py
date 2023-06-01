from torch.utils.data import Dataset
import numpy as np
import os, cv2, time, math
from PIL import Image
from datasets.data_io import *
from torchvision import transforms


DEBUG = True # local debugging

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
        self.color_augment = transforms.ColorJitter(saturation = 0.4 , contrast=0.5, brightness=0.6, hue=0.01)
        self.pair_fname = kwargs.get("pair_fname", "pair.txt")
        self.Nlights = kwargs.get("Nlights", "1:1")
        self.dataset = kwargs.get("dataset", "dtu")        
        self.resolution = kwargs.get("resolution", "512x640") 
        self.max_dim = kwargs.get("max_dim", "640") 
        self.minD = kwargs.get("minD", 425)
        self.maxD = kwargs.get("maxD", 935)
        
        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()
        
        
        # Set datasets paths         
        if self.dataset == "dtu": 
            self.pair_path = os.path.join(self.datapath, self.pairfile)   
            if self.mode in ["train", "val"]:
                self.img_folder = 'Rectified/{}_train/rect_{:0>3}_{}_r5000.png'     # 640x512
                self.depth_folder = 'Depths_raw/{}_train/depth_map_{:0>4}.pfm'      # 1600x1200
                self.mask_folder = 'Depths_raw/{}_train/depth_mask_{:0>4}.png'      # 1600x1200
                self.cam_folder = 'Cameras/train/{:0>8}_cam.txt'                    # 160x128
            else:
                self.img_folder = 'Rectified_raw/{}/rect_{:0>3}_{}_r5000.png'       # 1600x1200
                self.depth_folder = 'Depths_raw/{}/depth_map_{:0>4}.pfm'            # 1600x1200
                self.mask_folder = 'Depths_raw/{}/depth_mask_{:0>4}.png'            # 1600x1200
                self.cam_folder = 'Cameras/{:0>8}_cam.txt'                          # 1600x1200
                
        elif self.dataset == "blender":  
            self.pair_path = os.path.join(self.datapath, self.pairfile) 
            if self.mode in ["train", "val"]:
                self.img_folder = 'Rectified_'+self.resolution+'/{}/rect_C{:0>3}_L{:0>2}.png'
                self.depth_folder = 'Depths_'+self.resolution+'/{}/depth_map_{:0>3}.pfm'
                self.mask_folder = 'Depths_'+self.resolution+'/{}/depth_mask_{:0>3}.png'
                self.cam_folder = 'Cameras_'+self.resolution+'/{:0>8}_cam.txt'
            else:
                self.img_folder = 'Rectified_'+self.resolution+'/{}/rect_C{:0>3}_L{:0>2}.png'
                self.depth_folder = 'Depths_'+self.resolution+'/{}/depth_map_{:0>3}.pfm'
                self.mask_folder = 'Depths_'+self.resolution+'/{}/depth_mask_{:0>3}.png'
                self.cam_folder = 'Cameras_'+self.resolution+'/{:0>8}_cam.txt'
                
        elif self.dataset == "bin":  
            self.pair_path = os.path.join(self.datapath, "../..", self.pairfile) 
            if self.mode not in ["train", "val"]:
                # suffix = self.pairfile.split("/")[0].split("_")[1]
                # self.img_folder = 'Rectified_'+suffix+'/{}/{:0>8}.png'
                # self.cam_folder = 'Cameras_'+suffix+'/{:0>8}_cam.txt'
                self.img_folder = 'Rectified/{}/{:0>8}.png'
                self.cam_folder = 'Cameras/{:0>8}_cam.txt'
                self.depth_folder = ''
                self.mask_folder = ''
            else:
                print(f"Mode {self.mode} not compatible with dataset {self.dataset}.")
                exit()                
        else:
            print("dataset unknown.")
            exit()


    def build_list(self):
        
        metas = []
        with open(self.scan_list) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        # scans (list of scans/scenes)
        for scan in scans:
            # read the pair file
            with open(self.pair_path) as f:
                num_viewpoint = int(f.readline())
                # viewpoints
                for _ in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    # light conditions 0-6
                    if self.Nlights == 0:
                        light_idx = 0 
                        metas.append((scan, light_idx, ref_view, src_views))
                    elif self.Nlights < 0:
                        light_idx = -self.Nlights
                        metas.append((scan, light_idx, ref_view, src_views))
                    else:
                        for light_idx in self.Nlights:    
                            metas.append((scan, light_idx, ref_view, src_views))
                                    
        print("[MVSDataset] mode:{}, List:{}, # metas: {}".format(self.mode, self.listfile, len(metas)))
        
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
        img = Image.open(filename) # read RGB
        if self.mode == 'train':
            img = self.color_augment(img)
            if random.random() < 0.1: img.convert('L')
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.
        return np_img
    
    def scale_to_max_dim(self, image: np.ndarray, max_dim: int):
        original_height = image.shape[0]
        original_width = image.shape[1]
        scale = max_dim / max(original_height, original_width)
        if 0 < scale < 1:
            width = int(scale * original_width)
            height = int(scale * original_height)
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        return image, original_height, original_width

    def read_scale_img(self, filename: str, max_dim: int = -1):
        image = Image.open(filename) # read RGB
        # scale 0~255 to 0~1
        np_image = np.array(image, dtype=np.float32) / 255.0
        return self.scale_to_max_dim(np_image, max_dim)
        
    def read_scale_map(self, path: str, max_dim: int = -1) -> np.ndarray:
        in_map, _ = read_pfm
        return self.scale_to_max_dim(in_map, max_dim)[0]

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
        
        if DEBUG: print("idx: ", index)
        
        # use only the reference view and first nviews-1 source views
        scan, light_idx, ref_view, src_views = self.metas[idx]
        if DEBUG: print("[MVSDataset] meta:",self.metas[idx])
        
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
        if DEBUG: print ("view_ids: ",view_ids)
        
        # Read DEPTH and MASK (for training and val ONLY)
        if self.mode in ["train", "val"]:
            if self.dataset == "dtu":   
                # DEPTH, read from "Depths_raw" folder (i.e 1600x1200) since shape is 160x128 in "Depths" folder
                depth_gt_filename = os.path.join(self.data_path, self.depth_folder.format(scan, view_ids[0]))  
                if DEBUG:  print ("[Dataloader] depth_gt_filename: ", depth_gt_filename)
                assert os.path.isfile(depth_gt_filename), "GT depth file not found"             
                depth_gt = self.read_scale_map(depth_gt_filename, 800)         # scaled from 1600x1200 by factor 2 to 800x600
                depth_gt = depth_gt[44:556, 80:720]                             # adjusted from 800x600 to 640x512
                # MASK
                mask = depth_gt >= depth_min                        # True/False
                # mask = mask * 1.0                                   # 0.0 / 1.0
            else:  
                # DEPTH 
                depth_gt_filename = os.path.join(self.data_path, self.depth_folder.format(scan, view_ids[0]))  
                if DEBUG:  print ("[Dataloader] depth_gt_filename: ", depth_gt_filename)           
                assert os.path.isfile(depth_gt_filename), "GT depth file not found"    
                depth_gt = self.read_scale_map(depth_gt_filename, self.max_dim)
                # MASK                
                # mask_filename = depth_gt_filename.replace("depth_map", "depth_mask").replace(".pfm", ".png")
                # if DEBUG:  print ("mask_filename: ", mask_filename)
                # assert os.path.isfile(depth_gt_filename), "GT depth file not found"    
                # mask = read_image(mask_filename, self.max_dim)[0]        # already read as  0.0 / 1.0
                # MASK option2
                mask = depth_gt >= depth_min                        # True/False
            if DEBUG: 
                print(f"[Dataloader] depth shape: {depth_gt.shape}")
                print(f"[Dataloader] mask shape: {mask.shape}")

            if self.mode == 'train' and self.rt:
                depth_gt *= scale
        
        # Init
        imgs = []
        mask = None
        depth_values = None
        proj_matrices = []
        
        # Read all ref and associated src images as stated in "pair.txt" 
        for i, vid in enumerate(view_ids):                        
            
            if DEBUG: print("[MVSDataset] GET_ITEM: idx=",i)
            
            # Filenames
            if self.dataset == "dtu":
                img_filename = os.path.join(self.datapath, self.img_folder.format(scan, vid+1 , light_idx))
            elif self.dataset == "blender":
                img_filename = os.path.join(self.datapath, self.img_folder.format(scan, vid , light_idx))
            elif self.dataset == "bin":
                img_filename = os.path.join(self.datapath, self.img_folder.format(scan, vid))
            else:
                print("Dataset not recognized.")
                exit()
            if DEBUG: print("[MVSDataset] img filename:", img_filename)
            cam_filename = os.path.join(self.datapath, self.cam_folder.format(vid))     
            if DEBUG: print("[MVSDataset] cam filename:", cam_filename)

            # Read camera parameters
            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(cam_filename)
            if self.mode == 'train' and self.rt: extrinsics[:3,3] *= scale       # Adjust extrinsics translation vector if robust training  
            if DEBUG: print("[MVSDataset] read intrinsics:\n", intrinsics[0,:])      
            
            # Modify intrinsics if needed      
            if self.dataset == "dtu" and self.mode in ["train", "val"] and self.cam_folder[:13] == "Cameras/train":  
                intrinsics[:2,:] *= 4   # 160x128 --> 512x640
            
            # Read image and process images if required
            image, original_h, original_w = self.read_scale_img(img_filename, self.max_dim)
            
            h, w = image.shape[:2]
            if DEBUG: print("[MVSDataset] GET_ITEM: img raw size:", image.shape)
            if self.mode in ["train", "val"] and (h, w) != (512, 640):
                print ("Image dimension must be 512x640")
           

            # Read mask and depth for ref image
            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics
            proj_matrices.append(proj_mat)
            imgs.append(image.transpose(2,0,1))
            
            if DEBUG:  print("[MVSDataset] proj_mat:\n", proj_mat)

        # Prepare Multi-Stage
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
            print("[MVSDataset] stg4 intrinsics:\n", proj_matrices_ms["stage4"][0][1]) # (Nv, camparams, 4, 4) with camparams = 2 for in-&extrinsics
            print("[MVSDataset] stg1 intrinsics:\n", proj_matrices_ms["stage1"][0][1]) # refView, intrinsics, 4, 4
            
        return {"imgs": imgs,                       # list of imgs: (Nv C H W)
                "proj_matrices": proj_matrices_ms,  # dict: 4 stages of (Nv 2 4 4)
                "depth": depth_ms,                  # dict: 4 stages of (H W)
                "depth_values": depth_values,       # array
                "mask": mask }                      # dict: 4 stages of (H W)
        