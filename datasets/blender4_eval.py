from torch.utils.data import Dataset
import numpy as np
import os, cv2, time
from PIL import Image
from datasets.data_io import *

s_h, s_w = 0, 0

class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, interval_scale=1.06, **kwargs):
        super(MVSDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = 192  # Hardcode
        self.interval_scale = interval_scale
        self.max_h, self.max_w = kwargs["max_h"], kwargs["max_w"]
        self.fix_res = kwargs.get("fix_res", False)  #whether to fix the resolution of input image.
        self.fix_wh = False
        self.use_raw_train = kwargs.get("use_raw_train", False)
        self.pair_fname = kwargs.get("pair_fname", "pair.txt")
        print ("=== MVSDataset init, pair file: ", self.pair_fname)
        self.lighting = kwargs.get("lighting", 3)
        self.debug = kwargs.get("debug", False)        
        self.dsname = kwargs.get("dsname", "blender")
        
        assert self.mode == "test"
        self.metas = self.build_list()

    def build_list(self):
        metas = []
        scans = self.listfile

        interval_scale_dict = {}
        
        # scans         
        for scan in scans:
            # determine the interval scale of each scene. default is 1.06
            if isinstance(self.interval_scale, float):
                interval_scale_dict[scan] = self.interval_scale
            else:
                interval_scale_dict[scan] = self.interval_scale[scan]

            pair_file = self.pair_fname
            print ("[blender4_eval] build_list:  Pair file used: ", pair_file)
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    # filter by no src view and fill to nviews
                    if len(src_views) > 0:
                        if len(src_views) < self.nviews:
                            print("[blender4_eval]build_list: Number of src views ({}) < requested num_views ({}). Padding with first src view.".format(len(src_views), self.nviews))
                            src_views += [src_views[0]] * (self.nviews - len(src_views))
                        metas.append((scan, ref_view, src_views, scan))
                    if self.debug > 0:
                        print ("[blender4_eval]build_list: idx:", view_idx, "refview:",ref_view,"srcview:", src_views)

        self.interval_scale = interval_scale_dict
        print("[blender4_eval]build_list: No of metas:", len(metas), ", itvl_scale:{}".format(self.interval_scale))
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

    def read_img(self, filename):
        img = Image.open(filename)
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.

        return np_img

    def read_depth(self, filename):
        # read pfm depth file
        return np.array(read_pfm(filename)[0], dtype=np.float32)

    def scale_mvs_input(self, img, intrinsics, max_w, max_h, base=64):
        """ Resize img & intrinsecs in case input image is larger than (h_max, w_max).
            Will determine which side is longer and rescale to a dimension in base 64.
        """
        
        h, w = img.shape[:2]
        if h > max_h or w > max_w:
            # setup scale factor ONLY if img is larger than max values
            scale = 1.0 * max_h / h
            if scale * w > max_w:
                scale = 1.0 * max_w / w
            # adjust h & w to be in base64
            new_w, new_h = scale * w // base * base, scale * h // base * base
        else:
            # adjust h & w to be in base64
            new_w, new_h = 1.0 * w // base * base, 1.0 * h // base * base

        # rescale intrinsics
        scale_w = 1.0 * new_w / w
        scale_h = 1.0 * new_h / h
        intrinsics[0, :] *= scale_w
        intrinsics[1, :] *= scale_h

        # rescale image
        img = cv2.resize(img, (int(new_w), int(new_h)))

        return img, intrinsics

    def __getitem__(self, idx):
        global s_h, s_w
        meta = self.metas[idx]
        scan, ref_view, src_views, scene_name = meta
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.nviews - 1] # nviews is no of views to evaluate depth including the ref view
        if self.debug > 0:
            print ()
            print ("[Blender_eval] GET_ITEM view_ids: ", view_ids)

        imgs = []
        depth_values = None
        proj_matrices = []

        for i, vid in enumerate(view_ids):
            
            if self.debug > 0:
                print()
                print("[Blender_eval] GET_ITEM: idx=",i," viewID: ",vid )
                
            # Setting suffix for input folders, H & W are just to assert dimensions
            if self.use_raw_train:
                suffix = "_1024x1280"
                H, W = 1024, 1280
            else:
                suffix = "_512x64"
                H, W = 512, 640

            # Define filenames (Nb: Blender image filenames go from 0 to N-1, not 1~N as in DTU)
            if self.dsname == "blender":
                img_filename = os.path.join(self.datapath, 'Rectified'+suffix, '{}/rect_C{:0>3}_L{:0>2}.png'.format(scan, vid, self.lighting))
                proj_mat_filename = os.path.join(self.datapath, 'Cameras'+suffix, '{:0>8}_cam.txt').format(vid)
            elif self.dsname == "merlin":
                img_filename = os.path.join(self.datapath, 'Rectified'+suffix, '{}/{:0>8}.png'.format(scan, vid)) 
                proj_mat_filename = os.path.join(self.datapath, 'Cameras'+suffix, '{}/{:0>8}_cam.txt').format(scan, vid)
            else:
                print("[Blender_eval] GET_ITEM: Data set name not recognized - BREAK")
                break
            
            if self.debug > 0:
                print("[Blender_eval] GET_ITEM: filenames:")
                print (img_filename)
                print (proj_mat_filename)
            
            # Read image
            img = self.read_img(img_filename) # H,W,C
            if self.debug > 0:
                print("[Blender_eval] GET_ITEM: img raw size:", img.shape)
            
            # Read camera
            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename, interval_scale=self.interval_scale[scene_name])
            if self.debug > 0:
                print("[Blender_eval] GET_ITEM: intrinsics:", intrinsics[0,:])
                
            # scale input
            img, intrinsics = self.scale_mvs_input(img, intrinsics, self.max_w, self.max_h) 
            if self.debug > 0:
                print("[Blender_eval] GET_ITEM: scale_mvs_input")
                print("[Blender_eval] GET_ITEM: img size:", img.shape)
                print("[Blender_eval] GET_ITEM: intrinsics:", intrinsics[0,:])
                
            if self.fix_res:
                # using the same standard height or width in entire scene.
                s_h, s_w = img.shape[:2]
                self.fix_res = False
                self.fix_wh = True

            if i == 0:
                if not self.fix_wh:
                    # using the same standard height or width in each nviews.
                    s_h, s_w = img.shape[:2]

            # resize to standard height or width
            c_h, c_w = img.shape[:2]
            assert (c_h,c_w) == (H,W), "Image dimension doubtful in blender4_eval.py line172. Please generate images with dims {}x{} !".format(H, W) 
            if (c_h != s_h) or (c_w != s_w):
                print ("WARNING: image rescale to to standard height or width, please check blender4_eval.py line174")
                scale_h = 1.0 * s_h / c_h
                scale_w = 1.0 * s_w / c_w
                img = cv2.resize(img, (s_w, s_h))
                intrinsics[0, :] *= scale_w
                intrinsics[1, :] *= scale_h

            # Append images, transpose from H,W,C to C,H,W ?
            imgs.append(img.transpose(2,0,1))
            
            # extrinsics, intrinsics
            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)  #
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics
            proj_matrices.append(proj_mat)
            
            if self.debug > 0:
                print("[Blender_eval] GET_ITEM: proj_mat:\n", proj_mat)

            if i == 0:  # reference view
                # depth_values = np.arange(depth_min, depth_interval * (self.ndepths - 0.5) + depth_min, depth_interval, dtype=np.float32)
                depth_values = np.arange(depth_min, depth_min + depth_interval * self.ndepths , depth_interval, dtype=np.float32)

            if self.debug > 0:
                print("[Blender_eval] GET_ITEM: Min/Max depth values:", depth_values.min(), depth_values.max())
                 
        if self.debug > 0:
            print("=> coming out of viewIDs")   
            
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
        
        if self.debug > 0:
            print("[Blender_eval] GET_ITEM: proj_matrices_ms stg4 shape:", proj_matrices_ms["stage4"].shape) # (Nv, camparams, 4, 4) with camparams = 2 for in-&extrinsics
            print("[Blender_eval] GET_ITEM: proj_matrices_ms stg4 intrinsics:\n", proj_matrices_ms["stage4"][0][1]) # refView, intrinsics, 4, 4
            
        return {"imgs": imgs,
                "proj_matrices": proj_matrices_ms,
                "depth_values": depth_values,
                "filename": scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}"}