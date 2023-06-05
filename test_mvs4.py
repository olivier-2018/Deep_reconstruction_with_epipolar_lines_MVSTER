import argparse, os, time, sys, gc, cv2
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from datasets import find_dataset_def
from models import *
from utils import *
from datasets.data_io import read_pfm, save_pfm
from plyfile import PlyData, PlyElement
from PIL import Image
import open3d as o3d
# from torchinfo import summary as torchinfo_summary
# from torchsummary import summary as torchsummary

from multiprocessing import Pool
from functools import partial
import signal


DEBUG=False
cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Predict depth, filter, and fuse')
parser.add_argument('--model', default='mvsnet', help='select model')

parser.add_argument('--dataset', default='blender4_eval', help='select dataset ')
parser.add_argument('--dataset_name', default='blender', choices=["dtu", "blender", "bin"], help='select dataset filenames format')
parser.add_argument('--datapath', help='testing data dir for some scenes')
parser.add_argument('--data_resolution', type=str, default="_512x640", help='suffix to input data')
parser.add_argument('--testlist', help='testing scene list')



parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--outdir', default='./outputs', help='output dir')
parser.add_argument('--pair_fname', default='pair.txt', help='view pair combination filename')
parser.add_argument('--lighting', type=int, default=3, help='index of light source to be used for inference')


parser.add_argument('--ndepths', type=str, default="8,8,4,4", help='ndepths')
parser.add_argument('--depth_inter_r', type=str, default="0.5,0.5,0.5,1", help='depth_intervals_ratio')
parser.add_argument('--batch_size', type=int, default=1, help='testing batch size')

parser.add_argument('--interval_scale', type=float, required=True, help='the depth interval scale')
parser.add_argument('--max_h', type=int, default=512, help='testing max h')
parser.add_argument('--max_w', type=int, default=640, help='testing max w')
parser.add_argument('--fix_res', action='store_true', help='scene all using same res')

parser.add_argument('--num_worker', type=int, default=1, help='depth_filter worker')
parser.add_argument('--save_freq', type=int, default=20, help='save freq of local pcd')

parser.add_argument('--filter_method', type=str, default='normal', choices=["gipuma", "normal"], help="filter method")
parser.add_argument('--save_ply', action='store_true', help='saves depthmaps as ply')


# depthmaps generation
parser.add_argument('--run_gendepth', action='store_true', help='Flag to run depthmaps generation')
parser.add_argument('--NviewGen', type=int, default=5, help='number of views used to generate depth maps (DTU=5)')
parser.add_argument('--depthgen_thres', type=float, default=0.8, help='Depth generation photometric confidence mask: pixels with photo confidence below threshold are dismissed')

#filter
parser.add_argument('--run_filter',action='store_true', help='Flag to run depthmaps filter and fusion')
parser.add_argument('--NviewFilter', type=int, default=10, help='number of src views used while filtering depth maps (DTU=10)')
parser.add_argument('--photomask', type=float, default=0.8, help='photometric confidence mask: pixels with photo confidence below threshold are dismissed')
parser.add_argument('--geomask', type=int, default=3, help='geometric view mask: pixels not seen by at least a certain number of views are dismissed ')
parser.add_argument('--condmask_pixel', type=float, default=1.0, help='conditional mask pixel: pixels which reproject back into the ref view at more than the threshold number of pixels are dismissed')
parser.add_argument('--condmask_depth', type=float, default=0.01, help='conditional mask on relative depth difference: pixels with depths prediction values above a threshold (1%) are dismissed')

parser.add_argument('--share_cr', action='store_true', help='whether share the cost volume regularization')
parser.add_argument("--fpn_base_channel", type=int, default=8)
parser.add_argument("--reg_channel", type=int, default=8)
parser.add_argument('--reg_mode', type=str, default="reg2d")
parser.add_argument('--dlossw', type=str, default="1,1,1,1", help='depth loss weight for different stage')
parser.add_argument('--resume', action='store_true', help='continue to train the model')
parser.add_argument('--group_cor', action='store_true',help='group correlation')
parser.add_argument('--group_cor_dim', type=str, default="8,8,4,4", help='group correlation dim')
parser.add_argument('--inverse_depth', action='store_true',help='inverse depth')
parser.add_argument('--agg_type', type=str, default="ConvBnReLU3D", help='cost regularization type')
parser.add_argument('--dcn', action='store_true',help='dcn')
parser.add_argument('--arch_mode', type=str, default="fpn")
parser.add_argument('--ot_continous', action='store_true',help='optimal transport continous gt bin')
parser.add_argument('--ot_eps', type=float, default=1)
parser.add_argument('--ot_iter', type=int, default=0)
parser.add_argument('--rt', action='store_true',help='robust training')
parser.add_argument('--use_raw_train', action='store_true',help='using highRes input pictures')
parser.add_argument('--mono', action='store_true',help='query to build mono depth prediction and loss')
parser.add_argument('--split', type=str, default='intermediate', help='intermediate or advanced')
parser.add_argument('--save_jpg', action='store_true')
parser.add_argument('--ASFF', action='store_true')
parser.add_argument('--vis_ETA', action='store_true')
parser.add_argument('--vis_stg_features', type=int, default=0, choices=[1,2,3,4], help='visual. Ref img features at selected stage from FPN')
parser.add_argument('--attn_temp', type=float, default=2)

# Debug options
parser.add_argument('--debug_model', type=int, default=0, help='powers of 2 for switches selection (debug = 2⁰+2¹+2³+2⁴+...) with '
                    '0: MVS4Net, plot input images & FPN4 features (add 1) '
                    '1: MVS4Net,plot depth (add 2) '
                    '2: MVS4Net,plot depth hpothesis (add 4) '
                    '3: MVS4Net,plot attention weights (add 8) '
                    '4: MVS4Net,plot mono depths (add 16) '
                    '5: StageNet, plot warped views(add 32) '
                    '6: StageNet, plot correl weights on depth hypos for each stage (add 64) '
                    '7: StageNet, plot Attention weights on depth hypos for each stage (add 128) '
                    '8: StageNet, (add 256) '
                    '63: ALL')

parser.add_argument('--debug_depth_gen', type=int, default=0, help='powers of 2 for switches selection (debug = 2⁰+2¹+2³+2⁴+...) with '
                    '0: plot input image (add 1) '
                    '1: plot depthmap and confidence for each view (add 2) '
                    '2: plot 3D point-cloud for each view (add 4)'
                    '3: plot combined 3D point-clouds  (add 8)'
                    '4:  (add 16)'
                    )
    
parser.add_argument('--debug_depth_filter', type=int, default=0, help='powers of 2 for switches selection (debug = 2⁰+2¹+2³+2⁴+...) with '
                    '0: plot depthmap and confidence for each view (add 1) '
                    '1: plot 3D point-cloud for each view (add 2) '
                    '2: plot combined 3D point-clouds (add 4)'
                    '3:   (add 8)'
                    )

# parse arguments and check
args = parser.parse_args()
print("argv:", sys.argv[1:])
print_args(args)

num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])
print ("num_stage: ", num_stage)
Interval_Scale = args.interval_scale


def NormalizeNumpy(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def get_powers(n):
    return [str(p) for p,v in enumerate(bin(n)[:1:-1]) if int(v)]

# read intrinsics and extrinsics
def read_camera_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=float, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=float, sep=' ').reshape((3, 3))
    return intrinsics, extrinsics


# read an image
def read_img(filename):
    img = Image.open(filename)
    # scale 0~255 to 0~1
    np_img = np.array(img, dtype=np.float32) / 255.
    return np_img


# read a binary mask
def read_mask(filename):
    return read_img(filename) > 0.5


# save a binary mask
def save_mask(filename, mask):
    assert mask.dtype == np.bool
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)


# read a pair file, [(ref_view1, [src_view1-1, ...]), (ref_view2, [src_view2-1, ...]), ...]
def read_pair_file(filename):
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        # 49 viewpoints
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            if len(src_views) > 0:
                data.append((ref_view, src_views))
    return data

def write_cam(file, cam):
    f = open(file, "w")
    f.write('extrinsic\n')
    for i in range(0, 4):
        for j in range(0, 4):
            f.write(str(cam[0][i][j]) + ' ')
        f.write('\n')
    f.write('\n')

    f.write('intrinsic\n')
    for i in range(0, 3):
        for j in range(0, 3):
            f.write(str(cam[1][i][j]) + ' ')
        f.write('\n')

    f.write('\n' + str(cam[1][3][0]) + ' ' + str(cam[1][3][1]) + ' ' + str(cam[1][3][2]) + ' ' + str(cam[1][3][3]) + '\n')

    f.close()

def depth2pts_np(depth_map, cam_intrinsic, cam_extrinsic):
    feature_grid = get_pixel_grids_np(depth_map.shape[0], depth_map.shape[1])

    uv = np.matmul(np.linalg.inv(cam_intrinsic), feature_grid)
    cam_points = uv * np.reshape(depth_map, (1, -1))

    R = cam_extrinsic[:3, :3]
    t = cam_extrinsic[:3, 3:4]
    R_inv = np.linalg.inv(R)

    world_points = np.matmul(R_inv, cam_points - t).transpose()
    return world_points

def get_pixel_grids_np(height, width):
    x_linspace = np.linspace(0.5, width - 0.5, width)
    y_linspace = np.linspace(0.5, height - 0.5, height)
    x_coordinates, y_coordinates = np.meshgrid(x_linspace, y_linspace)
    x_coordinates = np.reshape(x_coordinates, (1, -1))
    y_coordinates = np.reshape(y_coordinates, (1, -1))
    ones = np.ones_like(x_coordinates, dtype=np.float32)
    grid = np.concatenate([x_coordinates, y_coordinates, ones], axis=0)

    return grid


# Get Open3D box
def get_o3d_frame_bbox(dims=(0.57, 0.37, 0.22), delta = (0,0,0), scale = 1, context=None):
    """generate Open3D object for bin and frame
    Args:
        dims (tuple, optional): Bin box dimensions (in m). Defaults to (0.54, 0.34, 0.2).
        delta (tuple, optional): Offset to apply to bin box. Mostly for non-Blender (i.e real datasets) where world ref is determined empirically. Defaults to (0,0,0).
        scale (int, optional): scale factor. Defaults to 1.
    Returns:
        Open3D objects:frame, bounding_box, bounding_box2 (bbox with 2cm wall offset) - WARNING: in mm
    """
    
    # Test Configurations (overides other arguments if defined)
    if context is not None:
        if "overhead03" in context:
            # dims = (0.54, 0.34, 0.2)
            dims = (0.57, 0.37, 0.22)
            delta = (0.08, 0.03, .0)
        elif "overhead02" in context:
            # dims = (0.54, 0.34, 0.2)
            dims = (0.57, 0.37, 0.22)
            delta = (0.08, 0.03, .0)
        elif "Merlin_Mario_Set_with_GT" in context:
            dims = (0.57, 0.37, 0.22)
            delta = (0.125, 0.09, .0)
        else:
            dims = (0.57, 0.37, 0.22)
            delta = (0,0,0)
                
    # Plot axis and 3D points in WORLD ref
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100*scale, origin=[0, 0, 0])

    # Change dimensions from m to mm
    dims_bin = np.asarray(dims)*1000
    delta_orig = np.asarray(delta)*1000
    
    # Apply scaling factor if any    
    dims_bin *= scale
    delta_orig *= scale
    
    # Create Bounding box for bin internal walls
    min_bbox = -dims_bin / 2.0
    max_bbox = dims_bin / 2.0
    max_bbox[2] -= min_bbox[2]
    min_bbox[2] = 0 
    
    bbox = o3d.geometry.AxisAlignedBoundingBox()
    bbox.min_bound = min_bbox + delta_orig
    bbox.max_bound = max_bbox + delta_orig
    bbox.color = [0, 0, 1]
    
    # Create Bounding box for bin external walls
    wall_size = 20   # in mm
    
    bbox2 = o3d.geometry.AxisAlignedBoundingBox()
    bbox2.min_bound = min_bbox + delta_orig - np.array((wall_size, wall_size, wall_size))
    bbox2.max_bound = max_bbox + delta_orig + np.array((wall_size, wall_size, 0))
    bbox2.color = [1, 0, 0]
    
    # o3d.visualization.draw_geometries([frame] + [bbox] + [bbox2])
    
    return frame, bbox, bbox2 


def invert(rotation_translation):
    '''Invert a 3D rotation matrix in their (R | t) representation    '''
    rot = rotation_translation[0].T
    trans = -rot @ rotation_translation[1]
    return (rot, trans)


def get_o3d_cameras(cam_extrinsics, highlight_1st=False):
    
    cams=[]
    for i, extrinsics in enumerate(cam_extrinsics):
        rotation = extrinsics[:3,:3]
        translation = extrinsics[:3,-1]

        camera_rotation, camera_translation = invert((rotation, translation))

        height = 30
        cam = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=5,cone_radius=10,cylinder_height=height,cone_height=height/3)
        cam.compute_vertex_normals()
        if i==0 and highlight_1st:
        # if i in list(np.arange(11,19)):
            cam.paint_uniform_color((1,0,0))
        else:
            cam.paint_uniform_color((0,0,1))
        cam.translate([0, 0, -height])
        cam.rotate(camera_rotation, center=np.array([0, 0, 0]))
        cam.translate(camera_translation)
        
        cams.append(cam)
    
    return cams


######## SAVE DEPTHS ####################################################################################
#########################################################################################################

def save_depth(testlist):

    # Information
    print("============ DEPTH MAPS GENERATION using {} views".format(args.NviewGen))
        
    # CUDA preps
    torch.cuda.reset_peak_memory_stats()
    total_time = 0
    total_sample = 0
    for scene in testlist:
        time_this_scene, sample_this_scene = save_scene_depth([scene])
        total_time += time_this_scene
        total_sample += sample_this_scene
    gpu_measure = torch.cuda.max_memory_allocated() / 1024. / 1024. /1024.    
    print('total time: {}'.format(total_time))    
    print('avg time: {}'.format(total_time/total_sample))
    print('max gpu: {}'.format(gpu_measure))


def save_scene_depth(testlist):
    # dataset, dataloader
    MVSDataset = find_dataset_def(args.dataset)
    test_dataset = MVSDataset(datapath=args.datapath, 
                              resolution=args.data_resolution,
                              listfile=testlist, 
                              mode="test", 
                              nviews=args.NviewGen, 
                              interval_scale=Interval_Scale,
                              max_h=args.max_h, 
                              max_w=args.max_w, 
                              pair_fname=args.pair_fname, 
                              lighting=args.lighting,
                              dsname=args.dataset_name, 
                              )
    TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    # model
    model = MVS4net(arch_mode=args.arch_mode, 
                    reg_net=args.reg_mode, 
                    num_stage=4, 
                    fpn_base_channel=args.fpn_base_channel, 
                    reg_channel=args.reg_channel, 
                    stage_splits=[int(n) for n in args.ndepths.split(",")], 
                    depth_interals_ratio=[float(ir) for ir in args.depth_inter_r.split(",")],
                    group_cor=args.group_cor, 
                    group_cor_dim=[int(n) for n in args.group_cor_dim.split(",")],
                    inverse_depth=args.inverse_depth,
                    agg_type=args.agg_type,
                    dcn=args.dcn,
                    mono=args.mono,
                    asff=args.ASFF,
                    attn_temp=args.attn_temp,
                    vis_ETA=args.vis_ETA,
                    vis_stg_features=args.vis_stg_features,
                    debug=args.debug_model, 
                )
    
    # load checkpoint file specified by args.loadckpt
    print("=> loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict['model'], strict=True)
    model = nn.DataParallel(model)
    model.cuda()
    model.eval()
    
    # for the final point cloud
    vertices = []
    vertices_colors = []
    cam_extrinsics = []
        
    # Eval
    total_time = 0
    with torch.no_grad():
        for batch_idx, sample in enumerate(TestImgLoader):
            sample_cuda = tocuda(sample)
            start_time = time.time()
            
            # fwd pass through the mode for all metas
            outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"], sample["filename"])
            # outputs = dict_keys(['stage1', 'depth', 'photometric_confidence', 'hypo_depth', 'attn_weight', 'inverse_min_depth', 'inverse_max_depth', 'stage2', 'stage3', 'stage4'])
            
            end_time = time.time()
            total_time += end_time - start_time
            outputs = tensor2numpy(outputs)
            del sample_cuda 
            filenames = sample["filename"]
            cams = sample["proj_matrices"]["stage{}".format(num_stage)].numpy()
            imgs = tensor2numpy(sample["imgs"]) 
            print('=== Iter {}/{}, Ref view: {} Src views: {} FwdPassTime:{:.3f} '.format(batch_idx+1, 
                                                                                     len(TestImgLoader), 
                                                                                     test_dataset.metas[batch_idx][1], 
                                                                                     test_dataset.metas[batch_idx][2][:args.NviewGen-1],
                                                                                     end_time - start_time
                                                                                     ))
            sys.stdout.flush()
            
            # save depth maps and confidence maps (zip useful only if batch not = 1)
            for filename, cam, img, depth_est, photo_conf in zip(filenames, cams, imgs, outputs["stage4"]["depth"], outputs["stage4"]["photometric_confidence"]): 
                
                camID = filename.split("/")[2][-5:-2]                
                
                # Save images                
                img = img[0]  # ref view
                img_filename = os.path.join(args.outdir, filename.format('images', '.jpg'))
                os.makedirs(img_filename.rsplit('/', 1)[0], exist_ok=True)
                img = np.clip(np.transpose(img, (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(img_filename, img_bgr)
                res = img.shape
                scale = res[1]//600 
                res2 = str(tuple(int(x//scale) for x in res[:2]))
                 
                #  DEBUG plot input images
                if '0' in get_powers(args.debug_depth_gen): # add 1                  
                    cv2.imshow('[IMG] cam:{} Res:{}->{}'.format(camID, res[:2], res2), img_bgr[::scale, ::scale])    
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                # END DEBUG 

                # save depth maps
                depth_filename = os.path.join(args.outdir, filename.format('depth_est', '.pfm'))
                os.makedirs(depth_filename.rsplit('/', 1)[0], exist_ok=True)
                save_pfm(depth_filename, depth_est)
                cv2.imwrite(depth_filename.replace(".pfm",".png"),  np.uint8(NormalizeNumpy(depth_est)*255))

                #  Optional: save depthmaps with pretty colors 
                if args.save_jpg:     
                    print("Depth pictures saved to files for stage ", end="")
                    for stage_idx in range(4):
                        depth_jpg_filename = os.path.join(args.outdir, filename.format('depth_est', '{}_{}.jpg'.format('stage',str(stage_idx+1))))
                        stage_depth = outputs['stage{}'.format(stage_idx+1)]['depth'][0] # INFO outputs["stage4"]['depth'].shape = (1, 960, 1280)
                        mi = np.min(stage_depth[stage_depth>0])
                        ma = np.max(stage_depth)
                        depth = (stage_depth-mi)/(ma-mi+1e-8)
                        depth = (255*depth).astype(np.uint8)
                        depth_img = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
                        cv2.imwrite(depth_jpg_filename, depth_img) 
                        print("{}, ".format(stage_idx+1), end="")
                    print()
                    
                    if args.mono:
                        print("Mono-depth pictures saved to files for stage ", end="")
                        for stage_idx in range(4):
                            if stage_idx == 0:
                                continue
                            mono_depth_jpg_filename = os.path.join(args.outdir, filename.format('depth_est', '{}_{}.jpg'.format('mono',str(stage_idx+1))))
                            stage_mono_depth = outputs['stage{}'.format(stage_idx+1)]['mono_feat'][0][4] # INFO outputs["stage4"]['mono_feat'].shape = (1, 8, 960, 1280)
                            mi = np.min(stage_mono_depth[stage_mono_depth>0])
                            ma = np.max(stage_mono_depth)
                            depth = (stage_mono_depth-mi)/(ma-mi+1e-8)
                            depth = (255*depth).astype(np.uint8)
                            depth_img = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
                            cv2.imwrite(mono_depth_jpg_filename, depth_img)
                            print("{}, ".format(stage_idx+1), end="")
                    print()
                    
                #save confidence maps
                photo_conf_mask = (photo_conf > args.depthgen_thres) * 1.0  
                confidence_filename = os.path.join(args.outdir, filename.format('confidence', '.pfm'))
                os.makedirs(confidence_filename.rsplit('/', 1)[0], exist_ok=True)
                save_pfm(confidence_filename, photo_conf) 
                cv2.imwrite(confidence_filename.replace(".pfm", ".png"), np.uint8(photo_conf * 255))
                
                #  DEBUG plot depth & photo conf
                if '1' in get_powers(args.debug_depth_gen): # add 2   
                    cv2.imshow("[DEPTH] cam:{} Res:{}->{}".format(camID, depth_est.shape, res2), NormalizeNumpy(depth_est)[::scale, ::scale]) 
                    cv2.imshow("[CONF] cam:{} Res:{}->{}".format(camID, photo_conf.shape, res2), photo_conf[::scale, ::scale]) 
                    cv2.imshow("[CONF {:.0f}%] cam:{} Res:{}->{}".format(args.depthgen_thres*100, camID, photo_conf.shape, res2), photo_conf_mask[::scale, ::scale] ) 
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()    
                #  DEBUG: plot 3D point-cloud
                
                #save cams                
                cam = cam[0]  #ref cam
                ref_extrinsics = cam[0]
                ref_intrinsics = cam[1][:3,:3]
                cam_filename = os.path.join(args.outdir, filename.format('cams', '_cam.txt'))
                os.makedirs(cam_filename.rsplit('/', 1)[0], exist_ok=True)
                write_cam(cam_filename, cam)
                cam_extrinsics.append(ref_extrinsics)

                # generate local points cloud
                if args.save_ply: 
                    ply_filename = os.path.join(args.outdir, filename.format('ply_local', '.ply'))
                    os.makedirs(ply_filename.rsplit('/', 1)[0], exist_ok=True)
                    generate_pointcloud(img, depth_est, ply_filename, cam[1, :3, :3])
                
                # Create 3D points cloud     
                xyz_world = depth2pts_np(depth_est, ref_intrinsics, ref_extrinsics) # all points
                xyz_world_masked = xyz_world[photo_conf_mask.flatten()==1]
                # xyz_color_masked = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[photo_conf_mask==1]/255       
                xyz_color_masked = img[photo_conf_mask==1]/255       
                
                # Add to global point cloud
                vertices.append(xyz_world_masked)
                vertices_colors.append((xyz_color_masked * 255).astype(np.uint8)) # only keep points with certain pho_conf
        
                # DEBUG - Plot 3D point cloud for each view
                if '2' in get_powers(args.debug_depth_gen): # add 4
                                
                    # Create frame and bounding boxes
                    frame, bbox, bbox2 = get_o3d_frame_bbox(context = args.datapath)
                    
                    # get_camera objects
                    o3D_cameras = get_o3d_cameras([ref_extrinsics], True)
                    
                    # Create  point cloud
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(xyz_world_masked)
                    pcd.colors = o3d.utility.Vector3dVector(xyz_color_masked)
                    pcd.estimate_normals()
                    
                    # plot (type h for help in open3d)
                    if args.dataset_name in ["dtu"]:
                        o3d.visualization.draw_geometries([frame]+[pcd]+o3D_cameras,
                                                            front   = [ 0.85, 0.12, -0.50],
                                                            lookat  = [ 27.36, 26.01, 578.24],
                                                            up      = [ -0.37, -0.54, -0.76 ],
                                                            zoom    = 0.34)
                        # Display intensity of probability
                        # a = NormalizeNumpy(refine_conf[refine_conf_mask==1])
                        # b = np.zeros((len(c),))
                        # c = np.ones((len(c),))
                        # pcd.colors = o3d.utility.Vector3dVector(np.stack((a,c, c), axis=1))
                        # o3d.visualization.draw_geometries([pcd])                
                    else:
                        o3d.visualization.draw_geometries([frame]+[bbox]+[bbox2]+[pcd]+o3D_cameras)

    #  Once all images processed, concatenate all vertices and save as ply format
    vertices_allviews = np.concatenate(vertices, axis=0)
    vertices_colors_allviews = np.concatenate(vertices_colors, axis=0)
    
    
    # DEBUG - Plot final combined 3D point cloud
    if '3' in get_powers(args.debug_depth_gen): # add 8
        
        # Create frame and bounding boxes
        frame, bbox, bbox2 = get_o3d_frame_bbox(context=args.datapath)
            
        # get_camera objects
        o3D_cameras = get_o3d_cameras(cam_extrinsics, False)
        
        # Create  point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices_allviews)
        pcd.colors = o3d.utility.Vector3dVector(vertices_colors_allviews/255)
        pcd.estimate_normals()
        
        # plot (type h for help in open3d)
        if args.dataset_name in ["dtu"]:
            o3d.visualization.draw_geometries([frame]+[pcd]+o3D_cameras,
                                                front   = [ 0.85, 0.12, -0.50],
                                                lookat  = [ 27.36, 26.01, 578.24],
                                                up      = [ -0.37, -0.54, -0.76 ],
                                                zoom    = 0.34)
        else:
            o3d.visualization.draw_geometries([frame]+[bbox]+[bbox2]+[pcd]+o3D_cameras)
                 
            pcd = pcd.crop(bbox2) 
            pcd = pcd.voxel_down_sample(voxel_size=3)
            o3d.visualization.draw_geometries([frame]+[bbox]+[bbox2]+[pcd]+o3D_cameras)
    # DEBUG - END

    torch.cuda.empty_cache()
    gc.collect()
    return total_time, len(TestImgLoader)




###### FILTER ##############################################################################################
############################################################################################################


# project the reference point cloud into the source view, then project back
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    # reference 3D space
    xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref),
                        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))
    # source 3D space
    xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
                        np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    # source view x, y
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)
    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = np.matmul(np.linalg.inv(intrinsics_src),
                        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    # reference 3D space
    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src

############################################################################################################

def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, 
                                depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(
        depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src)
    
    # check |p_reproj-p_1| < 1
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    mask = np.logical_and(dist < args.condmask_pixel, relative_depth_diff < args.condmask_depth)
    depth_reprojected[~mask] = 0

    return mask, depth_reprojected, x2d_src, y2d_src

############################################################################################################

def filter_depth(scene_folder):
    
    # Information
    print("============ DEPTH MAPS FILTER / FUSION using {} views".format(args.NviewFilter))
    assert args.NviewFilter <= args.NviewGen, "Number of view for filter & fusion must be equal or lower than number of views for depthmap creation"

    # for the final point cloud
    vertices = []
    vertices_colors = []
    cam_extrinsics = []

    # Read pair file 
    if args.dataset_name == "bin":
        pair_file = os.path.join(args.datapath, "../..", args.pair_fname)  
    else:
        pair_file = os.path.join(args.datapath, args.pair_fname)     
    pair_data = read_pair_file(pair_file)    

    # for each reference view and the corresponding source views
    for ref_view, src_views in pair_data:
        
        start_time = time.time()
        
        src_views = src_views[:args.NviewFilter-1]        
        print (f'[FILTER] ref_view: {ref_view} src_views: {src_views}')
        
        # load the camera parameters
        ref_intrinsics, ref_extrinsics = read_camera_parameters(os.path.join(scene_folder, 'cams/{:0>8}_cam.txt'.format(ref_view)))
        cam_extrinsics.append(ref_extrinsics)
        if DEBUG:
            print ("ref_intrinsics:\n", ref_intrinsics)
            print ("ref_extrinsics:\n", ref_extrinsics)
        
        # load the reference image
        ref_img = read_img(os.path.join(scene_folder, 'images/{:0>8}.jpg'.format(ref_view)))
        
        # load the estimated depth of the reference view
        ref_depth_est = read_pfm(os.path.join(scene_folder, 'depth_est/{:0>8}.pfm'.format(ref_view)))[0]
        
        # load the photometric mask of the reference view
        confidence = read_pfm(os.path.join(scene_folder, 'confidence/{:0>8}.pfm'.format(ref_view)))[0]
        photo_mask = confidence > args.photomask
            
        all_srcview_depth_ests = []
        all_srcview_x = []
        all_srcview_y = []
        all_srcview_geomask = []

        # compute the geometric mask
        geo_mask_sum = 0
        for src_view in src_views:
            
            # camera parameters of the source view
            src_intrinsics, src_extrinsics = read_camera_parameters(os.path.join(scene_folder, 'cams/{:0>8}_cam.txt'.format(src_view)))
            if DEBUG:
                print ("src_intrinsics:\n", src_intrinsics)
                print ("src_extrinsics:\n", src_extrinsics)
            
            # the estimated depth of the source view
            src_depth_est = read_pfm(os.path.join(scene_folder, 'depth_est/{:0>8}.pfm'.format(src_view)))[0]

            geo_mask, depth_reprojected, x2d_src, y2d_src = check_geometric_consistency(ref_depth_est, ref_intrinsics, ref_extrinsics,
                                                                                        src_depth_est, src_intrinsics, src_extrinsics)
            geo_mask_sum += geo_mask.astype(np.int32)
            all_srcview_depth_ests.append(depth_reprojected)
            all_srcview_x.append(x2d_src)
            all_srcview_y.append(y2d_src)
            all_srcview_geomask.append(geo_mask)

        depth_est_averaged = (sum(all_srcview_depth_ests) + ref_depth_est) / (geo_mask_sum + 1)
        # at least 3 source views matched
        geo_mask = geo_mask_sum >= args.geomask
        
        # compute final mask
        final_mask = np.logical_and(photo_mask, geo_mask)

        print("processing {}, ref-view{:0>2}, photo/geo/final-mask:{:.2f}/{:.2f}/{:.2f}, FusionTime={:.3f}sec".format(scene_folder, ref_view,
                                                                                                photo_mask.mean()*100,
                                                                                                geo_mask.mean()*100, 
                                                                                                final_mask.mean()*100,
                                                                                                time.time()-start_time))
        
        os.makedirs(os.path.join(scene_folder, "mask"), exist_ok=True)
        save_mask(os.path.join(scene_folder, "mask/{:0>8}_photo.png".format(ref_view)), photo_mask)
        save_mask(os.path.join(scene_folder, "mask/{:0>8}_geo.png".format(ref_view)), geo_mask)
        save_mask(os.path.join(scene_folder, "mask/{:0>8}_final.png".format(ref_view)), final_mask)
        
        #  DEBUG: plot depth with masks
        if '0' in get_powers(args.debug_depth_filter): # add 1
             
            resol = ref_depth_est.shape
            scale = [1  if resol[1]//600 == 0 else resol[1]//600][0]
            
            # img_norm = cv2.cvtColor(ref_img[::scale,::scale], cv2.COLOR_BGR2RGB)
            cv2.imshow('ref_img', ref_img[::scale,::scale])
            cv2.imshow('ref_depth {}->{}'.format(resol[:2], str(tuple(int(x/scale) for x in resol[:2]))), NormalizeNumpy(ref_depth_est)[::scale,::scale])
            cv2.imshow('photo_mask (conf>{:.1f}%)'.format(args.photomask*100), photo_mask.astype(float)[::scale,::scale] )
            cv2.imshow('geo_mask', geo_mask.astype(float)[::scale,::scale] )
            cv2.imshow('final mask', final_mask.astype(float)[::scale,::scale] )
            cv2.imshow('fused depth', NormalizeNumpy(depth_est_averaged)[::scale,::scale] )
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        # DEBUG-END
        
        # BUILD 3D points (vertices) to be kept (appended) for a given ref_img
        height, width = depth_est_averaged.shape[:2]
        x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
        print("3D Pts-cloud: Number of valid points: {}/{} (average={:03f})\n".format(final_mask.sum(), height*width, final_mask.mean())) 
        
        xyz_world = depth2pts_np(depth_est_averaged, ref_intrinsics, ref_extrinsics) 
        xyz_world_masked = xyz_world[final_mask.flatten()]
        # xyz_color_masked = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)[final_mask]/255
        xyz_color_masked = ref_img[final_mask]
        
        # Add to global point cloud
        vertices.append(xyz_world_masked)
        vertices_colors.append((xyz_color_masked * 255).astype(np.uint8))
        
        #  DEBUG: plot 3D point-cloud for each view
        if '1' in get_powers(args.debug_depth_filter): # add 2
                            
            # Create frame and bounding boxes
            # frame, bbox, bbox2 = get_o3d_frame_bbox(dims=dims, delta=delta, scale=1)
            frame, bbox, bbox2 = get_o3d_frame_bbox(context = args.datapath)
            
            # get_camera objects
            o3D_cameras = get_o3d_cameras(cam_extrinsics, True)
            
            # Create  point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz_world_masked)
            pcd.colors = o3d.utility.Vector3dVector(xyz_color_masked)  # should get values 0-255
            pcd.estimate_normals()
            
            # plot (type h for help in open3d)
            if args.dataset_name in ["dtu"]:
                o3d.visualization.draw_geometries([frame]+[pcd]+o3D_cameras,
                                                    front   = [ 0.85, 0.12, -0.50],
                                                    lookat  = [ 27.36, 26.01, 578.24],
                                                    up      = [ -0.37, -0.54, -0.76 ],
                                                    zoom    = 0.34)
                
                # Display intensity of probability
                # a = NormalizeNumpy(refine_conf[refine_conf_mask==1])
                # b = np.zeros((len(c),))
                # c = np.ones((len(c),))
                # pcd.colors = o3d.utility.Vector3dVector(np.stack((a,c, c), axis=1))
                # o3d.visualization.draw_geometries([pcd])
            else:
                o3d.visualization.draw_geometries([frame]+[bbox]+[bbox2]+[pcd]+o3D_cameras)
        #  END DEBUG

    # Concatenate all views 
    vertices_all = np.concatenate(vertices, axis=0)
    vertices_colors_all = np.concatenate(vertices_colors, axis=0)

    if args.save_ply:
        ply_vertices = np.array([tuple(v) for v in vertices_all], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        ply_vertices_colors = np.array([tuple(v) for v in vertices_colors_all], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

        vertex_all = np.empty(len(ply_vertices), ply_vertices.dtype.descr + ply_vertices_colors.dtype.descr)
        for prop in ply_vertices.dtype.names:
            vertex_all[prop] = ply_vertices[prop]
        for prop in vertices_colors.dtype.names:
            vertex_all[prop] = ply_vertices_colors[prop]

        el = PlyElement.describe(vertex_all, 'vertex')
        plyfilename = os.path.join(scene_folder, "_fused_3Dpts.ply")
        PlyData([el]).write(plyfilename)
        print("saving the final model to", plyfilename)


    # DEBUG - Plot final fused 3D point cloud
    if '2' in get_powers(args.debug_depth_filter): # add 4
            
        # Create frame and bounding boxes
        # frame, bbox, bbox2 = get_o3d_frame_bbox(dims=dims, delta=delta, scale=1)
        frame, bbox, bbox2 = get_o3d_frame_bbox(context = args.datapath)
                        
        # get_camera objects
        o3D_cameras = get_o3d_cameras(cam_extrinsics, False)
            
        # Create  point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices_all)
        pcd.colors = o3d.utility.Vector3dVector(vertices_colors_all/255)
        pcd.estimate_normals()
        
        # plot (type h for help in open3d)
        if args.dataset_name in ["dtu"]:
            o3d.visualization.draw_geometries([frame]+[pcd]+o3D_cameras,
                                                front   = [ 0.85, 0.12, -0.50],
                                                lookat  = [ 27.36, 26.01, 578.24],
                                                up      = [ -0.37, -0.54, -0.76 ],
                                                zoom    = 0.34)
        else:
            o3d.visualization.draw_geometries([frame]+[bbox]+[bbox2]+[pcd]+o3D_cameras)
                    
            pcd = pcd.crop(bbox2) 
            o3d.visualization.draw_geometries([frame]+[bbox]+[bbox2]+[pcd])
            
            dwn_smpl = 5
            pcd = pcd.voxel_down_sample(voxel_size=dwn_smpl)
            o3d.visualization.draw_geometries([frame]+[bbox]+[bbox2]+[pcd])
            o3d.io.write_point_cloud(os.path.join(scene_folder, f"BDS9_s188_49cams_dwnsmpld_{dwn_smpl}mm.ply"), pcd.scale(0.01, (0,0,0)), write_ascii=False, compressed=False, print_progress=False)
        # DEBUG - END

############################################################################################################

def init_worker():
    '''
    Catch Ctrl+C signal to termiante workers
    '''
    signal.signal(signal.SIGINT, signal.SIG_IGN)

############################################################################################################

def pcd_filter_worker(scan):

    scene_folder = os.path.join(args.outdir, scan)
    filter_depth(scene_folder)

############################################################################################################

def pcd_filter(testlist, number_worker):

    partial_func = partial(pcd_filter_worker)

    p = Pool(number_worker, init_worker)
    try:
        p.map(partial_func, testlist)
    except KeyboardInterrupt:
        print("....\nCaught KeyboardInterrupt, terminating workers")
        p.terminate()
    else:
        p.close()
    p.join()

############################################################################################################

def mrun_rst(eval_dir, plyPath):
    print('Runing BaseEvalMain_func.m...')
    os.chdir(eval_dir)
    os.system('/mnt/cfs/algorithm/xiaofeng.wang/jeff/code/MVS/misc/matlab/bin/matlab -nodesktop -nosplash -r "BaseEvalMain_func(\'{}\'); quit" '.format(plyPath))
    print('Runing ComputeStat_func.m...')
    os.system('/mnt/cfs/algorithm/xiaofeng.wang/jeff/code/MVS/misc/matlab/bin/matlab -nodesktop -nosplash -r "ComputeStat_func(\'{}\'); quit" '.format(plyPath))
    print('Check your results! ^-^')


############################################################################################################
##############################################################################################################

if __name__ == '__main__':

    # Some checks
    if args.vis_ETA:
        # os.makedirs(os.path.join(args.outdir,'debug_figs/vis_ETA'), exist_ok=True)
        os.makedirs('./debug/figs/vis_ETA', exist_ok=True)
    assert args.batch_size == 1, "Please ensure batch size is 1 for output display and postprocessing purpose."

    # Get eval folders
    with open(args.testlist) as f:
        content = f.readlines()
        testlist = [line.rstrip() for line in content]
        
    # step1. save all the depth maps and the masks in outputs directory
    if args.run_gendepth:
        save_depth(testlist)

    # step2. filter saved depth maps with photometric confidence maps and geometric constraints 
    if args.run_filter:
        # pcd_filter(testlist, args.num_worker)
        for scene in testlist:
            scene_folder = os.path.join(args.outdir, scene)
            filter_depth(scene_folder)


 