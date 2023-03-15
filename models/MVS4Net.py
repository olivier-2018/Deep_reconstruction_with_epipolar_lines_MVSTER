import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.mvs4net_utils import stagenet, reg2d, reg3d, FPN4, FPN4_convnext, FPN4_convnext4, PosEncSine, PosEncLearned, \
        init_range, schedule_range, init_inverse_range, schedule_inverse_range, sinkhorn, mono_depth_decoder, ASFF


class MVS4net(nn.Module):
    def __init__(self, arch_mode="fpn", reg_net='reg2d', num_stage=4, fpn_base_channel=8, 
                reg_channel=8, stage_splits=[8,8,4,4], depth_interals_ratio=[0.5,0.5,0.5,1],
                group_cor=False, group_cor_dim=[8,8,8,8],
                inverse_depth=False,
                agg_type='ConvBnReLU3D',
                dcn=False,
                pos_enc=0,
                mono=False,
                mono_stg_itrpl="nearest",
                asff=False,
                attn_temp=2,
                attn_fuse_d=True,
                vis_ETA=False,
                vis_stg_features=False, 
                debug=0
                ):
        # pos_enc: 0 no pos enc; 1 depth sine; 2 learnable pos enc
        super(MVS4net, self).__init__()
        self.arch_mode = arch_mode
        self.num_stage = num_stage
        self.depth_interals_ratio = depth_interals_ratio
        self.group_cor = group_cor
        self.group_cor_dim = group_cor_dim
        self.inverse_depth = inverse_depth
        self.asff = asff
        if self.asff:
            self.asff = nn.ModuleList([ASFF(i) for i in range(num_stage)])
        self.attn_ob = nn.ModuleList()
        if arch_mode == "fpn":
            self.feature = FPN4(base_channels=fpn_base_channel, gn=False, dcn=dcn)
        self.vis_stg_features = vis_stg_features
        self.stagenet = stagenet(inverse_depth, mono, attn_fuse_d, vis_ETA, attn_temp)
        self.stage_splits = stage_splits
        self.reg = nn.ModuleList()
        self.pos_enc = pos_enc
        self.pos_enc_func = nn.ModuleList()
        self.mono = mono
        self.debug = debug
        if self.mono:
            self.mono_depth_decoder = mono_depth_decoder(mono_stg_itrpl)
        if reg_net == 'reg3d':
            self.down_size = [3,3,2,2]
        for idx in range(num_stage):
            if self.group_cor:
                in_dim = group_cor_dim[idx]
            else:
                in_dim = self.feature.out_channels[idx]
            if reg_net == 'reg2d':
                self.reg.append(reg2d(input_channel=in_dim, base_channel=reg_channel, conv_name=agg_type))
            elif reg_net == 'reg3d':
                self.reg.append(reg3d(in_channels=in_dim, base_channels=reg_channel, down_size=self.down_size[idx]))


    def forward(self, imgs, proj_matrices, depth_values, filename=None):
        
        # 
        depth_min = depth_values[:, 0].cpu().numpy()
        depth_max = depth_values[:, -1].cpu().numpy()
        depth_interval = (depth_max - depth_min) / depth_values.size(1)

        # step 1. feature extraction
        features = [] # container for ref+src imgs features
        for nview_idx in range(len(imgs)):  # imgs is a list of N=5 src img with shape [B, Color=3, H, W]
            img = imgs[nview_idx]           # img is a dict with 4 keys (stg1-4) which contains a tensor [B, Filter, Hstg, Wstg]
            features.append(self.feature(img)) # self.feature was extracted by FPN4. self.feature is a dict with keys: "stage1", "stage2", etc. 
            # self.feature(img)["stage1"].shape = torch.Size([1, 64, 64, 80])=> stage1, 2, 3 & 4 have 64, 32, 16 & 8 filters respectively (base channel is 8 by default)
        if self.vis_stg_features != 0:
            # scan_name = filename[0].split('/')[0]
            # image_name = filename[0].split('/')[2][:-2]
            # save_fn = './debug_figs/vis_stg_features/feat_{}'.format(scan_name+'_'+image_name)
            feat_ = features[-1]['stage4'].detach().cpu().numpy() # ref view (first)
            # feat_ = features[-1]['stage4'].detach().cpu().numpy() # last view
            # np.save(save_fn, feat_)
            
            import cv2
            # plot images features
            view = 0 # 0:ref image, -1 last src img
            
            stg = "stage"+str(self.vis_stg_features)
            skiplist = [8,4,2,1]
            skip = skiplist[self.vis_stg_features-1]
            for filter in range(0,features[view][stg].shape[1], skip): 
                img2plot = features[view][stg].permute(2,3,1,0)[:,:,filter,0].cpu().detach().numpy()
                cv2.imshow(f"feature {stg} ({skip})", img2plot)
                cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # step 2. iter (multi-scale)
        outputs = {}
        for stage_idx in range(self.num_stage):
            if not self.asff:
                features_stage = [feat["stage{}".format(stage_idx+1)] for feat in features] # [B, Filterstg, Hstg, Wstg]
            else:
                features_stage = [self.asff[stage_idx](feat['stage1'],feat['stage2'],feat['stage3'],feat['stage4']) for feat in features]

            proj_matrices_stage = proj_matrices["stage{}".format(stage_idx + 1)]
            B,C,H,W = features[0]['stage{}'.format(stage_idx+1)].shape # for ref img

            # init range
            if stage_idx == 0:
                if self.inverse_depth:
                    depth_hypo = init_inverse_range(depth_values, self.stage_splits[stage_idx], img[0].device, img[0].dtype, H, W)
                else:
                    depth_hypo = init_range(depth_values, self.stage_splits[stage_idx], img[0].device, img[0].dtype, H, W) 
            else:
                if self.inverse_depth:
                    depth_hypo = schedule_inverse_range(outputs_stage['inverse_min_depth'].detach(), outputs_stage['inverse_max_depth'].detach(), self.stage_splits[stage_idx], H, W)  # B D H W
                else:
                    depth_hypo = schedule_range(outputs_stage['depth'].detach(), self.stage_splits[stage_idx], self.depth_interals_ratio[stage_idx] * depth_interval, H, W)
            
            
            # print("STAGE {} INIT depth_hypo: Min {}, Max {}".format(stage_idx,
            #                                                         torch.amin(torch.amin(torch.amin(depth_hypo,1,True),2,True),3,True).flatten().cpu().numpy(), 
            #                                                         torch.amax(torch.amax(torch.amax(depth_hypo,1,True),2,True),3,True).flatten().cpu().numpy() ) )

            outputs_stage = self.stagenet(features_stage, proj_matrices_stage, depth_hypo=depth_hypo, regnet=self.reg[stage_idx], stage_idx=stage_idx,
                                        group_cor=self.group_cor, group_cor_dim=self.group_cor_dim[stage_idx],
                                        split_itv=self.depth_interals_ratio[stage_idx],
                                        fn=filename)
                # outputs_stage (stg 0)
                # 'depth': shape [6, 64, 80]
                # 'photometric_confidence': tensor(0., device='cuda:0')
                # 'hypo_depth': torch.Size([6, 8, 64, 80])
                # 'attn_weight': torch.Size([6, 8, 64, 80])
                # 'inverse_min_depth': torch.Size([6, 64, 80])
                # 'inverse_max_depth':torch.Size([6, 64, 80])
                # 'mono_feat': torch.Size([6, 64, 64, 80])
                # len(): 7

            # outputs_stage_d = outputs_stage['depth']
            # print("ESTIM depth: ",outputs_stage_d.shape) # B,stgH,stgW            
            # print ("MIN ESTIM depth: ",torch.amin(torch.amin(outputs_stage_d,2,True),1,True).flatten().cpu().numpy() )
            # print ("MAX ESTIM depth: ",torch.amax(torch.amax(outputs_stage_d,2,True),1,True).flatten().cpu().numpy() )
            

            outputs["stage{}".format(stage_idx + 1)] = outputs_stage
            
            outputs.update(outputs_stage) # CHECK: required during training?
        
        if self.mono and self.training:
        # if self.mono:
            outputs = self.mono_depth_decoder(outputs, depth_values[:,0], depth_values[:,1], self.debug)  # INFO; depth_values has only 2 depth values (min & max)   (batch,2) with len(D)=2 
            
            if self.debug > 1: # TODO
                import cv2  
                print (f"[DEBUG]mvs4Net.py, mono_depth, stage4")
                img2plot = outputs["stage4"]["mono_depth"][0].squeeze().cpu().detach().numpy()
                cv2.imshow(f"[DEBUG]mono feature stage4", img2plot/np.max(img2plot))
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        return outputs

def MVS4net_loss(inputs, depth_gt_ms, mask_ms, **kwargs):
    stage_lw = kwargs.get("stage_lw", [1,1,1,1])
    l1ot_lw = kwargs.get("l1ot_lw", [0,1])
    inverse = kwargs.get("inverse_depth", False)
    ot_iter = kwargs.get("ot_iter", 3)
    ot_eps = kwargs.get("ot_eps", 1)
    ot_continous = kwargs.get("ot_continous", False)
    mono = kwargs.get("mono", False)
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    stage_ot_loss = []
    stage_l1_loss = []
    range_err_ratio = []
    for stage_idx, (stage_inputs, stage_key) in enumerate([(inputs[k], k) for k in inputs.keys() if "stage" in k]):
        depth_pred = stage_inputs['depth']
        hypo_depth = stage_inputs['hypo_depth']
        attn_weight = stage_inputs['attn_weight']
        B,H,W = depth_pred.shape
        D = hypo_depth.shape[1]
        mask = mask_ms[stage_key]
        mask = mask > 0.5
        depth_gt = depth_gt_ms[stage_key]

        if mono and stage_idx!=0:
            this_stage_l1_loss = F.l1_loss(stage_inputs['mono_depth'][mask], depth_gt[mask], reduction='mean')
            # this_stage_l1_loss = F.smooth_l1_loss(stage_inputs['mono_depth'][mask], depth_gt[mask], reduction='mean')
            # this_stage_l1_loss = F.mse_loss(stage_inputs['mono_depth'][mask], depth_gt[mask], reduction='mean')
            # this_stage_l1_loss = F.HuberLoss(stage_inputs['mono_depth'][mask], depth_gt[mask], reduction='mean', delta=1.0)
        else:
            this_stage_l1_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)

        # mask range
        if inverse:
            depth_itv = (1/hypo_depth[:,2,:,:]-1/hypo_depth[:,1,:,:]).abs()  # B H W
            mask_out_of_range = ((1/hypo_depth - 1/depth_gt.unsqueeze(1)).abs() <= depth_itv.unsqueeze(1)).sum(1) == 0 # B H W
        else:
            depth_itv = (hypo_depth[:,2,:,:]-hypo_depth[:,1,:,:]).abs()  # B H W
            mask_out_of_range = ((hypo_depth - depth_gt.unsqueeze(1)).abs() <= depth_itv.unsqueeze(1)).sum(1) == 0 # B H W
        range_err_ratio.append(mask_out_of_range[mask].float().mean())

        this_stage_ot_loss = sinkhorn(depth_gt, hypo_depth, attn_weight, mask, iters=ot_iter, eps=ot_eps, continuous=ot_continous)[1]

        stage_l1_loss.append(this_stage_l1_loss)
        stage_ot_loss.append(this_stage_ot_loss)
        total_loss = total_loss + stage_lw[stage_idx] * (l1ot_lw[0] * this_stage_l1_loss + l1ot_lw[1] * this_stage_ot_loss)

    return total_loss, stage_l1_loss, stage_ot_loss, range_err_ratio


def Blend_loss(inputs, depth_gt_ms, mask_ms, **kwargs):
    stage_lw = kwargs.get("stage_lw", [1,1,1,1])
    l1ot_lw = kwargs.get("l1ot_lw", [0,1])
    inverse = kwargs.get("inverse_depth", False)
    ot_iter = kwargs.get("ot_iter", 3)
    ot_eps = kwargs.get("ot_eps", 1)
    ot_continous = kwargs.get("ot_continous", False)
    depth_max = kwargs.get("depth_max", 100)
    depth_min = kwargs.get("depth_min", 1)
    mono = kwargs.get("mono", False)
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    stage_ot_loss = []
    stage_l1_loss = []
    range_err_ratio = []
    for stage_idx, (stage_inputs, stage_key) in enumerate([(inputs[k], k) for k in inputs.keys() if "stage" in k]):
        depth_pred = stage_inputs['depth']
        hypo_depth = stage_inputs['hypo_depth']
        attn_weight = stage_inputs['attn_weight']
        B,H,W = depth_pred.shape
        mask = mask_ms[stage_key]
        mask = mask > 0.5
        depth_gt = depth_gt_ms[stage_key]
        depth_pred_norm = depth_pred * 128 / (depth_max - depth_min)[:,None,None]  # B H W
        depth_gt_norm = depth_gt * 128 / (depth_max - depth_min)[:,None,None]  # B H W

        if mono and stage_idx!=0:
            this_stage_l1_loss = F.l1_loss(stage_inputs['mono_depth'][mask], depth_gt[mask], reduction='mean')
        else:
            this_stage_l1_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)

        if inverse:
            depth_itv = (1/hypo_depth[:,2,:,:]-1/hypo_depth[:,1,:,:]).abs()  # B H W
            mask_out_of_range = ((1/hypo_depth - 1/depth_gt.unsqueeze(1)).abs() <= depth_itv.unsqueeze(1)).sum(1) == 0 # B H W
        else:
            depth_itv = (hypo_depth[:,2,:,:]-hypo_depth[:,1,:,:]).abs()  # B H W
            mask_out_of_range = ((hypo_depth - depth_gt.unsqueeze(1)).abs() <= depth_itv.unsqueeze(1)).sum(1) == 0 # B H W
        range_err_ratio.append(mask_out_of_range[mask].float().mean())

        this_stage_ot_loss = sinkhorn(depth_gt, hypo_depth, attn_weight, mask, iters=ot_iter, eps=ot_eps, continuous=ot_continous)[1]

        stage_l1_loss.append(this_stage_l1_loss)
        stage_ot_loss.append(this_stage_ot_loss)
        total_loss = total_loss + stage_lw[stage_idx] * (l1ot_lw[0] * this_stage_l1_loss + l1ot_lw[1] * this_stage_ot_loss)

    abs_err = torch.abs(depth_pred_norm[mask] - depth_gt_norm[mask])
    epe = abs_err.mean()
    err3 = (abs_err<=3).float().mean()*100
    err1= (abs_err<=1).float().mean()*100
    return total_loss, stage_l1_loss, stage_ot_loss, range_err_ratio, epe, err3, err1