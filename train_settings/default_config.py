"""
 Copyright 2022 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

from easydict import EasyDict as edict
import os
from source.utils.config_utils import override_options

def get_base_config():
    # default
    cfg = edict()
    cfg.copy_data = False  # if the data is initially tarred and must be copied to computing node
    cfg.distributed = False  # does not support distributed training
    cfg.model =  None   
    
    cfg.grad_acc_steps = 1  
    cfg.barf_c2f =  None  # coarse-to-fine positional encoding used in barf
    cfg.apply_cf_pe = True   # need to have barf_c2f and to apply_cf_pe
    cfg.seed = 0             
    cfg.do_eval = True

    cfg.increase_depth_range_by_x_percent = 0.
    
    # training schedules
    cfg.first_joint_pose_nerf_then_nerf = False
    cfg.restart_nerf = False
    cfg.ratio_end_joint_nerf_pose_refinement = None

    cfg.clip_by_norm = True   # instead of by value
    cfg.nerf_gradient_clipping = 0.1
    cfg.pose_gradient_clipping = None
    cfg.arch = {}                                                    # architectural options

    # loss module
    cfg.loss_type = 'photometric'
    cfg.load_colmap_depth = False

    # data options
    cfg.dataset = None                                               # dataset name
    cfg.scene = None
    cfg.resize = None                                     # input image sizes [height,width], after cropping (if applicable)
    cfg.num_workers = 8                                          # number of parallel workers for data loading
    cfg.preload = False                                          # preload the entire dataset into the memory
    cfg.preload_images=False                                     # preload the images into the memory
    cfg.crop_ratio = None                                            # center crop the image by ratio, if empty, will not crop
    cfg.val_on_test = False                                      # validate on test set during training
    cfg.train_sub =  None                                           # consider a subset of N training samples
    cfg.val_sub =  None                                             # consider a subset of N validation samples
    cfg.mask_img = False
    
    ### others

    cfg.loss_weight = {}                                             # loss weights (in log scale)
    cfg.optim = edict()                                                  # optimization options
    cfg.optim.lr = 1.e-3                                               # learning rate (main)
    cfg.optim.lr_end =  None                                               # terminal learning rate (only used with sched.type=ExponentialLR)
    cfg.optim.weight_decay = 1e-4                                            # optimizer (see PyTorch doc)
    cfg.optim.sched = {}                                               # learning rate scheduling options
        # type = StepLR                                      # scheduler (see PyTorch doc)
        # steps =                                            # decay every N epochs
        # gamma = 0.1                                        # decay rate (can be empty if lr_end were specified)

    cfg.max_iter = 200000                                            # train to maximum number of iterations
    # periodic actions during training
    cfg.vis_steps = 1000                                            # visualize results (every N iterations)
    cfg.log_steps = 100                                             # log losses and scalar states (every N iterations)
    cfg.val_steps = 5000
    cfg.snapshot_steps = 5000                                       # save checkpoint (every N epochs), same than val step (every N epochs) ==> what is an epoch here?                                      
    return cfg


def get_nerf_default_config_llff():
    cfg_base = get_base_config()
    
    cfg = edict()
    cfg.model = 'nerf_gt_poses'                                                 # type of model (must be specified from command line)

    # architectural optionss
    cfg.arch = edict()                                                 
    cfg.arch.layers_feat = [None,256,256,256,256,256,256,256,256]     # hidden layers for feature/density MLP]
    cfg.arch.layers_feat_fine = None    # in case we want to use a different one for coarse and fine MLPs
    cfg.arch.layers_rgb = [None,128,3]                                # hidden layers for color MLP]
    cfg.arch.skip = [4]                                               # skip connections
    # positional encoding 
    cfg.arch.posenc = edict()    
    cfg.arch.posenc.include_pi_in_posenc = True
    cfg.arch.posenc.add_raw_3D_points = True
    cfg.arch.posenc.add_raw_rays = True
    cfg.arch.posenc.log_sampling = True                                             
    cfg.arch.posenc.L_3D = 10                                            # number of bases (3D point), put to 0 if no high frequency
    cfg.arch.posenc.L_view = 4                                           # number of bases (viewpoint), put to 0 is no high frequency
    cfg.arch.density_activ = 'softplus'                                 # activation function for output volume density
    cfg.arch.tf_init = True                                           # initialize network weights in TensorFlow style

    # NeRF-specific options
    cfg.nerf = edict()                                                  
    cfg.nerf.view_dep = True                                          # condition MLP on viewpoint
    cfg.nerf.depth = edict()                                                 # depth-related options
    cfg.nerf.depth.param = 'inverse'                                      # depth parametrization (for sampling along the ray)
    cfg.nerf.depth.range = [1, 0]

    cfg.nerf.sample_intvs = 128                                       # number of samples
    cfg.nerf.sample_stratified = True                                 # stratified sampling
    cfg.nerf.fine_sampling = False                                    # hierarchical sampling with another NeRF
    cfg.nerf.sample_intvs_fine = 128                                     # number of samples for the fine NeRF
    cfg.nerf.rand_rays = 2048                                        # number of random rays for each step, can be largely increased 
    cfg.nerf.density_noise_reg =  False                                    # Gaussian noise on density output as regularization
    cfg.nerf.setbg_opaque = False                                       # fill transparent rendering with known background color (Blender only)
    # cfg.nerf.ratio_start_fine_sampling_at_x = None 
    # cfg.nerf.start_fine_sampling_at_x = None

    # camera options
    cfg.camera = edict()                                                    
    cfg.camera.model = 'perspective'                                      # type of camera model
    cfg.camera.ndc = False                                              # reparametrize as normalized device coordinates (NDC)
 
    # sampling of rays
    cfg.precrop_frac = 0.5   # at the beginning, sample within the center for x iteration
    cfg.precrop_iters = 0  # at the beginning, sampled within the center
    cfg.sample_fraction_in_fg_mask = 0.  # whether to sample a fraction specifically in fg mask
    cfg.sampled_fraction_in_center = 0.
    cfg.depth_regu_patch_size = 2
    
    cfg.huber_loss_for_photometric = True 
    
    # loss weights (in log scale)
    cfg.loss_weight = edict() 
    cfg.loss_weight.equalize_losses = False    
    cfg.loss_weight.parametrization = 'exp'   # loss_weight are then in log_scale                                     
    cfg.loss_weight.render = 0  # RGB rendering loss, this is for 10^render, such that 0 corresponds to 1
    cfg.loss_weight.render_matches = None
    cfg.loss_weight.depth_patch = None  # for depth smoothness
    cfg.loss_weight.distortion = None  # for distortion loss (of mipnerf360)
    cfg.loss_weight.fg_mask = None  # for mask loss 
    cfg.loss_weight.corres = None    # for multi-view correspondence loss
    cfg.loss_weight.depth_cons = None  # for depth-consistency loss

    # start iterations of the different losses
    cfg.start_iter = edict()
    cfg.start_iter.photometric = 0
    cfg.start_iter.corres = 0
    cfg.start_iter.depth_cons = 0
    
    # start ratio (percentage of the total number of iterations) of the different losses
    cfg.start_ratio = edict()
    cfg.start_ratio.photometric = None
    cfg.start_ratio.corres = None
    cfg.start_ratio.depth_cons = None

    # handling of the multi-view correspondence loss
    cfg.gradually_decrease_corres_weight = False
    cfg.ratio_start_decrease_corres_weight = None
    cfg.iter_start_decrease_corres_weight = 0
    cfg.corres_weight_reduct_at_x_iter = 10000
    cfg.stop_corres_loss_at = None

    # handling of depth-consistency loss
    cfg.gradually_decrease_depth_cons_loss = False
    cfg.depth_cons_loss_reduct_at_x_iter = 10000

    # optimization options for network
    cfg.optim = edict() 
    cfg.optim.start_decrease = 0                                                      
    cfg.optim.lr = 1.e-3                                               # learning rate (main)
    cfg.optim.lr_end = 1.e-4                                           # terminal learning rate (only used with sched.type=ExponentialLR)
    cfg.optim.sched = edict()                                                 # learning rate scheduling options
    cfg.optim.sched.type = 'ExponentialLR'                                 # scheduler (see PyTorch doc)
    cfg.optim.sched.gamma = None                                             # decay rate (can be empty if lr_end were specified)


    # correspondence prediction parameters
    # USER NEEDS TO CHANGE THIS VALUE
    cfg.use_flow = False
    cfg.matching_pair_generation = 'all_to_all'  # how the pairs are selected 
    cfg.pairing_angle_threshold = 45  # if matching_pair_generation is 'angle', acceptable angle between valid pairs
    
    cfg.flow_backbone='PDCNet' 
    cfg.flow_ckpt_path='/home/jupyter/shared/pre_trained_models/PDCNet_megadepth.pth.tar'  # change
    cfg.use_homography_flow = False
    cfg.flow_batch_size = 5

    cfg.renderrepro_do_pixel_reprojection_check =  False
    cfg.renderrepro_do_depth_reprojection_check =  False 
    cfg.renderrepro_pixel_reprojection_thresh =  20.
    cfg.renderrepro_depth_reprojection_thresh =  0.1 

    cfg.filter_corr_w_cc = False
    cfg.min_conf_valid_corr =  0.95
    cfg.min_conf_cc_valid_corr =  1/(1.+1.5)
    cfg.min_nbr_matches =  500
    cfg.diff_loss_type =  'huber'

    cfg = override_options(cfg_base, cfg)
    return cfg


def get_joint_pose_nerf_default_config_llff():
    cfg_base = get_nerf_default_config_llff()

    cfg = edict()

    cfg.model = 'joint_pose_nerf_training'
    cfg.barf_c2f =  [0.3, 0.7]                                         # coarse-to-fine scheduling on positional encoding
    cfg.increase_depth_range_by_x_percent = 0.2

    # camera options
    cfg.camera = edict()   
    cfg.camera.pose_parametrization = 'two_columns'
    cfg.camera.optimize_c2w = False   # we want to optimize w2c  
    cfg.camera.optimize_trans = True  # we want to optimize both translation and rotation
    cfg.camera.optimize_rot = True
    cfg.camera.optimize_relative_poses = False   
    cfg.camera.n_first_fixed_poses = 0 
    cfg.camera.initial_pose = 'identity'                                            
    cfg.camera.noise = None   # synthetic perturbations on the camera poses

    # optimization options  
    cfg.optim = edict() 
    cfg.optim.algo_pose = 'Adam'                                                   
    cfg.optim.lr_pose =  3.e-3                                          # learning rate of camera poses
    cfg.optim.lr_pose_end =  1.e-5                                      # terminal learning rate of camera poses (only used with sched_pose.type=ExponentialLR)

    cfg.optim.sched_pose=  edict()                                           # learning rate scheduling options
    cfg.optim.sched_pose.type=  'ExponentialLR'                                 # scheduler (see PyTorch doc)
    cfg.optim.sched_pose.gamma=  None                                          # decay rate (can be empty if lr_pose_end were specified)
    
    cfg.optim.warmup_pose = None                                            # linear warmup of the pose learning rate (N iterations)
    cfg.optim.test_photo=  True                                        # test-time photometric optimization for evaluation
    cfg.optim.test_iter= 100                                          # number of iterations for test-time optimization

    cfg = override_options(cfg_base, cfg)
    return cfg


def get_nerf_default_config_360_data():
    default_config = get_nerf_default_config_llff()

    cfg = edict()
    cfg.model = 'nerf_gt_poses'

    cfg.nerf = edict()                                                       # NeRF-specific options
    cfg.nerf.depth = edict()                                                  # depth-related options
    cfg.nerf.depth.param = 'metric' 
    cfg.nerf.rand_rays = 1024

    cfg.optim = edict()                                                     # optimization options
    cfg.optim.start_decrease = 0 
    cfg.optim.lr = 5.e-4                                               # learning rate (main)
    cfg.optim.lr_end = 1.e-4                                           # terminal learning rate (only used with sched.type=ExponentialLR)
    cfg.optim.sched = edict()                                                  # learning rate scheduling options
    cfg.optim.sched.type = 'ExponentialLR'                                 # scheduler (see PyTorch doc)
    cfg.optim.sched.gamma = None 


    cfg.trimesh = edict()                                                   # options for marching cubes to extract 3D mesh 
    cfg.trimesh.res = 128                                                # 3D sampling resolution
    cfg.trimesh.range = [-1.2,1.2]                                       # 3D range of interest (assuming same for x,y,z)
    cfg.trimesh.thres = 25.                                              # volume density threshold for marching cubes
    cfg.trimesh.chunk_size = 16384                                       # chunk size of dense samples to be evaluated at a time 
    return override_options(default_config, cfg)


def get_joint_pose_nerf_default_config_360_data():
    default_cfg = get_nerf_default_config_360_data()

    cfg = edict()
    cfg.model = 'joint_pose_nerf_training'
    cfg.barf_c2f = [0.3, 0.7]                                                   # coarse-to-fine scheduling on positional encoding
    cfg.increase_depth_range_by_x_percent = 0.2

    # camera options
    cfg.camera = edict()   
    cfg.camera.pose_parametrization = 'two_columns'
    cfg.camera.optimize_c2w = False   # we want to optimize w2c
    cfg.camera.optimize_trans = True
    cfg.camera.optimize_rot = True
    cfg.camera.optimize_relative_poses = False   
    cfg.camera.n_first_fixed_poses = 0 
    cfg.camera.initial_pose = 'noisy_gt'                   
    cfg.camera.noise = 0.15    # synthetic perturbations on the camera poses (Blender only)


    cfg.optim = edict()
    cfg.optim.algo_pose = 'Adam'                                                            # optimization options
    cfg.optim.lr_pose = 1.e-3                                          # learning rate of camera poses
    cfg.optim.lr_pose_end = 1.e-4                                      # terminal learning rate of camera poses (only used with sched_pose.type=ExponentialLR)
    cfg.optim.sched_pose = edict()                                             # learning rate scheduling options
    cfg.optim.sched_pose.type = 'ExponentialLR'                                 # scheduler (see PyTorch doc)
    cfg.optim.sched_pose.gamma = None                                           # decay rate (can be empty if lr_pose_end were specified)
    cfg.optim.warmup_pose =  None                                         # linear warmup of the pose learning rate (N iterations)
    
    cfg.optim.test_photo = True                                        # test-time photometric optimization for evaluation
    cfg.optim.test_iter = 100                                          # number of iterations for test-time optimization
    return override_options(default_cfg, cfg)


def get_fixed_colmap_poses_default_config_360_data():
    default_cfg = get_nerf_default_config_360_data()

    cfg = edict()
    cfg.model = 'nerf_fixed_noisy_poses'
    cfg.increase_depth_range_by_x_percent = 0.2  # to account for errors in initial poses

    # camera options
    cfg.camera = edict()   
    # for initialization 
    cfg.camera.optimize_c2w = False  
    cfg.camera.optimize_trans = True
    cfg.camera.optimize_rot = True
    cfg.camera.optimize_relative_poses = False 
    cfg.camera.n_first_fixed_poses = 0 
    cfg.camera.initial_pose = 'sfm_pdcnet' 
    
    # needed for test-time photometric optimization             
    cfg.optim = edict()
    cfg.optim.algo_pose = 'Adam'                                                            # optimization options
    cfg.optim.lr_pose = 1.e-3                                          # learning rate of camera poses
    cfg.optim.lr_pose_end = 1.e-4                                      # terminal learning rate of camera poses (only used with sched_pose.type=ExponentialLR)
    cfg.optim.test_photo = True                                        # test-time photometric optimization for evaluation
    cfg.optim.test_iter = 100                                          # number of iterations for test-time optimization
    return override_options(default_cfg, cfg)
