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

import numpy as np
import os,sys,time
import torch
import torch.utils.tensorboard
import json
from easydict import EasyDict as edict
import lpips
from typing import Callable, Sequence, List, Mapping, MutableMapping, Tuple, Union, Dict
from typing import Any, Optional

from source.training.core.metrics import compute_metrics_masked, compute_mse_on_rays
from third_party.pytorch_ssim.ssim import ssim as ssim_loss
from source.datasets.create_dataset import create_dataset
from source.training.engine.iter_based_trainer import IterBasedTrainer
from source.admin.loading import load_checkpoint, partial_load
from source.training.engine.iter_based_trainer import get_log_string
from source.utils.vis_rendering import *
from source.training.core.loss_factory import define_loss

# ============================ main engine for training and evaluation ============================
        
class PerSceneTrainer(IterBasedTrainer):
    """Base class for NeRF or joint pose-NeRF training and evaluation
    """

    def __init__(self,opt: Dict[str, Any]):
        super().__init__(settings=opt, max_iteration=opt.max_iter, snapshot_steps=opt.snapshot_steps, 
                         grad_acc_steps=opt.grad_acc_steps)

        self.lpips_loss = lpips.LPIPS(net="alex").to(self.device)

    def define_loss_module(self, opt: Dict[str, Any]):
        self.logger.info(f'Defining the loss: {opt.loss_type}')
        flow_net = self.flow_net if opt.use_flow else None

        self.loss_module = define_loss(opt.loss_type, opt, self.net, self.train_data, 
                                        self.device, flow_net=flow_net)

        # save the matching in tensorboard, when using flow_net
        to_plot = self.loss_module.plot_something()
        self.write_image('train', to_plot, self.iteration)

        # save the initial correspondence metrics, if ground-truth depth/poses are available
        epe_stats = self.loss_module.get_flow_metrics()
        self.write_event('train', epe_stats, self.iteration)
        return 

    def load_dataset(self,opt: Dict[str, Any],eval_split: str="val"):
        self.logger.info("loading training data...")
        self.train_data, train_sampler = create_dataset(opt, mode='train')
        train_loader = self.train_data.setup_loader(shuffle=True)
        
        self.logger.info("loading test data...")
        if opt.val_on_test: eval_split = "test"
        self.test_data = create_dataset(opt, mode=eval_split)
        test_loader = self.test_data.setup_loader(shuffle=False)
        self.register_loader(train_loader, test_loader)  # save them 
        return 

    def build_networks(self,opt: Dict[str, Any]):
        raise NotImplementedError

    def train_iteration_nerf(self, data_dict: Dict[str, Any]):
        """ Run one iteration of training. 
        Only the NeRF mlp is trained
        The pose network (if applicable) is frozen
        """
        # put to not training the weights of pose_net
        if hasattr(self, 'pose_net'):
            for p in self.pose_net.parameters():
                p.requires_grad = False
            self.pose_net.eval()

        # forward
        output_dict, result_dict, plotting_dict = self.train_step\
            (self.iteration, data_dict)

        # backward & optimization
        result_dict['loss'].backward()
        do_backprop = self.after_backward(self.net.get_network_components(), self.iteration,  
                                          gradient_clipping=self.settings.nerf_gradient_clipping)
        if self.iteration % self.grad_acc_steps == 0:
            if do_backprop:
                # could be skipped in case of large gradients or something, but we still want to put the optimizer
                # gradient to zero, so it is not further accumulated
                # skipped the accumulation gradient step
                self.optimizer.step()
            self.optimizer.zero_grad()

            # scheduler, here after each step
            if self.scheduler is not None: self.scheduler.step()
        return output_dict, result_dict, plotting_dict

    def train_iteration_nerf_pose_flow(self, data_dict: Dict[str, Any]):
        """ Run one iteration of training
        The nerf mlp is optimized
        The poses are also potentially optimized
        """
        self.iteration_nerf += 1

        # forward
        output_dict, result_dict, plotting_dict = self.train_step(self.iteration, data_dict)
        
        # backward & optimization
        self.update_parameters(result_dict['loss'])
        return output_dict, result_dict, plotting_dict


    def update_parameters(self, loss_out: Dict[str, Any]):
        """ Update weights of mlp"""
        loss_out.backward()
        do_backprop = self.after_backward(self.net.get_network_components(), self.iteration,  
                                          gradient_clipping=self.settings.nerf_gradient_clipping)
        if self.iteration % self.grad_acc_steps == 0:
            if do_backprop:
                # could be skipped in case of large gradients or something, but we still want to put the optimizer
                # gradient to zero, so it is not further accumulated
                # skipped the accumulation gradient step
                self.optimizer.step()
            self.optimizer.zero_grad()

            # scheduler, here after each step
            if self.scheduler is not None: self.scheduler.step()
        return 


    def train_iteration(self, data_dict: Dict[str, Any]):
        self.before_train_step(self.iteration, data_dict)
        self.timer.add_prepare_time()

        if self.settings.first_joint_pose_nerf_then_nerf:
            # define the iteration at which the poses should be frozen
            iter_end_joint_nerf_pose_refinement = self.settings.max_iter * \
                self.settings.ratio_end_joint_nerf_pose_refinement \
                if self.settings.ratio_end_joint_nerf_pose_refinement is not None else \
                    self.settings.end_joint_nerf_pose_refinement 
            
            if self.iteration < iter_end_joint_nerf_pose_refinement:
                # joint training of pose and NeRF
                output_dict, result_dict, plotting_dict = self.train_iteration_nerf_pose_flow(data_dict) 
            else:
                # training of NeRF with frozen poses
                if self.iteration == iter_end_joint_nerf_pose_refinement and self.settings.restart_nerf:
                    # restart nerf 
                    self.net.re_initialize() 

                    # define optimizer and scheduler
                    self.setup_optimizer(self.settings)
                self.iteration_nerf += 1
                output_dict, result_dict, plotting_dict = self.train_iteration_nerf(data_dict)
        else:
            # joint training of pose and NeRF
            output_dict, result_dict, plotting_dict = self.train_iteration_nerf_pose_flow(data_dict)  

        # after training
        self.timer.add_process_time()
        self.after_train_step(self.iteration, data_dict, output_dict, result_dict)
        result_dict = self.release_tensors(result_dict)
        
        self.summary_board.update_from_result_dict(result_dict)
        self.write_image('train', plotting_dict, self.iteration)
        return


    def set_train_mode(self):
        self.training = True
        self.net.train()
        if self.settings.use_flow:
            # THIS IS IMPORTANT
            # should always be in eval mode 
            self.flow_net.eval()
        torch.set_grad_enabled(True)

    def set_eval_mode(self):
        self.training = False
        self.net.eval()
        if self.settings.use_flow:
            self.flow_net.eval()
        torch.set_grad_enabled(False)
    
    def return_model_dict(self):
        state_dict = {}
        state_dict['nerf_net'] = self.net.state_dict()

        if hasattr(self, 'pose_net'):
            state_dict['pose_net'] = self.pose_net.state_dict()
        return state_dict
    
    def load_state_dict(self, model_dict, do_partial_load=False):

        if not 'nerf_net' in model_dict.keys():
            if do_partial_load:
                partial_load(model_dict, self.net, skip_keys=[])
            else:
                self.net.load_state_dict(model_dict, strict=True)   
            return
        
        assert 'nerf_net' in model_dict.keys()
        self.logger.info('Loading the nerf model')
        if do_partial_load:
            partial_load(model_dict['nerf_net'], self.net, skip_keys=[])
        else:
            self.net.load_state_dict(model_dict['nerf_net'], strict=True) 

        if 'flow_net' in model_dict.keys():
            self.logger.info('Loading flow net')
            self.flow_net.load_state_dict(model_dict['flow_net'], strict=True)

        if 'pose_net' in model_dict.keys():
            self.logger.info('Loading the poses')
            self.pose_net.load_state_dict(model_dict['pose_net'], strict=True)
        return 
    
    def run_debug(self, load_latest: bool=False, make_validation_first: bool=False):
        """
        Main training loop function. 
        Here, load the whole training data for each iteration!
        """
        assert self.train_loader is not None
        assert self.val_loader is not None

        if self.settings.render_video_pose_only:
            self.generate_videos_pose(self.settings)
            return 

        if self.settings.resume_snapshot is not None:
            # this is where it should load the weights of model only!!!
            checkpoint_path, weights = load_checkpoint(checkpoint=self.settings.resume_snapshot)
            self.load_state_dict(weights['state_dict'], do_partial_load=True)
            if 'nerf.progress' not in weights['state_dict']:
                self.net.nerf.progress.data.fill_(1.)

        self.just_started = True
        if load_latest:
            self.load_snapshot()

        if self.settings.test_metrics_only:
            self.eval_after_training(plot=self.settings.plot, 
                                     save_ind_files=self.settings.save_ind_files)
            return 

        if make_validation_first and self.iteration == 0:
            self.inference_debug()
        self.set_train_mode()

        self.summary_board.reset_all()
        self.timer.reset()

        self.before_train()
        self.optimizer.zero_grad()
        if self.optimizer_pose is not None: self.optimizer_pose.zero_grad()
        
        initial_it = self.iteration
        # important to exclude it for inference pose estimation
        while self.iteration < initial_it + 10:
            self.iteration += 1

            # here loads the full scene for training
            data_dict = self.train_data.all
            data_dict['iter'] = self.iteration
            # dict_keys(['idx', 'image', 'intr', 'pose']), all images of the scene already stacked here. 

            self.train_iteration(data_dict)  # where the loss is computed, gradients backpropagated and so on, 

            # logging
            self.log_steps = 1
            if self.iteration % self.log_steps == 0:
                summary_dict = self.summary_board.summary()
                message = '[Train] ' + get_log_string(
                    result_dict=summary_dict,
                    iteration=self.iteration,
                    max_iteration=self.max_iteration,
                    lr=self.get_lr(),
                    timer=self.timer,
                )
                self.logger.info(message)
                self.write_event('train', summary_dict, self.iteration)

            if self.iteration % self.settings.val_steps == 0:
                self.writer.flush()
                # run validation
                self.inference_debug()

                self.epoch = self.iteration
                # save best checkpoint
                if self.current_best_val is not None and self.current_best_val < self.best_val:
                    message = 'VALIDATION IMPROVED ! From best value = {} at iteration {} to best value = {} ' \
                              'at current iteration {}'.format(self.best_val, self.epoch_of_best_val,
                                                               self.current_best_val, self.iteration)
                    self.logger.critical(message)
                    self.best_val = self.current_best_val  # update best_val
                    self.epoch_of_best_val = self.iteration  # update_epoch_of_best_val

                    self.save_snapshot('model_best.pth.tar', training_info=False)

            # snapshot & validation, every certain number of iterations
            if self.iteration % self.snapshot_steps == 0:
                self.epoch = self.iteration
                self.save_snapshot(f'iter-{self.iteration}.pth.tar')
                self.delete_old_checkpoints()  # keep only the most recent set of checkpoints
            torch.cuda.empty_cache()

        self.after_train()
        message = 'Training finished.'
        self.generate_videos_synthesis(self.settings) 
        # self.eval_after_training(plot=self.settings.plot)
        # self.eval_after_training(load_best_model=True)
        self.logger.critical(message)
    
    def generate_videos_pose(self, opt: Dict[str, Any]):
        raise NotImplementedError

    def run(self, load_latest: bool=True, make_validation_first: bool=False):
        """
        Main training loop function
        Here, load the whole training data for each iteration!
        """
        assert self.train_loader is not None
        assert self.val_loader is not None

        if self.settings.render_video_pose_only:
            self.generate_videos_pose(self.settings)
            return 

        if self.settings.resume_snapshot is not None:
            # this is where it should load the weights of model only!!!
            checkpoint_path, weights = load_checkpoint(checkpoint=self.settings.resume_snapshot)
            self.load_state_dict(weights['state_dict'], do_partial_load=True)
            if 'nerf.progress' not in weights['state_dict']:
                self.net.nerf.progress.data.fill_(1.)

        self.just_started = True
        if load_latest:
            self.load_snapshot()

        if self.settings.test_metrics_only:
            self.eval_after_training(plot=self.settings.plot, 
                                     save_ind_files=self.settings.save_ind_files)
            self.eval_after_training(load_best_model=True)
            return 

        if self.settings.render_video_only:
            self.generate_videos_synthesis(self.settings)
            return 
            
        if make_validation_first and self.iteration == 0:
            self.inference()
            
        self.set_train_mode()
        self.summary_board.reset_all()
        self.timer.reset()

        self.before_train()
        self.optimizer.zero_grad()
        if self.optimizer_pose is not None: self.optimizer_pose.zero_grad()

        while self.iteration < self.max_iteration:
            self.iteration += 1

            # here loads the full scene for training, i.e. all the images 
            data_dict = self.train_data.all
            data_dict['iter'] = self.iteration
            # dict_keys(['idx', 'image', 'intr', 'pose']), all images of the scene already stacked here. 

            self.train_iteration(data_dict)  # where the loss is computed, gradients backpropagated and so on, 

            # logging
            if self.iteration % self.log_steps == 0:
                summary_dict = self.summary_board.summary()
                message = '[Train] ' + get_log_string(
                    result_dict=summary_dict,
                    iteration=self.iteration,
                    max_iteration=self.max_iteration,
                    lr=self.get_lr(),
                    timer=self.timer,
                )
                self.logger.info(message)
                self.write_event('train', summary_dict, self.iteration)

            # validation, every certain number of iterations
            if self.iteration % self.settings.val_steps == 0:
                self.writer.flush()
                # run validation
                self.inference()

                self.epoch = self.iteration
                # save best checkpoint
                if self.current_best_val is not None and self.current_best_val < self.best_val:
                    message = 'VALIDATION IMPROVED ! From best value = {} at iteration {} to best value = {} ' \
                              'at current iteration {}'.format(self.best_val, self.epoch_of_best_val,
                                                           self.current_best_val, self.iteration)
                    self.logger.critical(message)
                    self.best_val = self.current_best_val  # update best_val
                    self.epoch_of_best_val = self.iteration  # update_epoch_of_best_val

                    self.save_snapshot('model_best.pth.tar', training_info=False)

            # snapshot, every certain number of iterations
            if self.iteration % self.snapshot_steps == 0:
                # snapshot saving and so on
                self.epoch = self.iteration
                self.save_snapshot(f'iter-{self.iteration}.pth.tar')
                self.delete_old_checkpoints()  # keep only the most recent set of checkpoints

            torch.cuda.empty_cache()

        self.after_train()
        
        # run validation
        self.inference()
        message = 'Training finished.'
        self.generate_videos_synthesis(self.settings) 
        if self.settings.do_eval:
            # save metrics! 
            message = 'Training finished. Running evaluation.'
            self.eval_after_training(plot=self.settings.plot, 
                                     save_ind_files=self.settings.save_ind_files)
            self.eval_after_training(load_best_model=True)
        self.logger.critical(message)
        
    @torch.no_grad()
    def make_result_dict(self,opt: Dict[str, Any],data_dict: Dict[str, Any],output_dict: Dict[str, Any],
                         loss: Dict[str, Any],metric: Dict[str, Any]=None, split: str='train'):
        stats_dict = {}
        for key,value in loss.items():
            if key=="all": continue
            stats_dict["loss_{}".format(key)] = value
        if metric is not None:
            for key,value in metric.items():
                stats_dict["{}".format(key)] = value
        return stats_dict


    @torch.no_grad()
    def val_step(self, iteration: int, data_dict: Dict[str, Any]
                 ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]: 
        data_dict = edict(data_dict)
        H, W = data_dict.image.shape[-2:] 
        plotting_dict = {}

        output_dict = self.net.forward(self.settings,data_dict,mode="val",iter=self.iteration)  
        # will render the full image
        output_dict['mse'], output_dict['mse_fine'] = compute_mse_on_rays(data_dict, output_dict)

        # to compute the loss:
        # poses_w2c = self.net.get_w2c_pose(self.settings, data_dict, mode='val')  
        # data_dict.poses_w2c = poses_w2c
        # loss_dict, stats_dict, plotting_dict = self.loss_module.compute_loss\
        #     (self.settings, data_dict, output_dict, iteration=iteration, mode="val")

        results_dict = self.make_result_dict(self.settings, data_dict, output_dict, loss={}, split='val')
        # results_dict.update(stats_dict)
        # results_dict['loss'] = loss_dict['all']
        
        results_dict['best_value'] = - results_dict['PSNR_fine'] if 'PSNR_fine' in results_dict.keys() \
            else - results_dict['PSNR']
        
        # run some evaluations
        gt_image = data_dict.image.reshape(-1, 3, H, W)
        
        # coarse prediction
        pred_rgb_map = output_dict.rgb.reshape(-1, H, W, 3).permute(0,3,1,2)  # (B, 3, H, W)
        ssim = ssim_loss(pred_rgb_map, gt_image).item()
        lpips = self.lpips_loss(pred_rgb_map*2-1, gt_image*2-1).item()

        results_dict['ssim'] = ssim
        results_dict['lpips'] = lpips

        if 'fg_mask' in data_dict.keys():
            results_dict.update(compute_metrics_masked(data_dict, pred_rgb_map, gt_image, 
                                                       self.lpips_loss, suffix=''))

        if 'rgb_fine' in output_dict.keys():
            pred_rgb_map = output_dict.rgb_fine.reshape(-1, H, W, 3).permute(0,3,1,2)
            ssim = ssim_loss(pred_rgb_map, gt_image).item()
            lpips = self.lpips_loss(pred_rgb_map*2-1, gt_image*2-1).item()

            results_dict['ssim_fine'] = ssim
            results_dict['lpips_fine'] = lpips
            
            if 'fg_mask' in data_dict.keys():
                results_dict.update(compute_metrics_masked(data_dict, pred_rgb_map, gt_image, 
                                                           self.lpips_loss, suffix='_fine'))

        if iteration < 5 or (iteration % 4 == 0): 
            plotting_dict_ = self.visualize(self.settings, data_dict, output_dict, step=iteration, split="val")
            plotting_dict.update(plotting_dict_)
        return output_dict, results_dict, plotting_dict 

    def eval_after_training(self, load_best_model: bool=False, plot: bool=False, 
                            save_ind_files: bool=False):
        """ Run final evaluation on the test set. Computes novel-view synthesis performance. 
        When the poses were optimized, also computes the pose registration error. Optionally, one can run
        test-time pose optimization to factor out the pose error from the novel-view synthesis performance. 
        """
        self.logger.info('DOING EVALUATION')
        model_name = self.settings.model
        args = self.settings
        args.expname = self.settings.script_name

        # the loss is redefined here as only the photometric one, in case the 
        # test-time photometric optimization is used
        args.loss_type = 'photometric'
        args.loss_weight.render = 0.

        if load_best_model:
            checkpoint_path = '{}/{}/model_best.pth.tar'.format(self._base_save_dir, self.settings.project_path)
            self.logger.info('Loading {}'.format(checkpoint_path))
            weights =  torch.load(checkpoint_path, map_location=torch.device('cpu'))
            if hasattr(self, 'load_state_dict'):
                self.load_state_dict(weights['state_dict'])
            else:
                self.net.load_state_dict(weights['state_dict'], strict=True)
            if 'iteration' in weights.keys():
                self.iteration = weights['iteration']

        # define name of experiment
        dataset_name = args.dataset
        if hasattr(args, 'factor') and args.factor != 1:
            dataset_name += "_factor_" + str(args.factor)
        elif hasattr(args, 'llff_img_factor') and args.llff_img_factor != 1:
            dataset_name += "_factor_" + str(args.llff_img_factor)
        if hasattr(args, 'resize') and args.resize:
            dataset_name += "_{}x{}".format(args.resize[0], args.resize[1]) if len(args.resize) == 2 else "_" + str(args.resize)
        elif hasattr(args, 'resize_factor') and args.resize_factor:
            dataset_name += "_resizefactor_" + str(args.resize_factor)
        
        # define the output directory where the metrics (and qualitative results) will be stored
        if self.debug:
            out_dir = os.path.join(args.env.eval_dir + '_debug', dataset_name)
        else:
            out_dir = os.path.join(args.env.eval_dir, dataset_name)

        if args.train_sub is None:
            out_dir = os.path.join(out_dir, 'all_training_views')
        else:
            out_dir = os.path.join(out_dir, f'{args.train_sub}_training_views')
            
        out_dir = os.path.join(out_dir, args.scene)
        out_dir = os.path.join(out_dir, self.settings.module_name_for_eval)
        extra_out_dir = os.path.join(out_dir, args.expname + '_{}'.format(self.iteration))  # to save qualitative figures

        self.logger.critical('Experiment: {} / {}'.format(self.settings.module_name, args.expname))
        self.logger.critical("saving results to {}...".format(out_dir))
        os.makedirs(out_dir, exist_ok=True)

        # load the test step
        args.val_sub = None  
        self.load_dataset(args, eval_split='test')
        
        save_all = {}
        test_optim_options = [True, False] if model_name in ["joint_pose_nerf_training", 'nerf_fixed_noisy_poses'] else [False]
        for test_optim in test_optim_options:
            self.logger.info('test pose optim : {}'.format(test_optim))
            args.optim.test_photo = test_optim

            possible_to_plot = True
            if test_optim is False and model_name in ["joint_pose_nerf_training", 'nerf_fixed_noisy_poses']:
                possible_to_plot = False
            results_dict = self.evaluate_full(args, plot=plot and possible_to_plot, 
                                              save_ind_files=save_ind_files and possible_to_plot, 
                                              out_scene_dir=extra_out_dir)

            if test_optim:
                save_all['w_test_optim'] = results_dict
            elif model_name in ["joint_pose_nerf_training", 'nerf_fixed_noisy_poses']:
                save_all['without_test_optim'] = results_dict
            else:
                # nerf 
                save_all = results_dict
        
        save_all['iteration'] = self.iteration

        if load_best_model:
            name_file = '{}_best_model.txt'.format(args.expname)
        else:
            name_file = '{}.txt'.format(args.expname)
        self.logger.critical('Saving json file to {}/{}'.format(out_dir, name_file))
        with open("{}/{}".format(out_dir, name_file), "w+") as f:
            json.dump(save_all, f, indent=4)
        return 


    @torch.no_grad()
    def visualize(self,opt: Dict[str, Any],data_dict: Dict[str, Any],output_dict: Dict[str, Any],
                  step: int=0, split: str="train",eps: float=1e-10
                  ) -> Dict[str, Any]:
        """Creates visualization of renderings and gt. Here N is HW

        Attention:
            ground-truth image has shape (B, 3, H, W)
            rgb rendering has shape (B, H*W, 3)
        Args:
            opt (edict): settings
            data_dict (edict): Input data dict. Contains important fields:
                           - Image: GT images, (B, 3, H, W)
                           - intr: intrinsics (B, 3, 3)
                           - idx: idx of the images (B)
                           - depth_gt (optional): gt depth, (B, 1, H, W)
                           - valid_depth_gt (optional): (B, 1, H, W)
            output_dict (edict): Output dict from the renderer. Contains important fields
                             - idx_img_rendered: idx of the images rendered (B), useful 
                             in case you only did rendering of a subset
                             - ray_idx: idx of the rays rendered, either (B, N) or (N)
                             - rgb: rendered rgb at rays, shape (B, N, 3)
                             - depth: rendered depth at rays, shape (B, N, 1)
                             - rgb_fine: rendered rgb at rays from fine MLP, if applicable, shape (B, N, 3)
                             - depth_fine: rendered depth at rays from fine MLP, if applicable, shape (B, N, 1)
            step (int, optional): Defaults to 0.
            split (str, optional): Defaults to "train".
        """
        
        plotting_stats = {}
        to_plot, to_plot_fine = [], []

        # compute scaling factor for the depth, if the poses are optimized
        scaling_factor_for_pred_depth = 1.
        if self.settings.model == 'joint_pose_nerf_training' and hasattr(self.net, 'sim3_est_to_gt_c2w'):
            # adjust the rendered depth, since the optimized scene geometry and poses are valid up to a 3D
            # similarity, compared to the ground-truth. 
            scaling_factor_for_pred_depth = (self.net.sim3_est_to_gt_c2w.trans_scaling_after * self.net.sim3_est_to_gt_c2w.s) \
                if self.net.sim3_est_to_gt_c2w.type == 'align_to_first' else self.net.sim3_est_to_gt_c2w.s

        H, W = data_dict.image.shape[-2:]
        # cannot visualize if it is not rendering the full image!
        depth_map = output_dict.depth.view(-1,H,W,1)[0] * scaling_factor_for_pred_depth 
        # [B,H,W, 1] and then (H, W, 1)
        depth_map_var = output_dict.depth_var.view(-1,H,W,1)[0] # [B,H,W, 1] and then (H, W, 1)
        # invdepth = (1-var.depth)/var.opacity if opt.camera.ndc else 1/(var.depth/var.opacity+eps)
        # invdepth_map = invdepth.view(-1,opt.H,opt.W,1)[0] # [B,H,W, 1] and then (H, W, 1)
        rgb_map = output_dict.rgb.view(-1,H,W,3)[0] # [B,H,W, 3] and then (H, W, 3)
        rgb_map_var = output_dict.rgb_var.view(-1, H, W, 1)[0]  # (H, W, 1)
        idx_img_rendered = output_dict.idx_img_rendered[0]

        opacity = output_dict.opacity.view(-1, H, W, 1)[0]
        
        image = (data_dict.image.permute(0, 2, 3, 1)[idx_img_rendered].cpu().numpy() * 255.).astype(np.uint8) # (B, 3, H, W), then (H, W, 3)
        depth_range = None
        if hasattr(data_dict, 'depth_range') and opt.nerf.depth.param == 'metric':
            depth_range = data_dict.depth_range[idx_img_rendered].cpu().numpy().tolist()
            depth_range[0] = min(depth_range[0], depth_map.min().item())
            depth_range[1] = max(depth_range[1], depth_map.max().item())

        fine_pred_rgb_np_uint8 = (255 * np.clip(rgb_map.cpu().numpy(), a_min=0, a_max=1.)).astype(np.uint8)

        # apperance color error
        image_rgb_map_error = np.sum((image/255. - fine_pred_rgb_np_uint8/255.)**2, axis=-1)
        image_rgb_map_error = (colorize_np(image_rgb_map_error, range=(0, 1)) * 255.).astype(np.uint8)

        pred_image_var_colored = colorize_np(rgb_map_var.cpu().squeeze().numpy())
        pred_image_var_colored = (255 * pred_image_var_colored).astype(np.uint8)

        fine_pred_depth_colored = colorize_np(depth_map.cpu().squeeze().numpy(), range=depth_range)
        fine_pred_depth_colored = (255 * fine_pred_depth_colored).astype(np.uint8)

        pred_depth_var_colored = colorize_np(depth_map_var.cpu().squeeze().numpy())
        pred_depth_var_colored = (255 * pred_depth_var_colored).astype(np.uint8)

        opacity = (opacity.cpu().squeeze().unsqueeze(-1).repeat(1, 1, 3).numpy() * 255).astype(np.uint8)
        to_plot += [torch.from_numpy(x.astype(np.float32)/255.) for x in 
                    [image, fine_pred_rgb_np_uint8, fine_pred_depth_colored, opacity, 
                    pred_image_var_colored, pred_depth_var_colored, image_rgb_map_error]]
        
        name = 'gt-predc-depthc-acc-rgbvarc-depthvarc-err'        
        
        if 'depth_fine' in output_dict:
            depth_map = output_dict.depth_fine.view(-1,H,W,1)[0] * scaling_factor_for_pred_depth
            # [B,H,W, 1] and then (H, W, 1)
            depth_map_var = output_dict.depth_var_fine.view(-1,H,W,1)[0] # [B,H,W, 1] and then (H, W, 1)
            # invdepth = (1-var.depth_fine)/var.opacity_fine if opt.camera.ndc else 1/(var.depth_fine/var.opacity_fine+eps)
            # invdepth_map = invdepth.view(-1,opt.H,opt.W,1)[0]
            rgb_map = output_dict.rgb_fine.view(-1,H,W,3)[0]
            rgb_map_var = output_dict.rgb_var_fine.view(-1, H, W, 1)[0]  # (H, W, 1)

            opacity = output_dict.opacity_fine.view(-1, H, W, 1)[0]

            fine_pred_rgb_np_uint8 = (255 * np.clip(rgb_map.cpu().numpy(), a_min=0, a_max=1.)).astype(np.uint8)

            # apperance color error
            image_rgb_map_error = np.sum((image/255. - fine_pred_rgb_np_uint8/255.)**2, axis=-1)
            image_rgb_map_error = (colorize_np(image_rgb_map_error, range=(0, 1)) * 255.).astype(np.uint8)

            pred_image_var_colored = colorize_np(rgb_map_var.cpu().squeeze().numpy())
            pred_image_var_colored = (255 * pred_image_var_colored).astype(np.uint8)

            fine_pred_depth_colored = colorize_np(depth_map.cpu().squeeze().numpy(), range=depth_range)
            fine_pred_depth_colored = (255 * fine_pred_depth_colored).astype(np.uint8)

            pred_depth_var_colored = colorize_np(depth_map_var.cpu().squeeze().numpy())
            pred_depth_var_colored = (255 * pred_depth_var_colored).astype(np.uint8)

            opacity = (opacity.cpu().squeeze().unsqueeze(-1).repeat(1, 1, 3).numpy() * 255).astype(np.uint8)
            to_plot_fine += [torch.from_numpy(x.astype(np.float32)/255.) for x in  \
                [image, fine_pred_rgb_np_uint8, fine_pred_depth_colored, pred_depth_var_colored, opacity, 
                pred_image_var_colored, image_rgb_map_error]]

        if 'depth_gt' in data_dict.keys():
            depth_gt = data_dict.depth_gt[idx_img_rendered]
            depth_gt_colored = (255 * colorize_np(depth_gt.cpu().squeeze().numpy(), range=depth_range)).astype(np.uint8)
            to_plot += [torch.from_numpy(depth_gt_colored.astype(np.float32)/255.)]
            if len(to_plot_fine) > 0:
                to_plot_fine += [torch.from_numpy(depth_gt_colored.astype(np.float32)/255.)]
            name += '-depthgt'
        
        to_plot_img = torch.stack(to_plot, dim=0) # (N, H, W, 3)
        if len(to_plot_fine) > 0:
            to_plot_img = torch.cat((to_plot_img, torch.stack(to_plot_fine, dim=0)), dim=1) # (N, 2H, W, 3)
        to_plot = img_HWC2CHW(to_plot_img)  # (N, 3, 2H, W)
        plotting_stats[f'{split}_{step}/{name}'] = to_plot
        return plotting_stats

    @torch.no_grad()
    def save_ind_files(self, save_dir: str, name: str, image: torch.Tensor, 
                       rendered_img: torch.Tensor, rendered_depth: torch.Tensor, 
                       depth_range: List[float]=None, depth_gt: torch.Tensor=None):
        """Save rendering and ground-truth data as individual files. 
        
        Args:
            save_dir (str): dir to save the images
            name (str): name of image (without extension .png)
            image (torch.Tensor): gt image of shape [1, 3, H, W]
            rendered_img (torch.Tensor): rendered image of shape [1, 3, H, W]
            rendered_depth (torch.Tensor): rendered depth of shape [1, H, W, 1]
            depth_range (list of floats): depth range for depth visualization
            depth_gt (torch.Tensor): gt depth of shape [1, H, W, 1]
        """
        rend_img_dir = os.path.join(save_dir, 'rendered_imgs')
        rend_depth_dir = os.path.join(save_dir, 'rendered_depths')
        gt_img_dir = os.path.join(save_dir, 'gt_imgs')
        gt_depth_dir = os.path.join(save_dir, 'gt_depths')
        if not os.path.exists(rend_img_dir):
            os.makedirs(rend_img_dir, exist_ok=True)
        if not os.path.exists(gt_img_dir):
            os.makedirs(gt_img_dir, exist_ok=True)
        if not os.path.exists(rend_depth_dir):
            os.makedirs(rend_depth_dir, exist_ok=True)
        if not os.path.exists(gt_depth_dir):
            os.makedirs(gt_depth_dir, exist_ok=True)

        image = (image.permute(0, 2, 3, 1)[0].cpu().numpy() * 255.).astype(np.uint8) # (B, H, W, 3), B is 1
        imageio.imwrite(os.path.join(gt_img_dir, name + '.png'), image)
        H, W = image.shape[1:3]
        # cannot visualize if it is not rendering the full image!

        rgb_map = rendered_img.permute(0, 2, 3, 1)[0].cpu().numpy() # [B,3, H,W] and then (H, W, 3)
        fine_pred_rgb_np_uint8 = (255 * np.clip(rgb_map, a_min=0, a_max=1.)).astype(np.uint8)
        imageio.imwrite(os.path.join(rend_img_dir, name + '.png'), fine_pred_rgb_np_uint8)


        depth = rendered_depth[0].squeeze().cpu().numpy() # [B,H,W, 1] and then (H, W)
        fine_pred_depth_colored = colorize_np(depth, range=depth_range, append_cbar=False)
        fine_pred_depth_colored = (255 * fine_pred_depth_colored).astype(np.uint8)
        imageio.imwrite(os.path.join(rend_depth_dir, name + '.png'), fine_pred_depth_colored)

        if depth_gt is not None:
            depth = depth_gt[0].squeeze().cpu().numpy() # [B,H,W, 1] and then (H, W, 1)
            fine_pred_depth_colored = colorize_np(depth, range=depth_range, append_cbar=False)
            fine_pred_depth_colored = (255 * fine_pred_depth_colored).astype(np.uint8)
            imageio.imwrite(os.path.join(gt_depth_dir, name + '.png'), fine_pred_depth_colored)
        return 

    @torch.no_grad()
    def visualize_eval(self, to_plot: List[Any], image: torch.Tensor, rendered_img: torch.Tensor, 
                       rendered_depth: torch.Tensor, rendered_depth_var: torch.Tensor, depth_range: List[float]=None):
        """Visualization for the test set"""

        image = (image.permute(0, 2, 3, 1)[0].cpu().numpy() * 255.).astype(np.uint8) # (B, H, W, 3), B is 1
        H, W = image.shape[1:3]
        # cannot visualize if it is not rendering the full image!
        depth = rendered_depth[0].squeeze().cpu().numpy() # [B,H,W, 1] and then (H, W, 1)
        depth_var = rendered_depth_var[0].squeeze().cpu().numpy() # [B,H,W, 1] and then (H, W, 1)
        rgb_map = rendered_img.permute(0, 2, 3, 1)[0].cpu().numpy() # [B,3, H,W] and then (H, W, 3)

        
        fine_pred_rgb_np_uint8 = (255 * np.clip(rgb_map, a_min=0, a_max=1.)).astype(np.uint8)

        fine_pred_depth_colored = colorize_np(depth, range=depth_range, append_cbar=False)
        fine_pred_depth_colored = (255 * fine_pred_depth_colored).astype(np.uint8)

        fine_pred_depth_var_colored = colorize_np(depth_var, append_cbar=False)
        fine_pred_depth_var_colored = (255 * fine_pred_depth_var_colored).astype(np.uint8)

        to_plot += [image, fine_pred_rgb_np_uint8, fine_pred_depth_colored, fine_pred_depth_var_colored]
        return 
