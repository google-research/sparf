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

import os,sys
import torch
import sys
import torch
import configargparse
from easydict import EasyDict as edict
import json
from typing import Dict, Any, Tuple 

import source.admin.settings as ws_settings
from source.admin.loading import load_checkpoint
from source.utils.config_utils import load_options
from source.training.define_trainer import define_trainer
from source.training.engine.base_trainer import BaseTrainer
torch.set_grad_enabled(False)


def load_model(checkpoint_dir: str, settings: Dict[str, Any]):

    settings.args_to_update = {}
    
    if os.path.isdir(checkpoint_dir):
        # if it is a directory, it must have an option file
        checkpoint_path, weights = load_checkpoint(network_dir=checkpoint_dir, latest=True)
        config_path = os.path.join(checkpoint_dir, 'options.yaml')
        model_config = load_options(config_path)
    else:
        raise NotImplementedError
    
    model_config.test_metrics_only = True
    model_config.copy_data = False
    # if data is in zip file, and we need to copy and unzip first to the loggin node
    
    trainer = define_trainer(args=settings, settings_model=model_config, save_option=False)
    trainer.load_snapshot(checkpoint_path)
    return trainer


def run_eval(trainer: BaseTrainer, settings: Dict[str, Any], out_dir: str, expname: str, 
             plot: bool=False, save_ind_files: bool=False):
    """ Run final evaluation on the test set. Computes novel-view synthesis performance. 
    When the poses were optimized, also computes the pose registration error. Optionally, one can run
    test-time pose optimization to factor out the pose error from the novel-view synthesis performance. 
    """
    model_name = settings.model
    args = settings
    # the loss is redefined here as only the photometric one, in case the 
    # test-time photometric optimization is used
    args.loss_type = 'photometric'
    args.loss_weight.render = 0.
    
    # load the test step
    args.val_sub = None  
    trainer.load_dataset(args, eval_split='test')
    
    print("saving results to {}...".format(out_dir))
    os.makedirs(out_dir, exist_ok=True)
    
    out_dir_images = os.path.join(out_dir, expname)
    if plot:
        os.makedirs(out_dir_images, exist_ok=True)


    save_all = {}
    test_optim_options = [True, False] if model_name in ["joint_pose_nerf_training", 'nerf_fixed_noisy_poses'] else [False]
    for test_optim in test_optim_options:
        print('test pose optim : {}'.format(test_optim))
        args.optim.test_photo = test_optim

        possible_to_plot = True
        if test_optim is False and model_name in ["joint_pose_nerf_training", 'nerf_fixed_noisy_poses']:
            possible_to_plot = False
        results_dict = trainer.evaluate_full(args, plot=plot and possible_to_plot, 
                                             save_ind_files=save_ind_files and possible_to_plot, 
                                             out_scene_dir=out_dir_images)

        if test_optim:
            save_all['w_test_optim'] = results_dict
        elif model_name in ["joint_pose_nerf_training", 'nerf_fixed_noisy_poses']:
            save_all['without_test_optim'] = results_dict
        else:
            # nerf 
            save_all = results_dict
    
    save_all['iteration'] = trainer.iteration

    name_file = '{}.txt'.format(expname)
    print('Saving json file to {}/{}'.format(out_dir, name_file))
    with open("{}/{}".format(out_dir, name_file), "w+") as f:
        json.dump(save_all, f, indent=4)
    return 
    


def eval_parser() -> configargparse.ArgumentParser:
    '''
    use model parameter from checkpoint, only dataset options here
    '''
    parser = configargparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, required=True, 
                        help='Path to directory containing the checkpoint file and the option file')
    parser.add_argument('--data_root', type=str, default='/home/jupyter/shared',
                        help='Name of the train settings file.')
    parser.add_argument("--out_dir", type=str, default=None, required=True, 
                        help='Directory in which will be saved the metrics and' 
                              'potentially the plots')    
    parser.add_argument("--expname", type=str, default=None, required=True, help='Name of file to save')
    parser.add_argument("--plot", type=bool, default=False, help='plot the renderings?')
    parser.add_argument("--save_ind_files", type=bool, default=False, 
                        help='Save the rgb and depth renderings, gt images and depth as separate png?')
    return parser



if __name__ == '__main__':
    parser = eval_parser()
    args = edict(vars(parser.parse_args()))


    settings = edict(ws_settings.Settings(args.data_root).__dict__)
    args.update(settings)

    plot = args.plot
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    trainer = load_model(args.ckpt_dir, settings=args)
    run_eval(trainer, args, out_dir=args.out_dir, expname=args.expname, plot=plot)



