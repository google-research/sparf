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

from typing import Tuple, Dict
import torch
from typing import Any, Optional, Dict, Tuple

from source.training.engine.base_trainer import BaseTrainer
from source.utils.summary_board import SummaryBoard
from source.utils.timer import Timer
from source.utils.torch import get_log_string


class CycleLoader(object):
    def __init__(self, data_loader: torch.utils.data.DataLoader, epoch: int, distributed):
        self.data_loader = data_loader
        self.last_epoch = epoch
        # epoch here is when went through the whole dataset. 
        self.distributed = distributed
        self.iterator = self.initialize_iterator()

    def initialize_iterator(self):
        if self.distributed:
            self.data_loader.sampler.set_epoch(self.last_epoch + 1)
        return iter(self.data_loader)

    def __next__(self):
        try:
            data_dict = next(self.iterator)
        except StopIteration:
            self.last_epoch += 1
            self.iterator = self.initialize_iterator()
            data_dict = next(self.iterator)
        return data_dict


class IterBasedTrainer(BaseTrainer):
    def __init__(
        self,
        settings: Dict[str, Any],
        max_iteration: int,
        snapshot_steps: int,
        cudnn_deterministic: bool=True,
        autograd_anomaly_detection: bool=False,
        run_grad_check: bool=False,
        grad_acc_steps: int=1,
    ):
        super().__init__(
            settings,
            cudnn_deterministic=cudnn_deterministic,
            autograd_anomaly_detection=autograd_anomaly_detection,
            run_grad_check=run_grad_check,
            grad_acc_steps=grad_acc_steps,
        )
        self.max_iteration = max_iteration
        self.snapshot_steps = snapshot_steps
        self.val_steps = self.settings.val_steps
        self.scheduler_per_iteration = True

    def before_train(self) -> None:
        pass

    def after_train(self) -> None:
        pass

    def before_val(self) -> None:
        pass

    def after_val(self) -> None:
        pass

    def before_train_step(self, iteration: int, data_dict: Dict[str, Any]) -> None:
        pass

    def before_val_step(self, iteration: int, data_dict: Dict[str, Any]) -> None:
        pass

    def after_train_step(self, iteration: int, data_dict: Dict[str, Any], output_dict: Dict[str, Any], result_dict: Dict[str, Any]) -> None:
        pass

    def after_val_step(self, iteration: int, data_dict: Dict[str, Any], output_dict: Dict[str, Any], result_dict: Dict[str, Any]) -> None:
        pass

    def train_step(self, iteration: int, data_dict: Dict[str, Any]) -> Tuple[Dict, Dict, Dict]:
        pass

    def val_step(self, iteration: int, data_dict: Dict[str, Any]) -> Tuple[Dict, Dict, Dict]:
        pass

    def after_backward(self, net: torch.nn.Module, iteration: int, 
                       data_dict: Dict[str, Any]=None, output_dict: Dict[str, Any]=None, 
                       result_dict: Dict[str, Any]=None, gradient_clipping: float=None) -> bool:
        """ Returns if back propagation should actually happen for this step.
        Also can do some processing of the gradients like gradient clipping

        Args:
            net (nn.model, (list, tuple)): list of networks, or network 
            iteration (int): 
            data_dict (edict, optional):  Defaults to None.
            output_dict (edict, optional): Defaults to None.
            result_dict (edict, optional): Defaults to None.
            gradient_clipping (float, optional): Defaults to None.

        Returns:
            bool: do back_prop?
        """

        # gather all the network parameters
        if isinstance(net, (list, tuple)):
            parameters = []
            for i in range(len(net)):
                parameters += list(net[i].parameters())
        else:
            parameters = net.parameters()

        # check the gradients, if there are Nan or Inf
        do_backprop = self.check_invalid_gradients(net)

        # skipping gradients
        if do_backprop and hasattr(self.settings, 'skip_large_gradients') and self.settings.skip_large_gradients:
            if isinstance(parameters, torch.Tensor):
                parameters = [parameters]
            parameters = [p for p in parameters if p.grad is not None]
            device = parameters[0].grad.device
            norms = [p.grad.detach().abs().max().to(device) for p in parameters]
            max_ = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
            if max_ > self.settings.skip_large_gradients:
                do_backprop = False  # will skip optimizer

        # gradient clipping
        if do_backprop and gradient_clipping is not None:
            if self.settings.clip_by_norm:
                torch.nn.utils.clip_grad_norm_(parameters, gradient_clipping)
            else:
                # clip by value
                torch.nn.utils.clip_grad_value_(parameters, gradient_clipping)
        
        # printing gradients
        if hasattr(self.settings, 'print_gradients') and self.settings.print_gradients:
            norm_type = 2.0
            if isinstance(parameters, torch.Tensor):
                parameters = [parameters]
            parameters = [p for p in parameters if p.grad is not None]
            device = parameters[0].grad.device
            norms = [p.grad.detach().abs().max().to(device) for p in parameters]
            max = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))

            total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device)
                                                 for p in parameters]), norm_type)
            print('Max gradient={}, norm gradient={}'.format(max, total_norm))
        
        return do_backprop

    def check_gradients(self, iteration: int, data_dict: Dict[str, Any], output_dict: Dict[str, Any], 
                        result_dict: Dict[str, Any]):
        if not self.run_grad_check:
            return
        if not self.check_invalid_gradients():
            self.logger.error('Iter: {}, invalid gradients.'.format(iteration))
            torch.save(data_dict, 'data.pth')
            torch.save(self.net, 'model.pth')
            self.logger.error('Data_dict and model snapshot saved.')
            # ipdb.set_trace()
        return 

    @torch.no_grad()
    def inference(self, suffix: str=''):
        """Run validation. """
        self.set_eval_mode()
        self.before_val()
        summary_board = SummaryBoard(adaptive=True)
        timer = Timer()
        plotting_dict_total = {}
        # pbar = tqdm.tqdm(enumerate(self.val_loader), total=total_iterations)
        for iteration, data_dict in enumerate(self.val_loader):
            self.inner_iteration = iteration + 1
            data_dict = self.to_cuda(data_dict)
            self.before_val_step(self.inner_iteration, data_dict)

            timer.add_prepare_time()
            output_dict, result_dict, plotting_dict = self.val_step(iteration, data_dict)
            timer.add_process_time()
            
            self.after_val_step(iteration, data_dict, output_dict, result_dict)
            result_dict = self.release_tensors(result_dict)
            plotting_dict_total.update(plotting_dict)  # store the plotting at multiple iterations
            summary_board.update_from_result_dict(result_dict)
            torch.cuda.empty_cache()

        self.after_val()
        summary_dict = summary_board.summary()
        message = '[Val{}] '.format(suffix) + get_log_string(summary_dict, iteration=self.iteration, timer=timer)
        self.current_best_val = summary_dict['best_value']
        self.logger.critical(message)
        self.write_event('val' + suffix, summary_dict, self.iteration)
        self.write_image('val' + suffix, plotting_dict_total, self.iteration)
        self.set_train_mode()
        self.writer.flush()
        return 

    def run(self, load_latest: bool=False, make_validation_first: bool=False):
        assert self.train_loader is not None
        assert self.val_loader is not None

        if self.settings.resume_snapshot is not None:
            self.load_snapshot(self.settings.resume_snapshot)
        # self.set_train_mode() integrated in self.train_epoch()

        self.just_started = True
        if load_latest:
            self.load_snapshot()

        if make_validation_first and self.epoch == 0:
            self.inference()
        self.set_train_mode()

        self.summary_board.reset_all()
        self.timer.reset()

        train_loader = CycleLoader(self.train_loader, self.epoch, self.distributed)
        self.before_train()
        self.optimizer.zero_grad()
        while self.iteration < self.max_iteration:
            self.iteration += 1
            data_dict = next(train_loader)
            data_dict = self.to_cuda(data_dict)
            self.before_train_step(self.iteration, data_dict)
            self.timer.add_prepare_time()
            # forward
            output_dict, result_dict, plotting_dict = self.train_step(self.iteration, data_dict)
            # backward & optimization
            result_dict['loss'].backward()
            
            do_backprop = self.after_backward(self.iteration, data_dict, output_dict, result_dict)
            if self.settings.distributed:
                do_backprop = torch.tensor(do_backprop).float().to(self.device)
                torch.distributed.all_reduce(
                        do_backprop, torch.distributed.ReduceOp.PRODUCT)
                do_backprop = do_backprop > 0
            self.check_gradients(self.iteration, data_dict, output_dict, result_dict)
            
            self.optimizer_step(self.iteration, do_backprop=do_backprop)

            # after training
            self.timer.add_process_time()
            self.after_train_step(self.iteration, data_dict, output_dict, result_dict)
            result_dict = self.release_tensors(result_dict)

            # all_no_processing_of_pts tensors are detached (we do not keep track of gradients anymore, and put to numpy

            self.summary_board.update_from_result_dict(result_dict)
            self.write_image('train', plotting_dict, self.iteration)
            
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
            
            # scheduler, here after each step
            if self.scheduler is not None and self.iteration % self.grad_acc_steps == 0:
                self.scheduler.step()
            torch.cuda.empty_cache()
            
            # snapshot & validation, every certain number of iterations
            if self.iteration % self.snapshot_steps == 0:
                # run validation
                self.inference()

                # snapshot saving and so on
                self.epoch = train_loader.last_epoch  # here that corresponds to the number of iterations, correct
                # save best checkpoint
                if self.current_best_val is not None and self.current_best_val < self.best_val:
                    message = 'VALIDATION IMPROVED ! From best value = {} at iteration {} to best value = {} ' \
                              'at current iteration {}'.format(self.best_val, self.epoch_of_best_val,
                                                               self.current_best_val, self.iteration)
                    self.logger.critical(message)
                    self.best_val = self.current_best_val  # update best_val
                    self.epoch_of_best_val = self.iteration  # update_epoch_of_best_val

                    self.save_snapshot('model_best.pth.tar', training_info=False)

                self.save_snapshot(f'iter-{self.iteration}.pth.tar')
                self.delete_old_checkpoints()  # keep only the most recent set of checkpoints


        self.after_train()
        message = 'Training finished.'
        self.logger.critical(message)
        return 
    
    @torch.no_grad()
    def inference_debug(self, suffix: str=''):
        self.set_eval_mode()
        self.before_val()
        summary_board = SummaryBoard(adaptive=True)
        plotting_dict_total = {}
        timer = Timer()
        cycle_loader = CycleLoader(self.val_loader, 0, False)
        for iteration in range(2):
            data_dict = next(cycle_loader)
            self.inner_iteration = iteration + 1
            data_dict = self.to_cuda(data_dict)
            self.before_val_step(iteration, data_dict)
            timer.add_prepare_time()
            output_dict, result_dict, plotting_dict = self.val_step(iteration, data_dict)
            timer.add_process_time()

            self.after_val_step(self.inner_iteration, data_dict, output_dict, result_dict)
            result_dict = self.release_tensors(result_dict)
            plotting_dict_total.update(plotting_dict)
            summary_board.update_from_result_dict(result_dict)

            '''
            message = get_log_string(
                result_dict=summary_board.summary(),
                iteration=self.inner_iteration,
                max_iteration=total_iterations,
                timer=timer,
            )
            pbar.set_description(message)
            '''
            torch.cuda.empty_cache()


        self.after_val()
        summary_dict = summary_board.summary()
        message = '[Val{}] '.format(suffix) + get_log_string(summary_dict, iteration=self.iteration, timer=timer)
        self.current_best_val = summary_dict['best_value']
        self.logger.critical(message)
        self.write_event('val' + suffix, summary_dict, self.iteration)
        self.write_image('val' + suffix, plotting_dict_total, self.iteration)
        self.set_train_mode()
        return 

    # debugging
    def run_debug(self, load_latest: bool=False, make_validation_first: bool=False):
        assert self.train_loader is not None
        assert self.val_loader is not None

        if self.settings.resume_snapshot is not None:
            self.load_snapshot(self.settings.resume_snapshot)
        # self.set_train_mode() integrated in self.train_epoch()

        self.just_started = True
        if load_latest:
            self.load_snapshot()

        if make_validation_first and self.epoch == 0:
            self.inference_debug()
        self.set_train_mode()

        self.summary_board.reset_all()
        self.timer.reset()

        train_loader = CycleLoader(self.train_loader, self.epoch, self.distributed)
        self.before_train()
        self.optimizer.zero_grad()
        while self.iteration < 10:
            self.iteration += 1
            data_dict = next(train_loader)
            data_dict = self.to_cuda(data_dict)
            self.before_train_step(self.iteration, data_dict)
            self.timer.add_prepare_time()
            # forward
            output_dict, result_dict, plotting_dict = self.train_step(self.iteration, data_dict)
            # backward & optimization
            result_dict['loss'].backward()
            do_backprop = self.after_backward(self.iteration, data_dict, output_dict, result_dict)
            self.check_gradients(self.iteration, data_dict, output_dict, result_dict)
            self.optimizer_step(self.iteration, do_backprop=do_backprop)

            # after training
            self.timer.add_process_time()
            self.after_train_step(self.iteration, data_dict, output_dict, result_dict)
            
            result_dict = self.release_tensors(result_dict)
            # all_no_processing_of_pts tensors are detached (we do not keep track of gradients anymore, and put to numpy
            self.summary_board.update_from_result_dict(result_dict)
            self.write_image('train', plotting_dict, self.iteration)
            
            # logging at every step here
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
            
            # scheduler, here after each step
            if self.scheduler is not None and self.iteration % self.grad_acc_steps == 0:
                self.scheduler.step()
            torch.cuda.empty_cache()
            
            # snapshot & validation, every certain number of iterations
            if self.iteration % 5 == 0:
                # run validation
                self.inference_debug()

                # snapshot saving and so on
                self.epoch = train_loader.last_epoch  # here that corresponds to the number of iterations, correct
                print(self.epoch)
                # save best checkpoint
                if self.current_best_val is not None and self.current_best_val < self.best_val:
                    message = 'VALIDATION IMPROVED ! From best value = {} at iteration {} to best value = {} ' \
                              'at current iteration {}'.format(self.best_val, self.epoch_of_best_val,
                                                               self.current_best_val, self.iteration)
                    self.logger.critical(message)
                    self.best_val = self.current_best_val  # update best_val
                    self.epoch_of_best_val = self.iteration  # update_epoch_of_best_val

                    self.save_snapshot('model_best.pth.tar', training_info=False)

                self.save_snapshot(f'iter-{self.iteration}.pth.tar')
                self.delete_old_checkpoints()  # keep only the most recent set of checkpoints

        self.after_train()
        message = 'Training finished.'
        self.logger.critical(message)
        return 
    
