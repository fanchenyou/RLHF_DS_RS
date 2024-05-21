import os
import time
import torch
import random
import numpy as np
import torch.nn as nn
from diffusion_models.model_led_initializer import LEDInitializer as InitializationModel, LEDInitializerWithValueHead
from diffusion_models.model_diffusion import TransformerDenoisingModel as CoreDenoisingModel
from diffusion_models.config import Config

import setproctitle
from trl import PPOConfig
from trl.ppo_base import PPOTrainer
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple
from tools.metrics import *
from tools.loss import l2loss_new


NUM_Tau = 5


class Trainer:
    def __init__(self, dset, hist_len, fut_len, device, train_loader, test_loader, learning_rate, checkpoint_dir,
                 mode='train', model_path=None, args=None):

        self.hist_len = hist_len
        self.fut_len = fut_len
        self.device = device
        self.cfg = Config(checkpoint_dir)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.mode = mode
        self.dset = dset

        self.l2_wt = 1e-4
        if dset == 'eth':
            self.l2_wt *= 0.2
        self.use_ip_rl = args.use_ip_rl

        # data normalization parameters
        self.traj_mean = torch.FloatTensor(self.cfg.traj_mean).to(device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        self.traj_scale = self.cfg.traj_scale

        # ------------------------- define diffusion parameters -------------------------
        self.n_steps = 100  # define total diffusion steps

        # make beta schedule and calculate the parameters used in denoising process.
        self.betas = self.make_beta_schedule(schedule='linear', n_timesteps=self.n_steps, start=1e-4, end=5e-2).to(
            device)
        self.alphas = 1 - self.betas
        self.alphas_prod = torch.cumprod(self.alphas, 0)
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_prod)

        # ------------------------- define models -------------------------
        self.model = CoreDenoisingModel(input_dim=self.hist_len * 6).to(device)

        self.social_dist_sigma = args.social_dist_sigma
        if dset == 'nba':
            self.social_dist_sigma /= 5
            print(self.social_dist_sigma, '--nba social dist')

        # use reject sampling
        self.reject_sampling = args.reject_sampling
        self.reject_sampling_all = args.use_rs_all
        self.test_diff_score = args.diff_score

        if self.mode == 'train_rl':
            self.w_rl = args.w_rl

            # assert model_path is not None
            self.model_initializer = LEDInitializerWithValueHead(t_obs=self.hist_len, s=self.fut_len * 2, n=20).to(
                device)
            self.ref_model_initializer = InitializationModel(t_obs=self.hist_len, s=self.fut_len * 2, n=20).to(device)
            self.ppo_config = self.get_ppo_config()
            self.ppo_trainer = PPOTrainer(self.ppo_config)

            if model_path is not None:
                print('pretrain from %s' % (model_path,))
                model_cp = torch.load(model_path, map_location='cpu')
                self.model_initializer.load_state_dict(model_cp['model_initializer_dict'], strict=False)
                self.ref_model_initializer.load_state_dict(model_cp['model_initializer_dict'], strict=False)
                assert 'diffusion_model_dict' in model_cp
                self.model.load_state_dict(model_cp['diffusion_model_dict'])
                print('load from model checkpoint')

            self.opt = torch.optim.Adam(self.model_initializer.parameters(), lr=learning_rate)
            # if args.use_lrschd:
            args.use_lrschd = True
            # self.scheduler_model = torch.optim.lr_scheduler.MultiStepLR(self.opt, milestones=[30, 60], gamma=0.8)
            # self.scheduler_model = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=100, eta_min=1e-4,
            #                                                                   verbose=True)
            train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=100,
                                                                         eta_min=3e-5, verbose=True)
            number_warmup_epochs = 6

            def warmup(current_step: int):
                return 1 / (1.2 ** (float(number_warmup_epochs - current_step)))

            warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda=warmup)
            self.scheduler_model = torch.optim.lr_scheduler.SequentialLR(self.opt,
                                                                         [warmup_scheduler, train_scheduler],
                                                                         [number_warmup_epochs])

            self.print_model_param(self.model, name='Core Denoising Model')
            self.print_model_param(self.model_initializer, name='Initialization Model with Value Head')

            # temporal re-weight in the loss, it is not necessary.
            # t2 = self.fut_len + 1  # 12 + 1
            # t1 = self.hist_len  # 8
            # tmp = [t2 - i for i in range(1, t2)]
            # self.temporal_reweight = torch.FloatTensor(tmp).to(device).unsqueeze(0).unsqueeze(0) / t1
            # print(self.temporal_reweight.size())
            # print(self.temporal_reweight)
            self.temporal_reweight = torch.ones(self.fut_len).to(device).unsqueeze(0).unsqueeze(0)

        elif self.mode == 'train':
            self.model_initializer = InitializationModel(t_obs=self.hist_len, s=self.fut_len * 2, n=20).to(device)

            init_model = False
            if model_path is not None:
                print('pretrain from %s' % (model_path,))
                model_cp = torch.load(model_path, map_location='cpu')
                self.model_initializer.load_state_dict(model_cp['model_initializer_dict'], strict=False)
                if 'diffusion_model_dict' in model_cp:
                    self.model.load_state_dict(model_cp['diffusion_model_dict'])
                    print('load from model checkpoint')
                    # init_model = True
                    # tmp = {}
                    # tmp['diffusion_model_dict']=model_cp['diffusion_model_dict']
                    # torch.save(tmp, os.path.join('result_diffusion', 'pre_%s/val_best.pth' % (dset,)))
                    # exit()

            if not init_model:
                # load pretrained models
                pre_diffusion_path = os.path.join('model_save', 'pre_%s/val_best.pth' % (dset,))
                model_cp = torch.load(pre_diffusion_path, map_location='cpu')
                self.model.load_state_dict(model_cp['diffusion_model_dict'])
                print('Load pre_diffusion %s' % (pre_diffusion_path,))

            self.opt = torch.optim.Adam(self.model_initializer.parameters(), lr=learning_rate)
            # self.scheduler_model = torch.optim.lr_scheduler.StepLR(self.opt, step_size=self.cfg.decay_step,
            #                                                        gamma=self.cfg.decay_gamma)
            # self.scheduler_model = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=100,
            #                                                                   eta_min=2e-4, verbose=True)

            train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=100,
                                                                         eta_min=2e-4, verbose=True)
            number_warmup_epochs = 3

            def warmup(current_step: int):
                return 1 / (2 ** (float(number_warmup_epochs - current_step)))

            warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda=warmup)
            self.scheduler_model = torch.optim.lr_scheduler.SequentialLR(self.opt,
                                                                         [warmup_scheduler, train_scheduler],
                                                                         [number_warmup_epochs])

            self.print_model_param(self.model, name='Core Denoising Model')
            self.print_model_param(self.model_initializer, name='Initialization Model')
            print(self.scheduler_model)

            if not init_model:
                t2 = self.fut_len + 1  # 12 + 1
                t1 = self.hist_len  # 8
                tmp = [t2 - i for i in range(1, t2)]
                self.temporal_reweight = torch.FloatTensor(tmp).to(device).unsqueeze(0).unsqueeze(0) / t1
            else:
                # temporal re-weight in the loss not necessary.
                self.temporal_reweight = torch.ones(self.fut_len).to(device).unsqueeze(0).unsqueeze(0)

        elif self.mode == 'pre':
            self.diff_opt = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
            self.print_model_param(self.model, name='Core Denoising Model')
            if model_path is not None:
                model_cp = torch.load(model_path, map_location='cpu')
                print('pretrain from %s' % (model_path,))
                self.model.load_state_dict(model_cp['diffusion_model_dict'])
        elif self.mode == 'pre_test':
            if model_path is not None:
                model_cp = torch.load(model_path, map_location='cpu')
                print('pretrain from %s' % (model_path,))
                self.model.load_state_dict(model_cp['diffusion_model_dict'])
        elif self.mode == 'test':

            assert model_path is not None, 'not given a trained model'
            self.model_initializer = InitializationModel(t_obs=self.hist_len, s=self.fut_len * 2, n=20).to(device)

            model_cp = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(model_cp['diffusion_model_dict'])
            # maybe with ValueHead
            self.model_initializer.load_state_dict(model_cp['model_initializer_dict'], strict=False)
            print('Load model %s' % (model_path,))
        else:
            assert 1==2

    def print_model_param(self, model: nn.Module, name: str = 'Model') -> None:
        '''
        Count the trainable/total parameters in `model`.
        '''
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("[{}] Trainable/Total: {}/{}".format(name, trainable_num, total_num))
        return None

    def make_beta_schedule(self, schedule: str = 'linear',
                           n_timesteps: int = 1000,
                           start: float = 1e-5, end: float = 1e-2) -> torch.Tensor:
        '''
        Make beta schedule.

        Parameters
        ----
        schedule: ['linear', 'quad', 'sigmoid'],
        n_timesteps: diffusion steps,
        start: beta start, `start<end`,
        end: beta end,

        Returns
        ----
        betas: Tensor with the shape of (n_timesteps)

        '''
        if schedule == 'linear':
            betas = torch.linspace(start, end, n_timesteps)
        elif schedule == "quad":
            betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
        elif schedule == "sigmoid":
            betas = torch.linspace(-6, 6, n_timesteps)
            betas = torch.sigmoid(betas) * (end - start) + start
        return betas

    def extract(self, input, t, x):
        shape = x.shape
        out = torch.gather(input, 0, t.to(input.device))
        reshape = [t.shape[0]] + [1] * (len(shape) - 1)
        return out.reshape(*reshape)

    def noise_estimation_loss(self, x, y_0, mask):
        batch_size = x.shape[0]
        # Select a random step for each example
        t = torch.randint(0, self.n_steps, size=(batch_size // 2 + 1,)).to(x.device)
        t = torch.cat([t, self.n_steps - t - 1], dim=0)[:batch_size]
        # x0 multiplier
        a = self.extract(self.alphas_bar_sqrt, t, y_0)
        beta = self.extract(self.betas, t, y_0)
        # eps multiplier
        am1 = self.extract(self.one_minus_alphas_bar_sqrt, t, y_0)
        e = torch.randn_like(y_0)
        # model input
        y = y_0 * a + e * am1
        output = self.model(y, beta, x, mask)
        # batch_size, 20, 2
        return (e - output).square().mean()

    def noise_estimation_loss_score(self, x, y_0, mask, sample_interval=3, use_average=True):

        batch_size = x.shape[0]
        # T is sampled denoising steps from 100 steps, so 100//3 = 34
        ts = torch.arange(0, self.n_steps, sample_interval).to(x.device)
        n_sample = ts.size(0)

        ts = ts.unsqueeze(0).repeat(batch_size, 1)
        # print(x.size(),y_0.size())

        ys, betas, xs, es, ms = [], [], [], [], []
        for n in range(n_sample):
            t = ts[:, n]
            a = self.extract(self.alphas_bar_sqrt, t, y_0)
            beta = self.extract(self.betas, t, y_0)
            betas.append(beta)
            # eps multiplier
            am1 = self.extract(self.one_minus_alphas_bar_sqrt, t, y_0)
            e = torch.randn_like(y_0)
            # model input
            y = y_0 * a + e * am1
            ys.append(y)
            es.append(e)
            ms.append(mask)
            # print(mask.size())

        # print(len(ys))
        ys = torch.cat(ys, dim=0)
        betas = torch.cat(betas, dim=0)
        es = torch.cat(es, dim=0)
        ms = torch.block_diag(*ms)
        # print(ms)

        with torch.no_grad():
            output = self.model(ys, betas, x.repeat(n_sample, 1, 1), ms)

        l_all = (es - output).square()  # .mean().item()

        if use_average:
            # scalar score for the entire scene
            return l_all.mean().item(), l_all[:, -1, :].mean().item()

        # # [N_pred, N_sample=34, T_fut=20, xy=2]
        l_all = l_all.view(mask.size(0), n_sample, ys.size(1), ys.size(2))

        # [N_pred, 20]
        return l_all.mean(1).mean(-1)

    def noise_estimation_loss_score_vis(self, x, y_0, mask, sample_interval=3, use_average=True):

        batch_size = x.shape[0]
        # T is sampled denoising steps from 100 steps, so 100//3 = 34
        ts = torch.arange(0, self.n_steps, sample_interval).to(x.device)
        n_sample = ts.size(0)

        ts = ts.unsqueeze(0).repeat(batch_size, 1)

        ys, betas, xs, es, ms = [], [], [], [], []
        for n in range(n_sample):
            t = ts[:, n]
            a = self.extract(self.alphas_bar_sqrt, t, y_0)
            beta = self.extract(self.betas, t, y_0)
            betas.append(beta)
            # eps multiplier
            am1 = self.extract(self.one_minus_alphas_bar_sqrt, t, y_0)
            e = torch.randn_like(y_0)
            # model input
            y = y_0 * a + e * am1
            ys.append(y)
            es.append(e)
            ms.append(mask)
            # print(mask.size())

        # print(len(ys))
        ys = torch.cat(ys, dim=0)
        betas = torch.cat(betas, dim=0)
        es = torch.cat(es, dim=0)
        ms = torch.block_diag(*ms)
        # print(ms)

        with torch.no_grad():
            output = self.model(ys, betas, x.repeat(n_sample, 1, 1), ms)

        l_all = (es - output).square()  # .mean().item()
        ll = l_all.mean(0).mean(-1)

        # [T_f]
        return l_all.mean().item(), l_all[:, -1, :].mean().item(), ll

    def p_sample(self, x, mask, cur_y, t):
        # if t == 0:
        #     z = torch.zeros_like(cur_y).to(x.device)
        # else:
        #     z = torch.randn_like(cur_y).to(x.device)
        t = torch.tensor([t]).to(x.device)
        # Factor to the model output
        eps_factor = (
                (1 - self.extract(self.alphas, t, cur_y)) / self.extract(self.one_minus_alphas_bar_sqrt, t, cur_y))
        # Model output
        beta = self.extract(self.betas, t.repeat(x.shape[0]), cur_y)
        eps_theta = self.model(cur_y, beta, x, mask)
        mean = (1 / self.extract(self.alphas, t, cur_y).sqrt()) * (cur_y - (eps_factor * eps_theta))
        # Generate z
        z = torch.randn_like(cur_y).to(x.device)
        # Fixed sigma
        sigma_t = self.extract(self.betas, t, cur_y).sqrt()
        sample = mean + sigma_t * z
        return sample

    def p_sample_accelerate(self, x, mask, cur_y, t):
        # if t == 0:
        #     z = torch.zeros_like(cur_y).to(x.device)
        # else:
        #     z = torch.randn_like(cur_y).to(x.device)
        t = torch.tensor([t]).to(x.device)
        # Factor to the model output
        eps_factor = (
                (1 - self.extract(self.alphas, t, cur_y)) / self.extract(self.one_minus_alphas_bar_sqrt, t, cur_y))
        # Model output
        beta = self.extract(self.betas, t.repeat(x.shape[0]), cur_y)

        # print(cur_y.size(), beta.size(), x.size(), mask.size(), '-----4')
        eps_theta = self.model.generate_accelerate(cur_y, beta, x, mask)

        mean = (1 / self.extract(self.alphas, t, cur_y).sqrt()) * (cur_y - (eps_factor * eps_theta))
        # Generate z
        z = torch.randn_like(cur_y).to(x.device)
        # Fixed sigma
        sigma_t = self.extract(self.betas, t, cur_y).sqrt()
        sample = mean + sigma_t * z * 0.00001
        return sample

    def p_sample_loop(self, x, mask, shape):
        self.model.eval()
        prediction_total = torch.Tensor().to(x.device)
        for _ in range(20):
            cur_y = torch.randn(shape).to(x.device)
            for i in reversed(range(self.n_steps)):
                cur_y = self.p_sample(x, mask, cur_y, i)
            prediction_total = torch.cat((prediction_total, cur_y.unsqueeze(1)), dim=1)
        return prediction_total

    def p_sample_loop_mean(self, x, mask, loc):
        prediction_total = torch.Tensor().to(x.device)
        for loc_i in range(1):
            cur_y = loc
            for i in reversed(range(NUM_Tau)):
                cur_y = self.p_sample(x, mask, cur_y, i)
            prediction_total = torch.cat((prediction_total, cur_y.unsqueeze(1)), dim=1)
        return prediction_total

    def p_sample_loop_accelerate(self, x, mask, loc):
        '''
        Batch operation to accelerate the denoising process.

        x: NBA [11, 10, 6] => ETH should be [N, 8, 6]
        mask: [11, 11]
        cur_y: [11, 10, 20, 2]
        '''

        # loc [11, 20, 20, 2]
        # print(loc.size(), x.size(), mask.size(),'----')

        # the first 10 samples, denoise from Step 10, 9, 8,..., 0
        cur_y = loc[:, :10]
        # print(cur_y.size(), '---2', x.size(), mask.size())
        for i in reversed(range(NUM_Tau)):
            cur_y = self.p_sample_accelerate(x, mask, cur_y, i)

        # print(cur_y.size(), '---3', loc.size(), NUM_Tau)
        cur_y_ = loc[:, 10:]
        for i in reversed(range(NUM_Tau)):
            cur_y_ = self.p_sample_accelerate(x, mask, cur_y_, i)

        # shape: B=b*n, K=10, T, 2
        prediction_total = torch.cat((cur_y_, cur_y), dim=1)
        return prediction_total

    def fit(self, eval_epoch=5):
        constant_metrics = {'min_val_epoch': -1, 'min_val_ade': 1e6, 'min_val_fde': 1e6}
        self.l2_loss = l2loss_new()

        # Training loop
        for epoch in range(0, 300):
            loss_total, loss_distance, loss_uncertainty, loss_l2 = self._train_single_epoch(epoch)
            print(
                '[{}] Epoch: {}\t\tLr:{:.4f}\tLoss: {:.6f}\tLoss Dist.: {:.6f}\tLoss Uncertainty: {:.6f}\tLoss L2: {:.6f}'.format(
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    epoch, self.opt.param_groups[0]['lr'], loss_total, loss_distance, loss_uncertainty, loss_l2))

            if (epoch + 1) % eval_epoch == 0:
                ade, fde = self._test_single_epoch()
                print('Epoch %d, ADE: %.4f, FDE: %.4f' % (epoch, ade, fde))

                if constant_metrics['min_val_ade'] > ade:
                    constant_metrics['min_val_ade'] = ade
                    constant_metrics['min_val_fde'] = fde
                    constant_metrics['min_val_epoch'] = epoch
                    model_cp = {'model_initializer_dict': self.model_initializer.state_dict(),
                                'diffusion_model_dict': self.model.state_dict()}

                    torch.save(model_cp, os.path.join(self.cfg.model_dir, 'val_best.pth'))
                    torch.save(model_cp, os.path.join(self.cfg.model_dir,
                                                      'val_best_epoch_%03d_ade_%.3f_fde_%.3f.pth' % (epoch, ade, fde)))
                    print(
                        '****** Found best model at epoch %d with min_val_ade %.3f / %.3f' % (epoch, ade, fde))
                    print('Save to %s' % (self.cfg.model_dir,))

                else:
                    print('----- Current ade, fde %.3f / %.3f' % (ade, fde))
                    print('----- Min   fde %.3f / %.3f  in epoch %d, ' % (
                        constant_metrics['min_val_ade'], constant_metrics['min_val_fde'],
                        constant_metrics['min_val_epoch']))
                    print('Save folder %s' % (self.cfg.model_dir,))
                    if (epoch + 1) % (10 * eval_epoch) == 0:
                        model_cp = {'model_initializer_dict': self.model_initializer.state_dict(),
                                    'diffusion_model_dict': self.model.state_dict()}
                        torch.save(model_cp, os.path.join(self.cfg.model_dir,
                                                          'val_save_epoch_%03d_ade_%.3f_fde_%.3f.pth' % (
                                                              epoch, ade, fde)))

            self.scheduler_model.step()

    def fit_rl(self, eval_epoch=1):
        constant_metrics = {'min_val_epoch': -1, 'min_val_ade': 1e6, 'min_val_fde': 1e6}
        self.l2_loss = l2loss_new()

        state_str = 'Train diffusion ppo'
        setproctitle.setproctitle(state_str)

        # self.writer = SummaryWriter(log_dir=self.cfg.model_dir)
        self.global_iter_count = 0

        # Training loop
        for epoch in range(0, 199):
            # loss_total, loss_distance, loss_uncertainty = self._train_rl_single_epoch(epoch)
            # print('Epoch: {}\t\tLoss: {:.6f}\tLoss Dist.: {:.6f}\tLoss Uncertainty: {:.6f}'.format(
            #     epoch, loss_total, loss_distance, loss_uncertainty))
            self._train_rl_single_epoch(epoch)

            if epoch % eval_epoch == 0:
                ade, fde = self._test_single_epoch()
                print('Epoch %d, ADE: %.4f, FDE: %.4f' % (epoch, ade, fde))

                if constant_metrics['min_val_ade'] > ade:
                    constant_metrics['min_val_ade'] = ade
                    constant_metrics['min_val_fde'] = fde
                    constant_metrics['min_val_epoch'] = epoch
                    model_cp = {'model_initializer_dict': self.model_initializer.state_dict(),
                                'diffusion_model_dict': self.model.state_dict()}

                    torch.save(model_cp, os.path.join(self.cfg.model_dir, 'val_best.pth'))
                    torch.save(model_cp, os.path.join(self.cfg.model_dir,
                                                      'val_best_epoch_%03d_lr_%.5f_ade_%.3f_fde_%.3f.pth'
                                                      % (epoch, self.opt.param_groups[0]['lr'], ade, fde)))
                    print(
                        '****** Found best model at epoch %d with min_val_ade %.3f / %.3f' % (epoch, ade, fde))
                    print('Save to %s' % (self.cfg.model_dir,))
                else:
                    print('----- Current ade, fde %.3f / %.3f' % (ade, fde))
                    print('----- Min   fde %.3f / %.3f  in epoch %d, ' % (
                        constant_metrics['min_val_ade'], constant_metrics['min_val_fde'],
                        constant_metrics['min_val_epoch']))
                    print('Save folder %s' % (self.cfg.model_dir,))
                    # if (epoch + 1) % (10 * eval_epoch) == 0:
                    if epoch % (3 * eval_epoch) == 0:
                        model_cp = {'model_initializer_dict': self.model_initializer.state_dict(),
                                    'diffusion_model_dict': self.model.state_dict()}
                        torch.save(model_cp, os.path.join(self.cfg.model_dir,
                                                          'val_epoch_%03d_lr_%.5f_ade_%.3f_fde_%.3f.pth' % (
                                                              epoch, self.opt.param_groups[0]['lr'], ade, fde)))

            self.scheduler_model.step()

    def pre(self, eval_epoch=20):
        constant_metrics = {'min_val_epoch': -1, 'min_val_noise_loss': 1e6}
        pre_epoch = 2000
        # Training loop
        for epoch in range(0, pre_epoch):
            loss_total = self._pretrain_diffusion_single_epoch(epoch)
            print('[{}] Epoch: {}/{}\t\tNoise Loss: {:.6f}'.format(
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                epoch, pre_epoch, loss_total))

            if (epoch + 1) % eval_epoch == 0:
                if constant_metrics['min_val_noise_loss'] > loss_total:
                    constant_metrics['min_val_noise_loss'] = loss_total
                    constant_metrics['min_val_epoch'] = epoch
                    model_cp = {'diffusion_model_dict': self.model.state_dict()}
                    torch.save(model_cp, os.path.join(self.cfg.model_dir, 'val_best.pth'))
                    print(
                        '****** Found best model at epoch %d with min_val_ade %.6f' % (epoch, loss_total))
                    print('Save to %s' % (self.cfg.model_dir,))

    def pre_test(self, eval_epoch=20):
        constant_metrics = {'min_val_epoch': -1, 'min_val_noise_loss': 1e6}

        self.model.train()
        loss_total, count = 0, 0

        for j, data in enumerate(self.test_loader):
            batch_size, traj_mask, past_traj, fut_traj = self.data_preprocess(data)

            noise_est_loss = self.noise_estimation_loss_score(past_traj, fut_traj, traj_mask)

            loss_total += noise_est_loss

            print(noise_est_loss)

            count += 1

        return loss_total / count

        # # Training loop
        # for epoch in range(0, pre_epoch):
        #
        #     print('[{}] Epoch: {}/{}\t\tNoise Loss: {:.6f}'.format(
        #         time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        #         epoch, pre_epoch, loss_total))
        #
        #     if (epoch + 1) % eval_epoch == 0:
        #         if constant_metrics['min_val_noise_loss'] > loss_total:
        #             constant_metrics['min_val_noise_loss'] = loss_total
        #             constant_metrics['min_val_epoch'] = epoch
        #             model_cp = {'diffusion_model_dict': self.model.state_dict()}
        #             torch.save(model_cp, os.path.join(self.cfg.model_dir, 'val_best.pth'))
        #             print(
        #                 '****** Found best model at epoch %d with min_val_ade %.6f' % (epoch, loss_total))
        #             print('Save to %s' % (self.cfg.model_dir,))

    def data_preprocess(self, data, use_rl_data=False):
        """
            pre_motion_3D: torch.Size([32, 11, 10, 2]), [batch_size, num_agent, past_frame, dimension]
            fut_motion_3D: torch.Size([32, 11, 20, 2])
            fut_motion_mask: torch.Size([32, 11, 20])
            pre_motion_mask: torch.Size([32, 11, 10])
            traj_scale: 1
            pred_mask: None
            seq: nba
        """
        # torch.Size([1, 53, 2, 8]) torch.Size([1, 53, 2, 12])
        obs_traj, pred_traj_gt = data[0:2]

        batch_size = obs_traj.shape[0]  # =1
        N_obj = obs_traj.size(1)

        obs_traj = obs_traj.to(self.device)
        pred_traj_gt = pred_traj_gt.to(self.device)

        data_pre_motion_3D = obs_traj.permute([0, 1, 3, 2])  # [1, N_obj, 8, 2] [1, N_obj, 12, 2]
        data_fut_motion_3D = pred_traj_gt.permute([0, 1, 3, 2])

        traj_mask = torch.ones(batch_size * N_obj, batch_size * N_obj).to(self.device)
        # for i in range(batch_size):
        #     traj_mask[i * 11:(i + 1) * 11, i * 11:(i + 1) * 11] = 1.

        initial_pos = data_pre_motion_3D[:, :, -1:]
        # print(initial_pos.size(),'---')
        # augment input: absolute position, relative position, velocity
        # past_traj_abs = data_pre_motion_3D.view(-1, 8, 2)
        # past_traj_rel = pred_traj_gt.view(-1, 12, 2)
        # print(past_traj_rel.size())
        obs_len, pred_len = data_pre_motion_3D.size(2), data_fut_motion_3D.size(2)

        past_traj_abs = ((data_pre_motion_3D - self.traj_mean) / self.traj_scale).contiguous().view(-1, obs_len, 2)
        past_traj_rel = ((data_pre_motion_3D - initial_pos) / self.traj_scale).contiguous().view(-1, obs_len, 2)
        past_traj_vel = torch.cat(
            (past_traj_rel[:, 1:] - past_traj_rel[:, :-1], torch.zeros_like(past_traj_rel[:, -1:])), dim=1)
        past_traj = torch.cat((past_traj_abs, past_traj_rel, past_traj_vel), dim=-1)

        fut_traj = ((data_fut_motion_3D - initial_pos) / self.traj_scale).contiguous().view(-1, pred_len, 2)
        if not use_rl_data:
            return batch_size, traj_mask, past_traj, fut_traj

        # print(initial_pos[0,...], past_traj_abs[:,-1,:])
        # exit()
        V_tr = data[-1].squeeze().to(self.device)
        return batch_size, traj_mask, past_traj, fut_traj, obs_traj, V_tr

    def _train_single_epoch(self, epoch):

        self.model.train()
        self.model_initializer.train()
        loss_total, loss_dt, loss_dc, loss_l2_total, count, count_opt = 0, 0, 0, 0, 0, 0
        batch_size = 64
        self.opt.zero_grad()

        lt, lc, l2 = 0, 0, 0
        for j, data in enumerate(self.train_loader):
            # TODO: accumulate batch

            if j % 100 == 0:
                print('Epoch %d lr %.5f train %d / %d / %d' % (
                    epoch, self.opt.param_groups[0]['lr'], j, count_opt, len(self.train_loader)))

            bsz, traj_mask, past_traj, fut_traj = self.data_preprocess(data)
            # assert bsz == 1

            sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask)
            sample_prediction = torch.exp(variance_estimation / 2)[
                                    ..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(
                dim=(1, 2))[:, None, None, None]
            loc = sample_prediction + mean_estimation[:, None]

            # print(past_traj.size(), loc.size(), sample_prediction.size(), fut_traj.size(),'----11111')
            # sample_prediction : [N, 20, 12, 2]
            # fut_traj: [N, 12, 2]
            # p_sample_loop_accelerate :  [N, 8, 6] -> [N, 20, 12, 2]

            generated_y = self.p_sample_loop_accelerate(past_traj, traj_mask, loc)

            loss_l2 = self.l2_loss(self.model_initializer) * self.l2_wt

            loss_dist = self.loss_dist(generated_y, fut_traj)
            loss_uncertainty = self.loss_uncertainty(variance_estimation, generated_y, fut_traj, wt=1 / 50.0)
            # print(loss_dist.item(), loss_uncertainty.item(),'---')
            loss = (loss_dist + loss_uncertainty + loss_l2) / batch_size
            loss.backward()

            loss_total += loss.item()
            lt += loss_dist.item()
            lc += loss_uncertainty.item()
            l2 += loss_l2.item()

            if (j + 1) % batch_size == 0:
                # accumulate gradient
                torch.nn.utils.clip_grad_norm_(self.model_initializer.parameters(), 1.)
                self.opt.step()
                self.opt.zero_grad()
                count_opt += 1
                loss_dt += lt / batch_size
                loss_dc += lc / batch_size
                loss_l2_total += l2 / batch_size
                lt, lc, l2 = 0, 0, 0

            count += 1

        return loss_total / count, loss_dt / count, loss_dc / count, loss_l2_total / count

    # use hand-crafted IPScore
    def nodes_sample_rel_to_nodes_abs(self, nodes, init_node):
        nodes_ = np.zeros_like(nodes)
        for s in range(nodes.shape[1]):
            for ped in range(nodes.shape[2]):
                # [N_sample, 2] += [1, 2]
                nodes_[:, s, ped, :] = nodes[:, s, ped, :] + init_node[ped:ped + 1, :]

        return nodes_.squeeze()


    # use Diffusion as Score
    def get_score(self, past_traj, fut_traj, pred_traj, traj_mask):

        noise_gt_loss = self.noise_estimation_loss_score(past_traj, fut_traj, traj_mask, use_average=False)

        # print(past_traj.size(), fut_traj.size(), pred_traj.size())
        noise_preds = []
        for ns in range(pred_traj.size(1)):
            noise_pred_loss_i = self.noise_estimation_loss_score(past_traj, pred_traj[:, ns, ...],
                                                                 traj_mask, use_average=False)
            noise_preds.append(noise_pred_loss_i)

        # [N_sample, N_ped, T_fut]
        noise_preds_loss = torch.stack(noise_preds, dim=0)

        noise_gt_loss = noise_gt_loss.unsqueeze(0)
        noise_loss_diff = (noise_gt_loss - noise_preds_loss) / (noise_gt_loss + 1e-8)

        noise_loss_diff = torch.clamp(noise_loss_diff, min=0.0, max=1.0)
        # noise_loss_diff = noise_loss_diff.permute(0, 2, 1)
        # print(noise_loss_diff, noise_loss_diff.size())

        return noise_loss_diff

    def get_logprobs(self, sample_prediction, mean_estimation, variance_estimation):

        sample_prediction = torch.exp(variance_estimation / 2)[
                                ..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(
            dim=(1, 2))[:, None, None, None]
        loc = sample_prediction + mean_estimation[:, None]
        # print(mean_estimation.size(),'--',mean_estimation[:, None].size(),)
        # should be [N_obj, 20, fut_len, 2] -> [N_obj, fut_len, 2]
        # loc_mean = torch.mean(loc, dim=1)
        loc_var = torch.std(loc, dim=1, keepdim=True).repeat(1, loc.size(1), 1, 1)
        loc_var2 = torch.std(loc, dim=1, keepdim=False)  # .repeat(1,loc.size(1),1,1)

        pred_dist = torch.distributions.normal.Normal(loc, loc_var)
        pred_logprobs_xy = pred_dist.log_prob(loc)
        # print(loc_var.size(), pred_logprobs_xy.size())

        all_pred_logprobs = torch.sum(pred_logprobs_xy, dim=-1)
        all_pred_logprobs = all_pred_logprobs.permute(1, 0, 2)

        if 1 == 2:
            all_pred_logprobs2 = []
            for n_sample in range(loc.size(1)):
                # loc:[35, 20, 12, 2] mean:[35, 12, 2]  var: torch.Size([35, 1]

                pred_xy = loc[:, n_sample, ...]
                pred_dist = torch.distributions.normal.Normal(pred_xy, loc_var2)
                # print(pred_xy.size(), loc)

                # print(pred_xy.size(), mean_estimation.size(),'---', pred_var.size())
                pred_logprobs_xy = pred_dist.log_prob(pred_xy)
                pred_logprobs = torch.sum(pred_logprobs_xy, dim=-1)
                all_pred_logprobs2.append(pred_logprobs)

            # # print(pred_dist)
            # pred_logprobs_xy = pred_dist.log_prob(loc[:,0,:,:])
            # # [N_obj, fut_len]
            # pred_logprobs = torch.sum(pred_logprobs_xy, dim=-1)
            # # [20, N_obj, fut_len]
            # pred_logprobs = pred_logprobs.unsqueeze(0).repeat(sample_prediction.size(1), 1, 1)
            all_pred_logprobs2 = torch.stack(all_pred_logprobs2).float()
            # print(all_pred_logprobs2[0,0,:], all_pred_logprobs[0,0,:])

        return all_pred_logprobs, loc, pred_dist

    def loss_uncertainty(self, var_est, gen_y, fut_traj, wt=1.0):
        loss_u = (torch.exp(-var_est) * (gen_y - fut_traj.unsqueeze(dim=1)).norm(p=2, dim=-1).mean(
            dim=(1, 2)) + var_est).mean()
        return loss_u * wt

    def loss_dist(self, gen_y, fut_traj):
        loss_dist = \
            ((gen_y - fut_traj.unsqueeze(dim=1)).norm(p=2, dim=-1) * self.temporal_reweight).mean(dim=-1).min(dim=1)[
                0].mean()
        return loss_dist

    def _train_rl_single_epoch(self, epoch):

        self.model.train()
        self.model_initializer.train()
        loss_total, loss_dt, loss_dc, count, count_opt = 0, 0, 0, 0, 0
        batch_size = 32  # // 2
        accum_iter = batch_size
        N_sample = 20

        QtState = namedtuple('QtState',
                             ('logprobs', 'values', 'rewards', 'past_traj', 'fut_traj', 'v_tr', 'traj_mask', 'ind'))

        mini_batch_dict = []
        loss_batch = 0.0
        iter_count = 0
        train_count = 0
        all_logprobs_list = []
        all_ref_logprobs_list = []
        # writer = self.writer
        loader_len = len(self.train_loader)
        self.opt.zero_grad()
        num_minus_kl_div = 0

        for j, data in enumerate(self.train_loader):
            # TODO: accumulate batch

            if j % 100 == 0:
                print('Epoch %d lr %.5f train %d / %d / %d' % (
                    epoch, self.opt.param_groups[0]['lr'], j, count_opt, len(self.train_loader)))

            bsz, traj_mask, past_traj, fut_traj, obs_traj, V_tr = self.data_preprocess(data, use_rl_data=True)
            # assert bsz == 1
            # N_obj = past_traj.size(0)

            ##########################
            # Collect training batch #
            ##########################
            with torch.no_grad():
                self.model_initializer.eval()
                sp_pred, m_pred, var_pred, values = self.model_initializer(past_traj, traj_mask)
                sp_pred_ref, m_pred_ref, var_pred_ref = self.ref_model_initializer(past_traj, traj_mask)

                pred_logprobs, loc, _ = self.get_logprobs(sp_pred, m_pred, var_pred)
                ref_logprobs, _, _ = self.get_logprobs(sp_pred_ref, m_pred_ref, var_pred_ref)
                all_logprobs_list.append(pred_logprobs)
                all_ref_logprobs_list.append(ref_logprobs)

                ################################################
                ##   collect denoised generated trajectories  ##
                ################################################
                # values [N_obj, N_sample, fut_len=12]
                # print(past_traj.size(), loc.size(), sample_prediction.size(), fut_traj.size(),'----11111')
                # sample_prediction : [N, 20, 12, 2]
                # fut_traj: [N, 12, 2]
                # p_sample_loop_accelerate :  [N, 8, 6] -> [N, 20, 12, 2]
                pred_traj = self.p_sample_loop_accelerate(past_traj, traj_mask, loc)
                # [35, 20, 12, 2] , [1, 35, 2, 8] , [12, 35, 2])
                # print(generated_y.size(),'---',obs_traj.size(), '===', V_tr.size())

                # use diffusion score
                scores = self.get_score(past_traj, fut_traj, pred_traj, traj_mask)

                # [N_sample, N_obj, fut_len]
                rewards, _ = self.ppo_trainer.compute_rewards(scores, pred_logprobs, ref_logprobs)

                old_logprobs = pred_logprobs  # .detach().clone()
                old_values = values  # .detach().clone()

                ## if use reject sampling, leave one best path reward
                if self.reject_sampling or self.reject_sampling_all:
                    scores_avg = scores.mean(dim=[1, 2])
                    best_ind = scores_avg.max(0)[1].item()
                    rewards = rewards[best_ind:best_ind + 1, ...]
                    old_values = old_values[best_ind:best_ind + 1, ...]
                    old_logprobs = old_logprobs[best_ind:best_ind + 1, ...]

                    mini_batch_dict.append(
                        QtState(old_logprobs, old_values, rewards, past_traj, fut_traj, V_tr, traj_mask, best_ind))
                else:
                    mini_batch_dict.append(
                        QtState(old_logprobs, old_values, rewards, past_traj, fut_traj, V_tr, traj_mask, 0))

            iter_count += 1

            if iter_count % accum_iter == 0:
                self.model_initializer.train()
                # del sample_prediction, mean_estimation, variance_estimation, values
                mini_batch_size = accum_iter
                # assert len(mini_batch_dict) == accum_iter
                shuffle_index = torch.randperm(mini_batch_size)

                train_stats_all = []
                train_loss = torch.zeros(6).to(self.device)

                for pe in range(1):  # self.ppo_config.ppo_epochs):
                    for k in range(mini_batch_size):

                        qstate = mini_batch_dict[shuffle_index[k]]
                        old_logprobs, old_values, rewards, past_traj, fut_traj, V_tr, traj_mask, best_ind = qstate
                        # N_obj = past_traj.size(0)

                        sample_pred, mean_pred, var_pred, pred_values = self.model_initializer(past_traj, traj_mask)
                        pred_logprobs, loc, pred_dist = self.get_logprobs(sample_pred, mean_pred, var_pred)
                        generated_y = self.p_sample_loop_accelerate(past_traj, traj_mask, loc)

                        if self.reject_sampling or self.reject_sampling_all:
                            # print(old_logprobs.size(), pred_values.size(), pred_logprobs.size())
                            pred_values = pred_values[best_ind:best_ind + 1, ...]
                            pred_logprobs = pred_logprobs[best_ind:best_ind + 1, ...]

                        if self.reject_sampling_all:
                            # print(generated_y.size(),'---',var_pred.size(),'---')
                            generated_y = generated_y[:, best_ind:best_ind + 1, ...]

                        loss_p, loss_v, train_stats = self.ppo_trainer.loss_diffusion(old_logprobs, old_values, rewards,
                                                                                      pred_values,
                                                                                      pred_logprobs, ret_stat=False)
                        # print(generated_y.size(), fut_traj.size())
                        loss_l2 = self.l2_loss(self.model_initializer) * self.l2_wt
                        loss_dist = self.loss_dist(generated_y, fut_traj)
                        loss_uncertainty = self.loss_uncertainty(var_pred, generated_y, fut_traj, wt=1 / 50.0)
                        # print(loss_dist.item(), loss_uncertainty.item(), loss_p.item(), loss_v.item())
                        # loss_all = loss_dist + loss_uncertainty + self.w_rl * (loss_p + 0.2 * loss_v) + loss_l2
                        loss_all = loss_dist + 0.2 * loss_uncertainty + self.w_rl * (loss_p + 5.0 * loss_v) + loss_l2
                        loss_all_n = loss_all.item()
                        loss_all /= mini_batch_size
                        loss_all.backward()

                        train_loss[0] += loss_all_n
                        train_loss[1] += loss_p
                        train_loss[2] += loss_v
                        train_loss[3] += loss_dist
                        train_loss[4] += loss_uncertainty
                        train_loss[5] += loss_l2
                        # train_stats_all.append(train_stats)

                    # accumulate gradient
                    torch.nn.utils.clip_grad_norm_(self.model_initializer.parameters(), 1.)
                    self.opt.step()
                    self.opt.zero_grad()
                    count_opt += 1

                ####################################################
                ############ Update the KL control #################
                ####################################################
                # [12, N] - > [12, sum(N)]
                all_logprobs = torch.cat(all_logprobs_list, dim=1)  # .sum(0).mean()
                all_ref_logprobs = torch.cat(all_ref_logprobs_list, dim=1)
                kl_list = (all_logprobs - all_ref_logprobs).sum(0)
                mean_kl = kl_list.mean()

                # multiply the batch_size by the number of processes
                self.ppo_trainer.kl_ctl.update(mean_kl.item(), mini_batch_size)

                if mean_kl.item() < -1.0:
                    # warn users
                    # print(
                    #     f"KL divergence is starting to become negative: {mean_kl.item():.2f} - this might be a precursor for failed training."
                    #     " sometimes this happens because the generation kwargs are not correctly set. Please make sure"
                    #     " that the generation kwargs are set correctly, or review your training hyperparameters."
                    # )
                    num_minus_kl_div += 1

                train_loss /= mini_batch_size

                train_count += 1

                # TODO: make per step
                if iter_count % 10 == 0:
                    print(
                        '[Train] Epoch %d, %d/%d, train_cnt %d, lr %.5f, loss_total: %.3f, loss_p: %.3f, loss_v: %.3f, loss_dist %.3f, '
                        'loss_uncertainty %.5f, loss_l2: %.5f, minus_KL: %.3f' % (
                            epoch, iter_count, loader_len, train_count, self.opt.param_groups[0]['lr'],
                            train_loss[0].item(), train_loss[1].item(),
                            train_loss[2].item(), train_loss[3].item(), train_loss[4].item(), train_loss[5].item(),
                            num_minus_kl_div / train_count))

                # keep track of distance loss
                loss_batch += train_loss[3].item()
                # clear cache
                train_loss[:] = 0
                mini_batch_dict = []
                train_stats_all = []
                all_logprobs_list = []
                all_ref_logprobs_list = []
                # torch.cuda.empty_cache()
                self.global_iter_count += 1

            count += 1

        # writer.add_scalar('epoch/train_loss', loss_batch / train_count, epoch)

        # return loss_total / count, loss_dt / count, loss_dc / count
        return

    def _pretrain_diffusion_single_epoch(self, epoch):

        self.model.train()
        loss_total, count = 0, 0

        for j, data in enumerate(self.train_loader):
            batch_size, traj_mask, past_traj, fut_traj = self.data_preprocess(data)

            noise_est_loss = self.noise_estimation_loss(past_traj, fut_traj, traj_mask)

            loss_total += noise_est_loss.item()

            self.diff_opt.zero_grad()
            noise_est_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
            self.diff_opt.step()
            count += 1

        return loss_total / count

    def _test_single_epoch(self):
        performance = {'FDE': 0.0, 'ADE': 0.0}
        samples = 0
        assert self.traj_scale == 1

        def prepare_seed(rand_seed):
            np.random.seed(rand_seed)
            random.seed(rand_seed)
            torch.manual_seed(rand_seed)
            torch.cuda.manual_seed_all(rand_seed)

        prepare_seed(0)
        count = 0
        N_sample = 20
        with torch.no_grad():
            for data in self.test_loader:
                batch_size, traj_mask, past_traj, fut_traj = self.data_preprocess(data)

                sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask)[
                                                                          :3]
                sample_prediction = torch.exp(variance_estimation / 2)[
                                        ..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(
                    dim=(1, 2))[:, None, None, None]
                loc = sample_prediction + mean_estimation[:, None]

                # [2, 20, 12, 2]
                pred_traj = self.p_sample_loop_accelerate(past_traj, traj_mask, loc)

                fut_traj = fut_traj.unsqueeze(1)  # .repeat(1, N_sample, 1, 1)
                # b*n, N_sample, T=fut_len, 2

                # fut_traj last dim (x,y) -> l2 distance = [N, 20, 12]
                distances = torch.norm(fut_traj - pred_traj, dim=-1)

                ade = (distances[:, :, :]).mean(dim=-1).min(dim=-1)[0].sum()
                fde = (distances[:, :, -1]).min(dim=-1)[0].sum()
                performance['ADE'] += ade.item()
                performance['FDE'] += fde.item()

                samples += distances.shape[0]
                count += 1
            # if count==100:
            # 	break
        ade_all, fde_all = performance['ADE'] / samples, performance['FDE'] / samples
        return ade_all, fde_all

    def save_data(self):
        '''
        Save the visualization data.
        '''
        model_path = './results/checkpoints/led_vis.p'
        model_dict = torch.load(model_path, map_location=torch.device('cpu'))['model_initializer_dict']
        self.model_initializer.load_state_dict(model_dict)

        def prepare_seed(rand_seed):
            np.random.seed(rand_seed)
            random.seed(rand_seed)
            torch.manual_seed(rand_seed)
            torch.cuda.manual_seed_all(rand_seed)

        prepare_seed(0)
        root_path = './visualization/data/'

        with torch.no_grad():
            for data in self.test_loader:
                _, traj_mask, past_traj, _ = self.data_preprocess(data)

                sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask)
                torch.save(sample_prediction, root_path + 'p_var.pt')
                torch.save(mean_estimation, root_path + 'p_mean.pt')
                torch.save(variance_estimation, root_path + 'p_sigma.pt')

                sample_prediction = torch.exp(variance_estimation / 2)[
                                        ..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(
                    dim=(1, 2))[:, None, None, None]
                loc = sample_prediction + mean_estimation[:, None]

                pred_traj = self.p_sample_loop_accelerate(past_traj, traj_mask, loc)
                pred_mean = self.p_sample_loop_mean(past_traj, traj_mask, mean_estimation)

                torch.save(data['pre_motion_3D'], root_path + 'past.pt')
                torch.save(data['fut_motion_3D'], root_path + 'future.pt')
                torch.save(pred_traj, root_path + 'prediction.pt')
                torch.save(pred_mean, root_path + 'p_mean_denoise.pt')

                raise ValueError

    def test_single_model(self):

        performance = {'FDE': 0.0, 'ADE': 0.0}
        samples = 0

        # self.model.eval()

        def prepare_seed(rand_seed):
            np.random.seed(rand_seed)
            random.seed(rand_seed)
            torch.manual_seed(rand_seed)
            torch.cuda.manual_seed_all(rand_seed)

        prepare_seed(0)
        count = 0
        N_sample = 20
        int_pol_rt_bigls = []

        diff_gt_score = 0.0
        diff_gt_pred_ratio = 0.0
        diff_gt_pred_mean_ratio = 0.0
        diff_gt_pred_final_ratio = 0.0

        num_match_ade, num_match_fde = 0, 0
        num_match_ade_5, num_match_fde_5 = 0, 0
        num_rank = 0.0

        with torch.no_grad():
            for ix, data in enumerate(self.test_loader):
                if ix % 100 == 0:
                    print('%d/%d' % (ix, len(self.test_loader)))
                batch_size, traj_mask, past_traj, fut_traj, obs_traj, V_tr = self.data_preprocess(data,
                                                                                                  use_rl_data=True)

                sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask)
                sample_prediction = torch.exp(variance_estimation / 2)[
                                        ..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(
                    dim=(1, 2))[:, None, None, None]
                loc = sample_prediction + mean_estimation[:, None]

                pred_traj = self.p_sample_loop_accelerate(past_traj, traj_mask, loc)
                fut_traj_2 = fut_traj.unsqueeze(1)
                distances = torch.norm(fut_traj_2 - pred_traj, dim=-1) * self.traj_scale

                ade = (distances[:, :, :]).mean(dim=-1).min(dim=-1)[0].sum()
                fde = (distances[:, :, -1]).min(dim=-1)[0].sum()
                performance['ADE'] += ade.item()
                performance['FDE'] += fde.item()

                # get diff-score
                if self.test_diff_score:
                    noise_gt_loss, noise_gt_loss_f = self.noise_estimation_loss_score(past_traj, fut_traj, traj_mask)
                    # assert pred_traj.size(1)==20
                    # print(past_traj.size(), fut_traj.size(), pred_traj.size())
                    noise_preds = []
                    noise_preds_final = []
                    for ns in range(pred_traj.size(1)):
                        noise_pred_loss_i, noise_pred_loss_f = self.noise_estimation_loss_score(past_traj,
                                                                                                pred_traj[:, ns, ...],
                                                                                                traj_mask)
                        noise_preds.append(noise_pred_loss_i)
                        noise_preds_final.append(noise_pred_loss_f)
                    # print(noise_gt_loss, noise_min_pred, '-----', noise_mean_pred)
                    noise_min_pred = min(noise_preds)
                    noise_mean_pred = np.mean(noise_preds)
                    noise_min_final_pred = min(noise_preds_final)

                    diff_gt_score += noise_gt_loss

                    if 1 == 1:
                        ade_ind = (distances[:, :, :]).mean(dim=-1).mean(0).min(dim=0)[1].item()
                        fde_ind = (distances[:, :, -1]).mean(dim=0).min(dim=0)[1].item()
                        noise_ind = np.argmin(noise_preds)
                        # noise_fde_min_pred = noise_preds[fde_ind.item()]
                        # print(noise_fde_min_pred, noise_gt_loss)
                        sorted_index = np.argsort(noise_preds)[:5]
                        sorted_index_final = np.argsort(noise_preds_final)[:5]
                        assert noise_ind == sorted_index[0]

                        # noise_preds_2 = np.array([noise_gt_loss.item()] + noise_preds)
                        # order = noise_preds_2.argsort()
                        # ranks = order.argsort()
                        # num_rank += ranks[0]

                        if ade_ind == sorted_index[0]:
                            num_match_ade += 1
                        if fde_ind == sorted_index_final[0]:
                            num_match_fde += 1
                        if ade_ind in sorted_index:
                            print(fde_ind, sorted_index, '--ade')
                            num_match_ade_5 += 1
                        if fde_ind in sorted_index_final:
                            print(fde_ind, sorted_index, '--fde')
                            num_match_fde_5 += 1


                        diff_gt_pred_ratio += 1 - noise_min_pred / (noise_gt_loss + 1e-8)
                        diff_gt_pred_mean_ratio += 1 - noise_mean_pred / (noise_gt_loss + 1e-8)
                        diff_gt_pred_final_ratio += 1 - noise_min_final_pred / (noise_gt_loss + 1e-8)

                # get ip-score
                if True:
                    int_pol_rt_ls = []
                    # [12, N, 2=xy]
                    V_x = seq_to_nodes(obs_traj.data.cpu().numpy().copy())
                    V_y_rel_to_abs = nodes_rel_to_nodes_abs(V_tr.data.cpu().numpy().squeeze().copy(),
                                                            V_x[-1, :, :].copy())
                    V_y_rel_to_abs_tensor = torch.from_numpy(V_y_rel_to_abs).to(obs_traj.device)

                    pred_xy = pred_traj.permute(1, 2, 0, 3).data.cpu().numpy().squeeze().copy()
                    V_pred_rel_to_abs = self.nodes_sample_rel_to_nodes_abs(pred_xy, V_x[-1, :, :].copy())
                    V_pred_rel_to_abs_tensor = torch.from_numpy(V_pred_rel_to_abs).to(obs_traj.device)
                    V_gt_abs_diff = V_y_rel_to_abs_tensor.unsqueeze(2) - V_y_rel_to_abs_tensor.unsqueeze(1)

                    # true_pair_dist = torch.mean(torch.sqrt(torch.sum(V_gt_abs_diff ** 2, dim=-1)), 0)
                    # print(true_pair_dist.size())
                    # true_pair_dist = true_pair_dist[0, :]
                    # print(true_pair_dist)

                    V_pred_abs_diff_all = V_pred_rel_to_abs_tensor.unsqueeze(
                        2 + 1) - V_pred_rel_to_abs_tensor.unsqueeze(1 + 1)
                    # print(V_pred_rel_to_abs_tensor.size())
                    for n_sample in range(V_pred_rel_to_abs_tensor.size(0)):
                        # [12, N, N, 2]
                        V_pred_abs_diff = V_pred_abs_diff_all[n_sample, ...]
                        intimacy_score, politeness_score = intimacy_politeness_score(V_pred_abs_diff, V_gt_abs_diff,
                                                                                     self.social_dist_sigma,
                                                                                     hinge_ratio=0.25)
                        # print(intimacy_score, politeness_score)
                        # int_pol_rt_ls.append(0.5 * (intimacy_score + politeness_score))
                        score_n = 0
                        score_sum = 0
                        if intimacy_score > 0:
                            score_sum += intimacy_score
                            score_n += 1
                        if politeness_score > 0:
                            score_sum += politeness_score
                            score_n += 1
                        if score_n > 0:
                            score_sum /= score_n
                        int_pol_rt_ls.append(score_sum)

                    int_pol_rt_bigls.append(max(int_pol_rt_ls))

                samples += distances.shape[0]
                count += 1
            # if count==2:
            # 	break

        ade_all, fde_all = performance['ADE'] / samples, performance['FDE'] / samples
        ipscore = sum(int_pol_rt_bigls) / len(int_pol_rt_bigls)
        diff_gt_score /= count
        diff_gt_pred_ratio /= count
        diff_gt_pred_mean_ratio /= count
        diff_gt_pred_final_ratio /= count

        print('Test %s, ADE: %.4f, FDE: %.4f, IP: %.4f, DS: %.3f %.3f %.3f, Hit@1 %.3f/%.3f Hit@5 %.3f/%.3f' % (
            self.dset, ade_all, fde_all, ipscore, diff_gt_pred_ratio, diff_gt_pred_final_ratio, diff_gt_pred_mean_ratio,
            num_match_ade / count, num_match_fde / count, num_match_ade_5 / count, num_match_fde_5 / count))
        print(np.round(ade_all, 3), " ", np.round(fde_all, 3), " ",
              np.round(ipscore, 3), " ",
              np.round(diff_gt_pred_ratio, 3), " ",
              np.round(diff_gt_pred_final_ratio, 3), " ",
              np.round(diff_gt_pred_mean_ratio, 3))
        # print(num_rank/count)

    def get_ppo_config(self):
        ppo_config = PPOConfig(
            # mini_batch_size=args.batch_size,
            # batch_size=args.batch_size,
            # init_kl_coef=0.05,
            gradient_accumulation_steps=1,
            early_stopping=False,
            adap_kl_ctrl=True,
            ppo_epochs=1,
            gamma=0.99,
            lam=0.95,
            cliprange_value=0.2,
            vf_coef=0.001,
            target_kl=6,
            kl_penalty="kl",
            seed=123,
            # project_kwargs={'logging_dir': checkpoint_dir}
        )

        return ppo_config

