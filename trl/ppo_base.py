import inspect
import os
import time
import typing
import warnings
from typing import Callable, List, Optional, Union

import torch


from .core import (
    PPODecorators,
    clip_by_value,
    convert_to_scalar,
    entropy_from_logits,
    flatten_dict,
    logprobs_from_logits,
    masked_mean,
    masked_var,
    masked_whiten,
    set_seed,
    stack_dicts,
    stats_to_np,
)

from . import AdaptiveKLController, BaseTrainer, FixedKLController, PPOConfig


class PPOTrainer(BaseTrainer):
    """
    The PPOTrainer uses Proximal Policy Optimization to optimise language models.
    Note, this trainer is heavily inspired by the original OpenAI learning to summarize work here:
    https://github.com/openai/summarize-from-feedback

    Attributes:
        **config** (`PPOConfig`) -- Configuration object for PPOTrainer. Check the documentation of `PPOConfig` for more
         details.
        **model** (`PreTrainedModelWrapper`) -- Model to be optimized, Hugging Face transformer model with a value head.
            Check the documentation of `PreTrainedModelWrapper` for more details.
        **ref_model** (`PreTrainedModelWrapper`, *optional*) -- Reference model to be used for KL penalty, Hugging Face
            transformer model with a casual language modelling head. Check the documentation of `PreTrainedModelWrapper`
            for more details. If no reference model is provided, the trainer will create a reference model with the same
             architecture as the model to be optimized with shared layers.
        **tokenizer** (`PreTrainedTokenizerBase`) -- Tokenizer to be used for encoding the
            data. Check the documentation of `transformers.PreTrainedTokenizer` and
            `transformers.PreTrainedTokenizerFast` for more details.
        **dataset** (Union[`torch.utils.data.Dataset`, `datasets.Dataset`], *optional*) -- PyTorch dataset or Hugging
            Face dataset. This is used to create a PyTorch dataloader. If no dataset is provided, the dataloader must be
             created outside the trainer users needs to design their own dataloader and make sure the batch
            size that is used is the same as the one specified in the configuration object.
        **optimizer** (`torch.optim.Optimizer`, *optional*) -- Optimizer to be used for training. If no optimizer is
            provided, the trainer will create an Adam optimizer with the learning rate specified in the configuration
            object.
        **data_collator** (DataCollatorForLanguageModeling, *optional*) -- Data collator to be used for training and
            passed along the dataloader
        **num_shared_layers** (int, *optional*) -- Number of layers to be shared between the model and the reference
            model, if no reference model is passed. If no number is provided, all the layers will be shared.
        **lr_scheduler** (`torch.optim.lr_scheduler`, *optional*) -- Learning rate scheduler to be used for training.
    """

    def __init__(self, config: PPOConfig):
        """
        Initialize PPOTrainer.

        Args:
            config (`PPOConfig`):
                Configuration object for PPOTrainer. Check the documentation of `PPOConfig` for more details.
            model (`PreTrainedModelWrapper`):
                Hugging Face transformer model with a value head.
            ref_model (`PreTrainedModelWrapper`):
                Hugging Face transformer model with a casual language modelling head. Used for KL penalty

        """
        super().__init__(config)

        # initial seed for reproducible experiments
        set_seed(config.seed)
        self.ppo_config = config

        if config.adap_kl_ctrl:
            self.kl_ctl = AdaptiveKLController(config.init_kl_coef,
                                               config.target, config.horizon)
        else:
            self.kl_ctl = FixedKLController(config.init_kl_coef)

        # PPODecorators.optimize_cuda_cache = self.config.optimize_cuda_cache


    def compute_rewards(
            self,
            scores: torch.FloatTensor,
            logprobs: torch.FloatTensor,
            ref_logprobs: torch.FloatTensor
    ):
        """
        Compute per token rewards from scores and KL-penalty.

        Args:
            scores (`torch.FloatTensor`):
                Scores from the reward model, shape (`batch_size`, `future_length`)
            logprobs (`torch.FloatTensor`):
                Log probabilities of the model, shape (`batch_size`, `future_length`)
            ref_logprobs (`torch.FloatTensor`):
                Log probabilities of the reference model, shape (`batch_size`, `future_length`)
        """
        kl = self._kl_penalty(logprobs, ref_logprobs)
        # print(logprobs, ref_logprobs)
        non_score_rewards = -self.kl_ctl.value * kl
        rewards = scores + non_score_rewards
        # print(scores.size(),'---')
        #print(non_score_rewards.mean(), scores.mean())
        return rewards, non_score_rewards

        # rewards, non_score_rewards = [], []
        # for score, logprob, ref_logprob in zip(scores, logprobs, ref_logprobs):
        #     # compute KL penalty (from difference in logprobs)
        #     kl = self._kl_penalty(logprob, ref_logprob)
        #     non_score_reward = -self.kl_ctl.value * kl
        #     non_score_rewards.append(non_score_reward)
        #     reward = non_score_reward.clone()
        #     # last_non_masked_index = mask.nonzero()[-1]
        #
        #     # reward is preference model score + KL penalty
        #     reward[last_non_masked_index] += score
        #     rewards.append(reward)
        # return torch.stack(rewards), torch.stack(non_score_rewards)

    def _kl_penalty(self, logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor) -> torch.FloatTensor:
        if self.config.kl_penalty == "kl":
            return logprob - ref_logprob

        if self.config.kl_penalty == "abs":
            return (logprob - ref_logprob).abs()

        if self.config.kl_penalty == "mse":
            return 0.5 * (logprob - ref_logprob).square()

        raise NotImplementedError

    def loss_old(
            self,
            old_logprobs: torch.FloatTensor,
            values: torch.FloatTensor,
            rewards: torch.FloatTensor,
            vpreds: torch.FloatTensor,
            logprobs: torch.FloatTensor
    ):
        """
        Calculate policy and value losses.

        Args:
            old_logprobs (`torch.FloatTensor`):
                Log probabilities of the model, shape (`batch_size`, `response_length`)
            values (`torch.FloatTensor`):
                Values of the value head, shape (`batch_size`, `response_length`)
            rewards (`torch.FloatTensor`):
                Rewards from the reward model, shape (`batch_size`, `response_length`)
            vpreds (`torch.FloatTensor`):
                Values of the value head, shape (`batch_size`, `response_length`)
            logprobs (`torch.FloatTensor`):
                Log probabilities of the model, shape (`batch_size`, `response_length`)
        """
        lastgaelam = 0
        advantages_reversed = []
        gen_len = rewards.shape[0]
        assert values.size(1) == gen_len and values.size(0) == rewards.size(1)

        mask = None

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = rewards[t, :] + self.config.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.config.gamma * self.config.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)

        # print(advantages.size(), values.size(),'---')
        # [N, 12]
        returns = advantages + values
        advantages = masked_whiten(advantages, mask)
        advantages = advantages.detach()

        vpredclipped = clip_by_value(
            vpreds, values - self.config.cliprange_value, values + self.config.cliprange_value
        )

        vf_losses1 = (vpreds - returns) ** 2
        vf_losses2 = (vpredclipped - returns) ** 2
        vf_loss = 0.5 * masked_mean(torch.max(vf_losses1, vf_losses2), mask)
        vf_clipfrac = masked_mean(torch.gt(vf_losses2, vf_losses1).float(), mask)

        ratio = torch.exp(logprobs - old_logprobs).transpose(0, 1)

        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - self.config.cliprange, 1.0 + self.config.cliprange)

        pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), mask)
        pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses).float(), mask)

        loss = pg_loss + self.config.vf_coef * vf_loss

        avg_ratio = masked_mean(ratio, mask).item()
        if avg_ratio > self.config.ratio_threshold:
            warnings.warn(
                f"The average ratio of batch ({avg_ratio:.2f}) exceeds threshold {self.config.ratio_threshold:.2f}. Skipping batch."
            )
            pg_loss = pg_loss * 0.0
            vf_loss = vf_loss * 0.0
            loss = loss * 0.0

        # entropy = masked_mean(entropy_from_logits(logits), mask)
        # print(advantages.size())

        approxkl = 0.5 * masked_mean((logprobs - old_logprobs) ** 2, mask)
        policykl = masked_mean(old_logprobs - logprobs, mask)

        return_mean, return_var = masked_mean(returns, mask), masked_var(returns, mask)
        value_mean, value_var = masked_mean(values, mask), masked_var(values, mask)

        stats = dict(
            loss=dict(policy=pg_loss.detach(), value=vf_loss.detach(), total=loss.detach()),
            policy=dict(
                # entropy=entropy.detach(),
                approxkl=approxkl.detach(),
                policykl=policykl.detach(),
                clipfrac=pg_clipfrac.detach(),
                # advantages=torch.mean(advantages,0).detach(),
                advantages_mean=masked_mean(advantages, mask).detach(),
                ratio=masked_mean(ratio, mask).detach(),
            ),
            returns=dict(mean=return_mean.detach(), var=return_var.detach()),
            val=dict(
                vpred=masked_mean(vpreds, mask).detach(),
                error=masked_mean((vpreds - returns) ** 2, mask).detach(),
                clipfrac=vf_clipfrac.detach(),
                mean=value_mean.detach(),
                var=value_var.detach(),
            ),
        )
        return pg_loss, self.config.vf_coef * vf_loss, flatten_dict(stats)


    def loss_diffusion(
            self,
            old_logprobs: torch.FloatTensor,
            values: torch.FloatTensor,
            rewards: torch.FloatTensor,
            vpreds: torch.FloatTensor,
            logprobs: torch.FloatTensor,
            ret_stat: bool = False
    ):
        """
        Calculate policy and value losses.

        Args:
            old_logprobs (`torch.FloatTensor`):
                Log probabilities of the model, shape (`batch_size`, `response_length`)
            values (`torch.FloatTensor`):
                Values of the value head, shape (`batch_size`, `response_length`)
            rewards (`torch.FloatTensor`):
                Rewards from the reward model, shape (`batch_size`, `response_length`)
            vpreds (`torch.FloatTensor`):
                Values of the value head, shape (`batch_size`, `response_length`)
            logprobs (`torch.FloatTensor`):
                Log probabilities of the model, shape (`batch_size`, `response_length`)
        """
        lastgaelam = 0
        advantages_reversed = []
        gen_len = rewards.shape[1]

        mask = None
        #print(old_logprobs.size(), values.size(),rewards.size(),vpreds.size(),logprobs.size())

        old_logprobs = old_logprobs.reshape(-1, gen_len)
        values = values.reshape(-1, gen_len)
        rewards = rewards.reshape(-1, gen_len)
        vpreds = vpreds.reshape(-1, gen_len)
        logprobs = logprobs.reshape(-1, gen_len)

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = rewards[:, t] + self.config.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.config.gamma * self.config.lam * lastgaelam
            #print(lastgaelam.size())
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)

        # print(advantages.size(), values.size(),'---')
        # [N, 12]
        returns = advantages + values
        advantages = masked_whiten(advantages, mask)
        advantages = advantages.detach()

        vpredclipped = clip_by_value(
            vpreds, values - self.config.cliprange_value, values + self.config.cliprange_value
        )

        vf_losses1 = (vpreds - returns) ** 2
        vf_losses2 = (vpredclipped - returns) ** 2
        vf_loss = 0.5 * masked_mean(torch.max(vf_losses1, vf_losses2), mask)
        vf_clipfrac = masked_mean(torch.gt(vf_losses2, vf_losses1).float(), mask)

        ratio = torch.exp(logprobs - old_logprobs)#.transpose(0, 1)

        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - self.config.cliprange, 1.0 + self.config.cliprange)

        pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), mask)
        pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses).float(), mask)

        loss = pg_loss + self.config.vf_coef * vf_loss

        avg_ratio = masked_mean(ratio, mask).item()
        if avg_ratio > self.config.ratio_threshold:
            warnings.warn(
                f"The average ratio of batch ({avg_ratio:.2f}) exceeds threshold {self.config.ratio_threshold:.2f}. Skipping batch."
            )
            pg_loss = pg_loss * 0.0
            vf_loss = vf_loss * 0.0
            loss = loss * 0.0

        # entropy = masked_mean(entropy_from_logits(logits), mask)
        # print(advantages.size())

        approxkl = 0.5 * masked_mean((logprobs - old_logprobs) ** 2, mask)
        policykl = masked_mean(old_logprobs - logprobs, mask)

        return_mean, return_var = masked_mean(returns, mask), masked_var(returns, mask)
        value_mean, value_var = masked_mean(values, mask), masked_var(values, mask)

        stats = {}
        if ret_stat:
            stats = dict(
                loss=dict(policy=pg_loss.detach(), value=vf_loss.detach(), total=loss.detach()),
                policy=dict(
                    # entropy=entropy.detach(),
                    approxkl=approxkl.detach(),
                    policykl=policykl.detach(),
                    clipfrac=pg_clipfrac.detach(),
                    # advantages=torch.mean(advantages,0).detach(),
                    advantages_mean=masked_mean(advantages, mask).detach(),
                    ratio=masked_mean(ratio, mask).detach(),
                ),
                returns=dict(mean=return_mean.detach(), var=return_var.detach()),
                val=dict(
                    vpred=masked_mean(vpreds, mask).detach(),
                    error=masked_mean((vpreds - returns) ** 2, mask).detach(),
                    clipfrac=vf_clipfrac.detach(),
                    mean=value_mean.detach(),
                    var=value_var.detach(),
                ),
            )
        return pg_loss, self.config.vf_coef * vf_loss, flatten_dict(stats)
