import inspect
import random
import warnings
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from accelerate.utils import is_deepspeed_available, tqdm
from datasets import Dataset
from huggingface_hub.utils._deprecation import _deprecate_arguments
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
)
from trl.trainer.utils import cap_exp
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput
from trl import DPOTrainer
from trl.trainer.dpo_config import DPOConfig, FDivergenceConstants, FDivergenceType



def _tokenize(
    features: Dict[str, List],
    tokenizer: PreTrainedTokenizerBase,
    args: DPOConfig,
    processor: Optional[Callable] = None,
    model: Optional[PreTrainedModel] = None,
) -> Dict[str, List]:
    """
    Tokenizes and processes a batch of input features using the provided tokenizer and processor.
    """
    batch = defaultdict(list)

    if model is None:
        prompt = features["prompt"]
        images = features.get("images", [None] * len(features["prompt"]))

        prompt_tokens = _process_prompt(prompt, processor, tokenizer, images)
        chosen_tokens = _process_answer(prompt, features["chosen"], processor, tokenizer, images, features["chosen_index"])
        rejected_tokens = _process_answer(prompt, features["rejected"], processor, tokenizer, images, features["rejected_index"])

        prompt_len_input_ids = _adjust_prompt_length(prompt_tokens, chosen_tokens, rejected_tokens)

        prompt_tokens, chosen_tokens, rejected_tokens = _add_special_tokens(
            tokenizer, prompt_len_input_ids, prompt_tokens, chosen_tokens, rejected_tokens
        )

        _truncate_tokens(chosen_tokens, rejected_tokens, prompt_tokens, args)

        _build_sequence_tokens(batch, chosen_tokens, args, "chosen")
        _build_sequence_tokens(batch, rejected_tokens, args, "rejected")

        _append_prompt_tokens_to_batch(batch, prompt_tokens)

    else:
        _tokenize_encoder_decoder(batch, tokenizer, features["prompt"], features["chosen"], features["rejected"], args)

    return dict(batch)


def _process_prompt(
    prompts: List[str], processor: Optional[Callable], tokenizer: PreTrainedTokenizerBase, images: List[Optional[Any]]
) -> List[Dict[str, List[int]]]:
    """
    Processes a list of prompts by tokenizing them, optionally using a processor for additional processing.
    """
    if processor:
        processor_kwargs = (
            {"add_special_tokens": False} if "add_special_tokens" in inspect.signature(processor).parameters else {}
        )
        prompt_tokens = []
        for prompt, image in zip(prompts, images):
            tokens = processor(images=image, text=prompt, **processor_kwargs)
            tokens = {k: v[0] for k, v in tokens.items()}
            if not isinstance(tokens["input_ids"], list):
                tokens["input_ids"] = tokens["input_ids"].tolist()
                tokens["attention_mask"] = tokens["attention_mask"].tolist()
            prompt_tokens.append(tokens)
    else:
        prompt_tokens = [tokenizer(prompt, add_special_tokens=False) for prompt in prompts]
    return [{f"prompt_{k}": v for k, v in tokens.items()} for tokens in prompt_tokens]


def _process_answer(
    prompts: List[str],
    answers: List[str],
    processor: Optional[Callable],
    tokenizer: PreTrainedTokenizerBase,
    images: List[Optional[Any]],
    indexes: List[int],
) -> List[Dict[str, Any]]:
    return [
        _build_tokenized_answer(prompt, answer, image, processor=processor, tokenizer=tokenizer, index=index)
        for prompt, answer, image, index in zip(prompts, answers, images, indexes)
    ]


def _adjust_prompt_length(
    prompt_tokens: List[Dict[str, List[int]]],
    chosen_tokens: List[Dict[str, List[int]]],
    rejected_tokens: List[Dict[str, List[int]]],
) -> List[int]:
    prompt_len_input_ids = []
    for p_tokens, c_tokens, r_tokens in zip(prompt_tokens, chosen_tokens, rejected_tokens):
        c_len = len(c_tokens["prompt_input_ids"])
        r_len = len(r_tokens["prompt_input_ids"])
        min_len = min(c_len, r_len)

        for k, v in p_tokens.items():
            p_tokens[k] = v[:min_len]

        num_diff_tokens = sum([a != b for a, b in zip(c_tokens["prompt_input_ids"], r_tokens["prompt_input_ids"])])
        num_diff_len = abs(c_len - r_len)
        if num_diff_tokens > 1 or num_diff_len > 1:
            raise ValueError(
                "Chosen and rejected prompt_input_ids might only differ on the last token due to tokenizer merge ops."
            )
        prompt_len_input_ids.append(min_len)
    return prompt_len_input_ids


def _add_special_tokens(
    tokenizer: PreTrainedTokenizerBase,
    prompt_len_input_ids: List[int],
    prompt_tokens: List[Dict[str, List[int]]],
    chosen_tokens: List[Dict[str, List[int]]],
    rejected_tokens: List[Dict[str, List[int]]],
) -> Tuple[List[Dict[str, List[int]]], List[Dict[str, List[int]]], List[Dict[str, List[int]]]]:
    for i in range(len(prompt_tokens)):
        prompt_tokens[i], chosen_tokens[i], rejected_tokens[i] = add_bos_token_if_needed(
            tokenizer.bos_token_id,
            prompt_len_input_ids[i],
            prompt_tokens[i],
            len(chosen_tokens[i]["prompt_input_ids"]),
            chosen_tokens[i],
            len(rejected_tokens[i]["prompt_input_ids"]),
            rejected_tokens[i],
        )

        chosen_tokens[i], rejected_tokens[i] = add_eos_token_if_needed(
            tokenizer.eos_token_id, chosen_tokens[i], rejected_tokens[i]
        )
    return prompt_tokens, chosen_tokens, rejected_tokens


def _truncate_tokens(
    chosen_tokens: List[Dict[str, List[int]]],
    rejected_tokens: List[Dict[str, List[int]]],
    prompt_tokens: List[Dict[str, List[int]]],
    args: DPOConfig,
) -> None:
    """
    Truncates the tokens in chosen, rejected, and prompt sequences to ensure they fit within the maximum length constraints.
    """
    if args.truncation_mode not in ["keep_start", "keep_end"]:
        raise ValueError(f"Invalid truncation mode: {args.truncation_mode}")

    for c_tokens, r_tokens, p_tokens in zip(chosen_tokens, rejected_tokens, prompt_tokens):
        longer_response_length = max(len(c_tokens["input_ids"]), len(r_tokens["input_ids"]))

        # if combined sequence is too long, truncate the prompt
        for answer_tokens in [c_tokens, r_tokens, p_tokens]:
            if len(answer_tokens["prompt_input_ids"]) + longer_response_length > args.max_length:
                if args.truncation_mode == "keep_start":
                    for k in ["prompt_input_ids", "prompt_attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][: args.max_prompt_length]
                elif args.truncation_mode == "keep_end":
                    for k in ["prompt_input_ids", "prompt_attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][-args.max_prompt_length :]

        # if that's still too long, truncate the response from the end
        for answer_tokens in [c_tokens, r_tokens]:
            if len(answer_tokens["prompt_input_ids"]) + longer_response_length > args.max_length:
                for k in ["input_ids", "attention_mask"]:
                    answer_tokens[k] = answer_tokens[k][: args.max_length - args.max_prompt_length]


def _build_sequence_tokens(
    batch: Dict[str, List[int]], tokens: List[Dict[str, List[int]]], args: DPOConfig, prefix: str
) -> None:
    for token in tokens:
        sequence_tokens = {f"{prefix}_{k}": token[f"prompt_{k}"] + token[k] for k in ["input_ids", "attention_mask"]}
        sequence_tokens[f"{prefix}_index"] = token["index"]
        sequence_tokens[f"{prefix}_labels"] = sequence_tokens[f"{prefix}_input_ids"][:]
        sequence_tokens[f"{prefix}_labels"][: len(token["prompt_input_ids"])] = [args.label_pad_token_id] * len(
            token["prompt_input_ids"]
        )
        for k, v in sequence_tokens.items():
            batch[k].append(v)


def _append_prompt_tokens_to_batch(batch: Dict[str, List[int]], prompt_tokens: List[Dict[str, List[int]]]) -> None:
    for p_tokens in prompt_tokens:
        for k, v in p_tokens.items():
            batch[k].append(v)


def _tokenize_encoder_decoder(
    batch: Dict[str, List[int]],
    tokenizer: PreTrainedTokenizerBase,
    prompt: List[str],
    chosen: List[str],
    rejected: List[str],
    args: DPOConfig,
) -> None:
    chosen_tokens = tokenizer(chosen, truncation=True, max_length=args.max_completion_length, add_special_tokens=True)
    rejected_tokens = tokenizer(
        rejected, truncation=True, max_length=args.max_completion_length, add_special_tokens=True
    )
    prompt_tokens = tokenizer(prompt, truncation=True, max_length=args.max_prompt_length, add_special_tokens=True)

    batch["chosen_labels"] = chosen_tokens["input_ids"]
    batch["rejected_labels"] = rejected_tokens["input_ids"]
    batch["prompt_input_ids"] = prompt_tokens["input_ids"]
    batch["prompt_attention_mask"] = prompt_tokens["attention_mask"]


def _build_tokenized_answer(
    prompt: str,
    answer: str,
    images: Optional[List[Any]] = None,
    processor: Optional[Callable] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    index: int = 0, # if it is the first response index=0
) -> Dict[str, Any]:
    """
    Build tokenized response, handling vision models and different tokenizers.
    """

    def tokenize(text, images=None):
        if processor:
            processor_kwargs = (
                {"add_special_tokens": False}
                if "add_special_tokens" in inspect.signature(processor).parameters
                else {}
            )
            tokenized = processor(images=images, text=text, **processor_kwargs)
            tokenized = {k: v[0] for k, v in tokenized.items()}
            if not isinstance(tokenized["input_ids"], list):
                tokenized["input_ids"] = tokenized["input_ids"].tolist()
                tokenized["attention_mask"] = tokenized["attention_mask"].tolist()
        else:
            tokenized = tokenizer(text, add_special_tokens=False)
        return tokenized

    full_tokenized = tokenize(prompt + answer, images)
    prompt_tokenized = tokenize(prompt, images)

    prompt_input_ids = prompt_tokenized["input_ids"]
    answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids) :]
    answer_attention_mask = full_tokenized["attention_mask"][len(prompt_input_ids) :]

    if len(full_tokenized["input_ids"]) != len(prompt_input_ids + answer_input_ids):
        raise ValueError("Prompt input ids and answer input ids should have the same length.")

    # On some tokenizers, like Llama-2 tokenizer, there are occasions where tokens
    # can be merged together when tokenizing prompt+answer. This could result
    # on the last token from the prompt being different when tokenized on its own
    # vs when done as prompt+answer.
    response_token_ids_start_idx = len(prompt_input_ids)

    # If tokenized prompt is different than both prompt+answer, then it means the
    # last token has changed due to merging.
    if prompt_input_ids != full_tokenized["input_ids"][:response_token_ids_start_idx]:
        response_token_ids_start_idx -= 1

    prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
    prompt_attention_mask = full_tokenized["attention_mask"][:response_token_ids_start_idx]

    if len(prompt_input_ids) != len(prompt_attention_mask):
        raise ValueError("Prompt input ids and attention mask should have the same length.")

    return_dict = {
        "prompt_input_ids": prompt_input_ids,
        "prompt_attention_mask": prompt_attention_mask,
        "input_ids": answer_input_ids,
        "attention_mask": answer_attention_mask,
        "index": index
    }
    if "pixel_values" in full_tokenized:
        return_dict["prompt_pixel_values"] = full_tokenized["pixel_values"]
    if "pixel_attention_mask" in full_tokenized:
        return_dict["prompt_pixel_attention_mask"] = full_tokenized["pixel_attention_mask"]

    return return_dict


def get_arctanh_cap(value, decimal=4):
    """
    Get the exponent cap of a value. This is used to cap the exponent of a value to avoid overflow. The formula is :
    log(value.dtype.max) E.g.
      For float32 data type, the maximum exponent value is 88.7228 to 4 decimal points.

    Args:
        value (`torch.Tensor`):
            The input tensor to obtain the data type
        decimal (`int`):
            The number of decimal points of the output exponent cap. eg: direct calling exp(log(torch.float32.max))
            will result in inf so we cap the exponent to 88.7228 to avoid overflow.
    """
    vdtype_max = torch.zeros([1]).to(value.dtype) + torch.finfo(value.dtype).max
    vdtype_tanh_max = torch.tanh(vdtype_max).to(value.device)
    return torch.floor(vdtype_tanh_max * 10**decimal) / 10**decimal if decimal > 0 else vdtype_tanh_max


def cap_arctanh(value, cap=-1):
    # Cap the exponent value below the upper-bound to avoid overflow, before calling torch.exp
    cap = get_arctanh_cap(value) if cap < 0 else cap
    return torch.arctanh(torch.clamp(value, max=cap))





















class MyWPOTrainer(DPOTrainer):
    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        beta: float = 0.1,
        label_smoothing: float = 0,
        loss_type: Optional[str] = None,
        args: Optional[DPOConfig] = None,
        data_collator: Optional[DataCollator] = None,
        label_pad_token_id: int = -100,
        padding_value: Optional[int] = None,
        truncation_mode: str = "keep_end",
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
        is_encoder_decoder: Optional[bool] = None,
        disable_dropout: bool = True,
        generate_during_eval: bool = False,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
        precompute_ref_log_probs: bool = False,
        dataset_num_proc: Optional[int] = None,
        model_init_kwargs: Optional[Dict] = None,
        ref_model_init_kwargs: Optional[Dict] = None,
        model_adapter_name: Optional[str] = None,
        ref_adapter_name: Optional[str] = None,
        reference_free: bool = False,
        force_use_ref_model: bool = False,
        alpha: float = 0,
        f_div: str='fkl'
    ):
        super().__init__(
            model=model,
            ref_model=ref_model,
            beta=beta,
            label_smoothing=label_smoothing,
            loss_type=loss_type,
            args=args,
            data_collator=data_collator,
            label_pad_token_id=label_pad_token_id,
            padding_value=padding_value,
            truncation_mode=truncation_mode,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
            max_target_length=max_target_length,
            peft_config=peft_config,
            is_encoder_decoder=is_encoder_decoder,
            disable_dropout=disable_dropout,
            generate_during_eval=generate_during_eval,
            compute_metrics=compute_metrics,
            precompute_ref_log_probs=precompute_ref_log_probs,
            dataset_num_proc=dataset_num_proc,
            model_init_kwargs=model_init_kwargs,
            ref_model_init_kwargs=ref_model_init_kwargs,
            model_adapter_name=model_adapter_name,
            ref_adapter_name=ref_adapter_name,
            reference_free=reference_free,
            force_use_ref_model=force_use_ref_model,
        )
        self.alpha = alpha
        self.f_div = f_div


    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        policy_chosen_ps,
        policy_rejected_ps,
        reference_chosen_ps,
        reference_rejected_ps,
        chosen_index: List[int],
        rejected_index: List[int],
    ) -> Tuple[torch.FloatTensor,torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        chosen_logratios = policy_chosen_logps.to(self.accelerator.device) - (
            not self.reference_free
        ) * reference_chosen_logps.to(self.accelerator.device)
        rejected_logratios = policy_rejected_logps.to(self.accelerator.device) - (
            not self.reference_free
        ) * reference_rejected_logps.to(self.accelerator.device)


        if self.f_divergence_type == FDivergenceType.ALPHA_DIVERGENCE.value:
            # The alpha-divergence formula: (1 - u^-alpha) / alpha
            # The divergence difference between the chosen and rejected sample is:
            #     (1 - u[w]^-alpha) / alpha - (1 - u[l]^-alpha) / alpha
            #        = (u[l]^-alpha - u[w]^-alpha) / alpha
            # where u[w] and u[l] are the policy/reference probability ratios
            # for the chosen and rejected samples, respectively.
            alpha_coef = FDivergenceConstants.ALPHA_DIVERGENCE_COEF_DEFAULT
            if self.f_divergence_params and FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY in self.f_divergence_params:
                alpha_coef = float(self.f_divergence_params[FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY])
            logits = (cap_exp(rejected_logratios * -alpha_coef) - cap_exp(chosen_logratios * -alpha_coef)) / alpha_coef
        else:
            pi_logratios = policy_chosen_logps - policy_rejected_logps
            if self.reference_free:
                ref_logratios = torch.tensor([0], dtype=pi_logratios.dtype, device=pi_logratios.device)
            else:
                ref_logratios = reference_chosen_logps - reference_rejected_logps

            pi_logratios = pi_logratios.to(self.accelerator.device)
            ref_logratios = ref_logratios.to(self.accelerator.device)
            logits = pi_logratios - ref_logratios

            if self.f_divergence_type == FDivergenceType.JS_DIVERGENCE.value:
                # The js-divergence formula: log(2 * u / (1 + u))
                # The divergence difference between the chosen and rejected sample is:
                #     log(2 * u[w] / (1 + u[w])) - log(2 * u[l] / (1 + u[l]))
                #       = log(u[w]) - log(u[l]) - (log(1 + u[w]) - log(1 + u[l]))
                # where u[w] and u[l] are the policy/reference probability ratios
                # for the chosen and rejected samples, respectively.
                logits -= F.softplus(chosen_logratios) - F.softplus(rejected_logratios)

        # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
        # calculates a conservative DPO loss.

        if self.f_div == 'kl':
            pass
        elif self.f_div == 'hel':
            logits = (cap_exp(rejected_logratios * -0.5) - cap_exp(chosen_logratios * -0.5)) / 0.5
        elif self.f_div == 'fkl':
            logits = (cap_exp(rejected_logratios * -1) - cap_exp(chosen_logratios * -1)) / 1
        else:
            raise NotImplementedError



        losses = (
            -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
            - F.logsigmoid(-self.beta * logits) * self.label_smoothing
        )

        # sppo
        # c = torch.tensor(0.5, device=chosen_logratios.device)
        # d = torch.tensor(0., device=chosen_logratios.device)
        # c1 = c - d
        # c2 = -c - d
        #
        # #KL
        # if self.f_div == 'kl':
        #     a = policy_chosen_logps - reference_chosen_logps
        #     b = policy_rejected_logps - reference_rejected_logps
        #
        # #hel
        # elif self.f_div == 'hel':
        #     a = (policy_chosen_logps.exp()/(reference_chosen_logps.exp()+1e-8)).sqrt()
        #     b = (policy_rejected_logps.exp()/(reference_rejected_logps.exp()+1e-8)).sqrt()
        #     c1 = c1.exp().sqrt()
        #     c2 = c2.exp().sqrt()
        #
        # #fKL
        # elif self.f_div == 'fkl':
        #     a =  - reference_chosen_logps.exp()/(policy_chosen_logps.exp()+1e-8)
        #     b = - reference_rejected_logps.exp()/(policy_rejected_logps.exp()+1e-8)
        #     c1 = -1/(c1.exp())
        #     c2 = -1/(c2.exp())
        #
        # else:
        #     raise NotImplementedError



        #sppo
        # losses = (a - c1 / self.beta) ** 2 + (b - c2 / self.beta) ** 2

        # constant
        # losses = (a - 0.1 / self.beta) ** 2 + (b + 0.9 / self.beta) ** 2

        # mean_ref_logps = (self.accelerator.gather(reference_chosen_logps).mean() + self.accelerator.gather(reference_rejected_logps).mean())/2
        # chosen_c = self.beta * reference_chosen_logps/mean_ref_logps/2
        # rejected_c = self.beta * reference_rejected_logps / mean_ref_logps/2
        # chosen_c = torch.where(chosen_c>0.5,0.5,chosen_c) / self.beta
        # rejected_c = torch.where(rejected_c > 0.5, 0.5, chosen_c) / self.beta
        # losses = (a - 0.5 / self.beta + chosen_c) ** 2 + (b + 0.5 / self.beta + rejected_c) ** 2
        # self.accelerator.print(chosen_c.mean(),rejected_c.mean())

        #slack version
        # losses = torch.where(a < 0.1 / self.beta,(a - 0.1 / self.beta) ** 2,(a.detach() - 0.1 / self.beta) ** 2) + \
        #          torch.where(b > -0.9 / self.beta,(b + 0.9 / self.beta) ** 2,(b.detach() + 0.9 / self.beta) ** 2)




        if self.loss_type in "dpo":
            bonus = torch.zeros_like(losses)
        
        elif self.loss_type == "geb_p":
            # bonus =  self.alpha * (
            #                     policy_chosen_logps.exp() * torch.tensor(chosen_index, device=chosen_logratios.device) + \
            #                     policy_rejected_logps.exp() * torch.tensor(rejected_index,device=rejected_logratios.device))
            if self.f_div == 'kl':
                bonus = self.alpha * policy_rejected_logps.exp()
            elif self.f_div == 'fkl':
                bonus = - self.alpha * torch.log(1 - policy_rejected_logps.exp())
            elif self.f_div == 'hel':
                bonus = - self.alpha * (1.5 - policy_rejected_logps.exp()).sqrt()
            else:
                print(self.f_div)
                raise NotImplementedError

        elif self.loss_type == "geb_f":
            if self.f_div == 'kl':
                bonus = - self.alpha * 1/( policy_rejected_logps.exp() + 1e-5 )
            elif self.f_div == 'hel':
                bonus = - self.alpha * 1/( (0.5*policy_rejected_logps).exp() + 1e-5 )
            elif self.f_div == 'fkl':
                bonus = self.alpha * policy_rejected_logps

        elif self.loss_type == "geb_tanh":
            # print(policy_chosen_ps,torch.arctanh(policy_chosen_ps-1))
            # print(policy_rejected_ps, torch.arctanh(policy_rejected_ps - 1))
            # bonus = self.alpha * (
            #         torch.arctanh(policy_chosen_ps-1) * torch.tensor(chosen_index,device=chosen_logratios.device) + \
            #         torch.arctanh(policy_rejected_ps-1)* torch.tensor(rejected_index,device=rejected_logratios.device))
            if self.f_div == 'kl':
                bonus = - self.alpha * torch.arctanh(1 - policy_rejected_logps.exp() - 1e-5)
            elif self.f_div == 'fkl':
                bonus = - self.alpha * torch.log(torch.arctanh(1 - policy_rejected_logps.exp() - 1e-5 ) )
            elif self.f_div == 'hel':
                bonus = - self.alpha * (torch.arctanh(1 - policy_rejected_logps.exp() - 1e-5) + 0.5).sqrt()
            else:
                print(self.f_div)
                raise NotImplementedError
        else:
            raise NotImplementedError
            
        chosen_rewards = (
            self.beta
            * (
                policy_chosen_logps.to(self.accelerator.device) - reference_chosen_logps.to(self.accelerator.device)
            ).detach()
        )
        rejected_rewards = (
            self.beta
            * (
                policy_rejected_logps.to(self.accelerator.device)
                - reference_rejected_logps.to(self.accelerator.device)
            ).detach()
        )

        return losses, bonus, chosen_rewards, rejected_rewards

    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
    ) -> Tuple[torch.FloatTensor,torch.FloatTensor, torch.LongTensor, torch.LongTensor]:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            label_pad_token_id: The label pad token id.
            is_encoder_decoder: Whether the model is an encoder-decoder model.

        Returns:
            A Tuple of two tensor of shape ((batch_size,), (batch_size,)) containing the sum of log probabilities of the given labels under the given logits in the first tensor and the number of non-masked tokens in the second tensor.
        """

        def entropy_from_logits(logits: torch.Tensor):
            """Calculate entropy from logits."""
            pd = torch.nn.functional.softmax(logits, dim=-1)
            entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
            return entropy

        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id
        entropy = entropy_from_logits(logits)
        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        logsoftmax_logits = logits.log_softmax(-1)

        per_token_logps = torch.gather(logsoftmax_logits, dim=2, index=labels.unsqueeze(2)).squeeze(2)
        per_token_ps = torch.gather(logsoftmax_logits, dim=2, index=labels.unsqueeze(2)).squeeze(2)

        return (per_token_logps * loss_mask).sum(-1), (per_token_ps * loss_mask).sum(-1)/loss_mask.sum(-1), loss_mask.sum(-1),(entropy*loss_mask).sum(-1)

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            is_vision_model=self.is_vision_model,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = {}

        if self.is_encoder_decoder:
            model_kwargs["labels"] = concatenated_batch["concatenated_labels"]
            model_kwargs["decoder_input_ids"] = concatenated_batch.pop("concatenated_decoder_input_ids", None)

        if self.is_vision_model:
            model_kwargs["pixel_values"] = concatenated_batch["pixel_values"]
            if "pixel_attention_mask" in concatenated_batch:
                model_kwargs["pixel_attention_mask"] = concatenated_batch["pixel_attention_mask"]

        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        outputs = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            **model_kwargs,
        )
        all_logits = outputs.logits

        if all_logits.shape[:2] != concatenated_batch["concatenated_labels"].shape[:2]:
            # for llava, the model returns logits for the entire sequence, including the image tokens (placed before the text tokens)
            seq_len = concatenated_batch["concatenated_labels"].shape[1]
            all_logits = all_logits[:, -seq_len:]

        all_logps, all_ps, size_completion,entropy = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            # average_log_prob=self.loss_type == "ipo",
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        def cross_entropy_loss(logits, labels):
            if not self.is_encoder_decoder:
                # Shift so that tokens < n predict n
                logits = logits[..., :-1, :].contiguous()
                labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            logits = logits.view(-1, logits.shape[-1])
            labels = labels.view(-1)
            # Enable model parallelism
            labels = labels.to(logits.device)
            loss = loss_fct(logits, labels)
            return loss

        labels = concatenated_batch["concatenated_labels"].clone()
        nll_loss = cross_entropy_loss(all_logits[:len_chosen], labels[:len_chosen])

        if self.loss_type == "ipo":
            all_logps = all_logps / size_completion

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]
        chosen_ps = all_ps[:len_chosen]
        rejected_ps = all_ps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        if self.aux_loss_enabled:
            return (chosen_logps, rejected_logps, chosen_ps, rejected_ps,chosen_logits, rejected_logits, nll_loss, outputs.aux_loss,entropy)

        return (chosen_logps, rejected_logps,chosen_ps, rejected_ps, chosen_logits, rejected_logits, nll_loss,entropy)

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        #print(batch["chosen_index"], batch["rejected_index"])

        forward_output = self.concatenated_forward(model, batch)
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_ps,
            policy_rejected_ps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_nll_loss,
            entropy
        ) = forward_output[:]
        if self.aux_loss_enabled:
            aux_loss = forward_output[5]

        # if reference_chosen_logps and reference_rejected_logps in batch use them, otherwise use the reference model
        if (
            "reference_chosen_logps" in batch
            and "reference_rejected_logps" in batch
            and self.args.rpo_alpha is not None
        ):
            reference_chosen_logps = batch["reference_chosen_logps"]
            reference_rejected_logps = batch["reference_rejected_logps"]
        else:
            with torch.no_grad():
                if self.ref_model is None:
                    with self.null_ref_context():
                        (
                            reference_chosen_logps,
                            reference_rejected_logps,
                            reference_chosen_ps,
                            reference_rejected_ps,
                            _,
                            _,
                            _,
                            _
                        ) = self.concatenated_forward(self.model, batch)
                else:
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        reference_chosen_ps,
                        reference_rejected_ps,
                        _,
                        _,
                        _,
                        _
                    ) = self.concatenated_forward(self.ref_model, batch)

        dpo_loss,loss_bonus, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            policy_chosen_ps,
            policy_rejected_ps,
            reference_chosen_ps,
            reference_rejected_ps,
            batch["chosen_index"],
            batch["rejected_index"],
        )
        losses = dpo_loss + loss_bonus
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        if self.args.rpo_alpha is not None:
            # RPO loss from V3 of the paper:
            losses = losses + policy_nll_loss * self.args.rpo_alpha

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().mean().cpu()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu()
        metrics[f"{prefix}entropy"] = entropy.detach().mean().cpu()
        metrics[f"{prefix}loss/dpo_loss"] = dpo_loss.detach().mean().cpu()
        metrics[f"{prefix}loss/loss_bonus"] = loss_bonus.detach().mean().cpu()
        if self.args.rpo_alpha is not None:
            metrics[f"{prefix}nll_loss"] = policy_nll_loss.detach().mean().cpu()

        if self.aux_loss_enabled:
            return losses.mean() + getattr(model.config, "router_aux_loss_coef", 0.0) * aux_loss, metrics

        return losses.mean(), metrics

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.use_dpo_data_collator:
            warnings.warn(
                "compute_loss is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )

        compute_loss_context_manager = amp.autocast("cuda") if self._peft_has_been_casted_to_bf16 else nullcontext()

        with compute_loss_context_manager:
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        # Make sure to move the loss to the device the original accumulating loss is at back in the `Trainer` class:
        loss = loss.to(self.args.device)
        # force log the metrics
        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss


    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        if not self.use_dpo_data_collator:
            warnings.warn(
                "prediction_step is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        prediction_context_manager = amp.autocast("cuda") if self._peft_has_been_casted_to_bf16 else nullcontext()

        with torch.no_grad(), prediction_context_manager:
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="eval")

        # force log the metrics
        self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        # logits for the chosen and rejected samples from model
        logits_dict = {
            "eval_logits/chosen": metrics["eval_logits/chosen"],
            "eval_logits/rejected": metrics["eval_logits/rejected"],
        }
        logits = tuple(v.unsqueeze(dim=0) for k, v in logits_dict.items() if k not in ignore_keys)
        logits = torch.stack(logits).mean(axis=1).to(self.accelerator.device)
        labels = torch.zeros(logits.shape[0], device=self.accelerator.device)

        return (loss.detach(), logits, labels)
