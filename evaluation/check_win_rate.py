import json
import os
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline
from accelerate import Accelerator

tqdm.pandas()

@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """
    data_name: Optional[str] = field(
        default="your model",
        metadata={"help": "the location of the SFT model name or path"},
    )
    model_name: Optional[str] = field(
        default="your model",
        metadata={"help": "the location of the SFT model name or path"},
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]


ds_dpo1 = load_dataset("json", data_files=f"{script_args.data_name}_{script_args.model_name}.json", split="train")
# ds_dpo2 = load_dataset("json", data_files=f"{script_args.data_name}_{script_args.model_name}2.json", split="train")
# ds_dpo3 = load_dataset("json", data_files=f"{script_args.data_name}_{script_args.model_name}3.json", split="train")

# baseline = load_dataset("json", data_files=f"{script_args.data_name}_LLaMA3-SFT.json", split="train")
baseline = load_dataset("json", data_files=f"{script_args.data_name}_mistral-instruct.json", split="train")

accelerator = Accelerator()
device = accelerator.device
pipe_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": 4,
}
# reward_model = "Ray2333/GRM-Llama3-8B-rewardmodel-ft"
reward_model = "LxzGordon/URM-LLaMa-3.1-8B"
rm_tokenizer = AutoTokenizer.from_pretrained(reward_model)
rm_pipe = pipeline(
    "sentiment-analysis",
    model=reward_model,
    device=device,
    tokenizer=rm_tokenizer,
    model_kwargs={"torch_dtype": torch.bfloat16},
    truncation=True,
)


def get_reward(test_texts):
    pipe_outputs = rm_pipe(test_texts, **pipe_kwargs)
    rewards = [output[0]["score"] for output in pipe_outputs]
    return rewards


def change_of_format(prom, resp):
    if type(resp) == str:
        message = [{"role": "user", "content": prom}] + [{"role": "assistant", "content": resp}]
        return rm_tokenizer.apply_chat_template(message, tokenize=False).replace(rm_tokenizer.bos_token, "")
    else:
        message = [[{"role": "user", "content": prom}] + [{"role": "assistant", "content": res}] for res in resp]
        return [rm_tokenizer.apply_chat_template(mes, tokenize=False).replace(rm_tokenizer.bos_token, "") for mes in message]


data_bl = []
with torch.no_grad():
    for sample in tqdm(baseline):
        # The VLLM may not generate responses for some prompts because it is too long, we skip them
        test_texts = change_of_format(sample['instruction'], sample['output'])

        rewards = [get_reward(test_text) for test_text in test_texts]
        data_bl.append({"prompt": sample["instruction"], "responses": sample["output"], "rewards": rewards})

# with open(f"{script_args.data_name}_LLaMA3-SFT-reward-urm.json",'w') as f:
#     json.dump(data_bl,f)
# with open(f"{script_args.data_name}_Mistral-reward-urm.json",'w') as f:
#     json.dump(data_bl,f)

# if 'mistral' in script_args.model_name:
#     data_bl = json.load(open(f"{script_args.data_name}_Mistral-reward-urm.json"))
# else:
#     data_bl = json.load(open(f"{script_args.data_name}_LLaMA3-SFT-reward-urm.json"))

for d in data_bl:
    d['rewards'] = [ele[0] for ele in d['rewards']]
# %%

# if not os.path.exists(f"{script_args.data_name}_{script_args.model_name}1-reward.json"):
data_dpo = []
with torch.no_grad():
    for sample in tqdm(ds_dpo1):
        # The VLLM may not generate responses for some prompts because it is too long, we skip them
        test_texts = change_of_format(sample['instruction'], sample['output'])
        rewards = [get_reward(test_text) for test_text in test_texts]
        # rewards = get_reward(test_texts)
        data_dpo.append({"prompt": sample["instruction"], "responses": sample["output"], "rewards": rewards})

with open(f"{script_args.data_name}_{script_args.model_name}1-reward.json",'w') as f:
    json.dump(data_dpo,f)
# else:
#     data_dpo = json.load(open(f"{script_args.data_name}_{script_args.model_name}1-reward.json"))
#     print('Load Existing File')
for d in data_dpo:
    d['rewards'] = [ele[0] for ele in d['rewards']]


for idx in [64]:
    winrate_dpo = [sum(data_dpo[i]["rewards"][:idx]) > sum(data_bl[i]["rewards"][:idx]) for i in range(len(data_bl))]

    print(f"WR@{idx}:")
    print(np.mean(winrate_dpo))

    print(f"AvgR@{idx}:")
    print(np.mean([ele for i in range(len(data_dpo)) for ele in data_dpo[i]["rewards"][:idx] ]))
   
