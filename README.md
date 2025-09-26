<div align="center">

# General Exploratory Bonus for Optimistic Exploration in RLHF

</div>

## Introduction
<div align="center">
<img src="optimistic_v2.drawio.png" width="822px">
</div>

Optimistic exploration is central to improving sample efficiency in reinforcement learning with human feedback, yet existing exploratory bonus methods to incentivize exploration often fail to realize optimism. We provide a theoretical analysis showing that current formulations, under KL or $\alpha$-divergence regularization, unintentionally bias exploration toward high-probability regions of the reference model, thereby reinforcing conservative behavior instead of promoting discovery of uncertain regions. To address this pitfall, we introduce the \textbf{General Exploratory Bonus} (\textbf{GEB}), a novel theoretical framework that provably satisfies the optimism principle. GEB counteracts divergence-induced bias via reference-dependent reward regulation and unifies prior heuristic bonuses as special cases, while extending naturally across the full $\alpha$-divergence family. Empirically, GEB consistently outperforms baselines on alignment tasks across multiple divergence settings and large language model backbones. These results demonstrate that GEB offers both a principled and practical solution for optimistic exploration in RLHF.

## Reproduction

### Train
To run the iterative online RLHF algorithm, please run

```
bash Llama_SFT_GEB.sh
```

There are some arguments you might adjust in the script:
```
loss_type: choose from [dpo,geb_p,geb_f,geb_tanh]
f_div: choose from [kl,hel,fkl]
kappa: the hyperparameter, adjust it according to section 5 of our paper
```



### Evaluation

#### 1. Generate response from models

```
python generation/generate_eval_test.py --model_name_or_path MODEL_NAME --output_name_or_path FILE_NAME
```

#### 2. Check the win rate and average reward

accelerate launch --main_process_port 29710 evaluation/check_win_rate.py --data_name test --model_name FILE_NAME

