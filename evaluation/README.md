# Evaluation


### 1. Generate response from models

```
python generation/generate_eval_alphca.py --model_name_or_path MODEL_NAME --output_name_or_path FILE_NAME
python generation/generate_eval_test.py --model_name_or_path MODEL_NAME --output_name_or_path FILE_NAME
```

### 2. Check the win rate and average reward

accelerate launch --main_process_port 29710 evaluation/check_win_rate.py --data_name test --model_name FILE_NAME
