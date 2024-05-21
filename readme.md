# Official Implementation of a recent submission under peer review


## Data
We include our collected robo dataset for this demo which can run self-contained.

We packed all processed data and provide a drive link.

## Training with PPO and Reject Sampling (only use best trajectory of reward)

Step 1: use RLHF-DS
```
python main.py --md='train_rl' --lr=2e-4 -d robo --gpu=0 --w_rl=0.30 -rs
```

Step 2: further fine-tune with RLHF-DS-RS
```
python main.py --md='train_rl' --lr=2e-4 -d robo --gpu=0 --w_rl=0.30 -rs -p result_diffusion/PREVIOUS_SAVED_RLHF-DS-METHOD
```

Step 3: test
```
python main.py --md='test' -d nba --gpu=0  -ds -p result_diffusion/PREVIOUS_SAVED_RLHF-DS-METHOD
```


## test with our trained RLHF_DS_RS model
```
python main.py --md='test' -d robo -ds -p model_save/model_robo --gpu=0 
```