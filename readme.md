# Official Implementation of a recent submission under peer review


## Data
We include our collected robo dataset for this demo which can run self-contained.

We packed all processed data and provide a drive link.

## Training with PPO and Reject Sampling (only use best trajectory of reward)

Step 1: use RLHF-DS
```
python train_led.py --md='train_rl' --lr=2e-4 -d robo --gpu=0 --w_rl=0.30 -rs
```

Step 2: further fine-tune with RLHF-DS-RS
```
python train_led.py --md='train_rl' --lr=2e-4 -d robo --gpu=0 --w_rl=0.30 -rs -p result_diffusion/PREVIOUS_SAVED_RLHF-DS-METHOD
```
