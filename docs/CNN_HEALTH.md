# CNN health monitor

`scripts/callbacks/cnn_health.py` watches the image network during training and stops a run whose
CNN has died.

## Why

In the run `dev-dict-obs-full` (November 2026) every unit of `conv3` stopped firing between step
30,000 and 35,123 (dead ReLU). From then on all 128 image features were zero, the CNN received no
gradient and the agent could not see obstacles. The run went on for another 24 hours before this
was found. A second run with the same settings died at the same point.

## What it does

Every `check_freq` steps it takes a batch of depth images from the replay buffer, runs them through
`conv1`, `conv2`, `conv3` and the pooling layer, and logs:

| Metric (W&B: `cnn_health/...`) | Meaning | Healthy | Dead |
|---|---|---|---|
| `conv1_active_frac`, `conv2_active_frac`, `conv3_active_frac` | share of the layer outputs above zero | clearly above 0 | 0 |
| `conv3_alive_channels` | channels of conv3 (of 8) with any output above zero | 8, or close to it | 0 |
| `features_nonzero_frac` | share of the 128 image features above zero | clearly above 0 | 0 |
| `conv3_preact_max` | largest value of conv3 before the ReLU | positive | negative or 0 |
| `conv3_preact_mean`, `conv3_bias_mean`, `features_mean` | to see the drift before the death | | |
| `dead` | 1 once the CNN counts as dead | 0 | 1 |

The same numbers are printed to the console (`[CNN health] step ...`).

If the image features are all zero for `patience` checks in a row after `learning_starts`, the CNN
counts as dead. The monitor then prints a warning and, with `stop_when_dead = true`, stops the
training. The model is still saved as `final_model.zip`.

The monitor does not change the training: it only reads the weights and the replay buffer, uses its
own random generator and runs without gradients.

## Settings

The monitor is on by default for SAC and TD3 with a CNN policy. To change it, add this optional
section to the config (the values shown are the defaults):

```ini
[CNN_HEALTH]
enabled = true
check_freq = 500
batch_size = 256
patience = 3
stop_when_dead = true
```

## Seed

The random seed of the algorithm can now be set with the optional `seed` in `[DRL]`. Without it the
seed is 0, the value of the thesis training run.

```ini
[DRL]
seed = 0
```
