# ACP-Mamba

### Value-Based Thrust Control of Unmanned Underwater Vehicles through Action-Conditioned Predictive State-Space Beliefs

**Tianze Zhang · Xiaowen Tao · Lei Wu · Yangyang Wang · Hongde Qin**

[Repository](https://github.com/Albert-Zhang-UPC/ACP-Mamba) · [Method](#method) · [Results](#results) · [Experimental-Settings](#experimental-settings) · [Citation](#citation)

**Learn what a command earns and how the vehicle responds.** ACP-Mamba is a value-based reinforcement learning controller for unmanned underwater vehicles (UUVs). It combines a Mamba-2 history encoder, branching dueling action values, and an action-conditioned prediction objective to learn control under unobserved flow and payload disturbances.

The prediction head is used **only during training**. Deployment retains the history encoder and action head, with no change to the selected commands after the prediction modules are removed.

![ACP-Mamba architecture: interaction history, Mamba-2 belief encoder, branching Q-networks, sequence replay, and a training-only prediction head.](assets/acp-mamba-overview.png)

*Architecture overview reproduced from Figure 1 of the manuscript.*

## Highlights

- **Predictive belief learning:** the encoder learns from both temporal-difference targets and the next observation produced by the executed command.
- **Scalable discrete control:** 11 levels per control input, with a linear number of branch action values rather than an exponential enumeration of joint commands.
- **Evaluation across tasks and vehicles:** four tasks and three simulated vehicles, compared with memoryless, Transformer, and Mamba-2 Q-networks.
- **Best overall mean decision rank:** **2.042**, with the highest mean return in **9 of 12** task–vehicle combinations.
- **Embedded execution:** the deployed policy runs on an NVIDIA Jetson Orin NX; the manuscript reports **303 MB process memory**, **105 MB GPU memory**, and **6.8 W whole-board power** during execution.

## Overview

Currents and payload changes alter how a UUV responds to its commands, while the controller does not directly observe these disturbances. A single observation can therefore be insufficient for choosing an effective command.

ACP-Mamba encodes recent observations, commands, and rewards into a learned belief representation. A branching Q-network uses this representation to select discrete control levels. During training, a second head predicts the next observation conditioned on the executed command, making the vehicle's response an explicit learning target.

The predictive objective shapes the representation; action selection remains value-based and does not require prediction rollouts or planning.

## Method

### 1. Encode interaction history

At each control step, the input token combines the current observation, the previous command, and the squashed previous reward:

$$
x_t = [o_t;\, a_{t-1};\, \tanh(r_{t-1})].
$$

Two Mamba-2 blocks encode a sliding window of up to **64 tokens** into the belief $b_t$. At an episode reset, the previous command and reward are set to zero. The reported implementation re-encodes the current window from a zero state on every control step; it does not carry the encoder state between policy calls.

### 2. Select a level for each control input

Each input chooses from 11 normalized levels:

$$
\mathcal{U} = \{-1.0, -0.8, \ldots, 0.8, 1.0\}.
$$

A shared state value and a separate advantage branch for each input produce the branch Q-values. Greedy selection takes the highest-valued level in every branch. This exactly maximizes the model's branch-averaged joint value.

| Vehicle | Independent control inputs | Branch Q-values | Possible joint commands |
| --- | ---: | ---: | ---: |
| iAUV | 3 | 33 | 1,331 |
| BlueROV | 6 | 66 | 1,771,561 |
| BlueROV Heavy | 8 | 88 | 214,358,881 |

For iAUV, the three inputs control one propeller and two pairs of fins. BlueROV and BlueROV Heavy have one input per thruster. The counts above refer to branch Q-values; the dueling head parameterizes them with one shared value output and the per-branch advantage outputs.

### 3. Predict the response during training

The auxiliary head predicts the next observation as a residual on the current observation:

$$
\hat{o}_{t+1} = o_t + g_\phi(b_t + W_a a_t).
$$

Here, $W_a$ denotes the learned command projection, including its bias. Training combines a double Q-learning TD loss and an observation-prediction loss:

$$
\mathcal{L} = \mathcal{L}_{\mathrm{TD}} + \lambda\mathcal{L}_{\mathrm{pred}},
\qquad \lambda = 20.
$$

Both use Huber penalties. Losses apply to valid positions after a **32-position warm-up prefix**. Prefix tokens provide context, and gradients still flow through them. Replay windows stay within episode boundaries.

### 4. Export the policy

After training, remove the prediction head $g_\phi$ and command projection $W_a$. Inference uses only:

```text
Observation + previous command + previous reward
                       |
              Interaction history
                       |
               Mamba-2 encoder
                       |
             Branching dueling head
                       |
              Greedy input levels
                       |
                Vehicle commands
```

## Results

The manuscript compares ACP-Mamba with **DQN**, **DTQN**, and **DQN-Mamba2** under a shared interaction budget and action discretization.

### Overall performance

| Method | First places across 72 metric comparisons ↑ | Mean decision rank ↓ | Task–vehicle wins: outright + shared | Mean online trainable parameters |
| --- | ---: | ---: | ---: | ---: |
| DQN | 12 | 3.014 | 0 + 0 | 159,082 |
| DTQN | 18 | 2.375 | 1 + 0 | 108,810 |
| DQN-Mamba2 | 19 | 2.333 | 3 + 1 | 76,474 |
| **ACP-Mamba** | **29** | **2.042** | **7 + 1** | 84,010 |

*Source: manuscript Table 5. Parameter counts are averaged over the 12 task–vehicle combinations. ACP-Mamba's training count includes the prediction modules that are removed before deployment.*

The mean decision rank aggregates six metrics: return and time to failure (higher is better), and failure-aware integrated absolute error, maximum tracking error, throttle-energy proxy, and command second-difference RMS (lower is better). Exact ties share first place, so first-place counts across methods can sum to more than 72.

### Largest mean-return gains

The three largest improvements over the strongest baseline all occur on BlueROV:

| Task / vehicle | Strongest baseline | Baseline return | ACP-Mamba return | Relative gain |
| --- | --- | ---: | ---: | ---: |
| TrackCircle / BlueROV | DQN-Mamba2 | 124.72 ± 58.26 | **196.75 ± 101.53** | **57.8%** |
| Track / BlueROV | DTQN | 137.82 ± 35.53 | **194.11 ± 81.32** | **40.8%** |
| Hover / BlueROV | DTQN | 83.97 ± 50.83 | **97.84 ± 40.48** | **16.5%** |

*Values are mean ± sample standard deviation across 10 paired evaluation episodes (manuscript Table 6). Relative gains compare the means.*

Across return, time to failure, and failure-aware integrated absolute error, **32 of 108** paired comparisons favor ACP-Mamba with a 95% confidence interval excluding zero; **none favors a baseline**. The intervals are not corrected for multiple comparisons.

### Trade-offs and evaluation scope

The performance gains come with **higher throttle energy and less smooth commands** overall. ACP-Mamba does not lead every task–vehicle combination. Throttle energy is a dimensionless proxy based on normalized actuator throttle, not measured electrical energy.

Each method–task–vehicle configuration is trained once with training seed `0`, then evaluated over 10 paired seeds. The reported intervals describe variation across evaluation episodes, not across independent training runs. Training and evaluation use the same disturbance setting. The baseline comparisons also differ in history length, token content, loss positions, and learning rate, so they do not isolate the predictive objective as a single-factor ablation.

Closed-loop performance is evaluated in simulation. Jetson execution demonstrates embedded policy execution; physical at-sea control is identified as future work.

## Experimental Settings

### Simulation tasks

The experiments use **MarineGym with Isaac Sim**. Physics and control advance every **0.016 s**, corresponding to **62.5 Hz**.

| Task | Objective | Maximum episode length |
| --- | --- | ---: |
| Hover | Hold a fixed target pose | 200 steps / 3.2 s |
| Track | Follow a lemniscate reference | 600 steps / 9.6 s |
| TrackCircle | Follow a circular reference | 600 steps / 9.6 s |
| TrackSpiral | Follow a helical reference | 600 steps / 9.6 s |

All four tasks are evaluated on iAUV, BlueROV, and BlueROV Heavy. Tracking-reference scale, orientation, and traversal rate are resampled at reset. Flow and payload variables are excluded from the observation.

### ACP-Mamba training configuration

| Setting | Value |
| --- | --- |
| Requested / effective environment frames | 2,000,000 / 1,998,848 |
| Parallel environments | 32 |
| Collection steps per environment / updates per cycle | 64 / 64 |
| Replay capacity | 500,000 transitions |
| Learning starts | 2,048 transitions |
| Batch size | 128 windows |
| History length / warm-up prefix | 64 / 32 |
| Mamba-2 blocks / model width | 2 / 64 |
| State dimension / convolution width / expansion | 64 / 4 / 2 |
| Head dimension / groups / chunk size | 16 / 1 / 32 |
| Optimizer / learning rate | Adam / 3 × 10⁻⁴ |
| Discount factor | 0.99 |
| Global gradient-norm clip | 10 |
| Soft target-update rate | 0.005 per update |
| Prediction-loss weight | 20 |
| Exploration | Joint-command ε-greedy; ε decreases from 1.0 to 0.05 over 1,000,000 frames |
| Training seed | 0 |
| Evaluation seeds | 61001–61010 |
| Evaluated checkpoint | Final parameters at the end of the frame budget |

The implementation uses **PyTorch** and the official **`mamba-ssm` version 2.2.4**. The configuration was fixed on Track/BlueROV using development seeds disjoint from the evaluation seeds, then applied unchanged to all 12 combinations.

<details>
<summary><strong>Reproduction protocol</strong></summary>

1. Configure the vehicle and task in MarineGym, using the platform's evaluation disturbance setting for both training and evaluation.
2. Discretize each control input into the same 11 normalized command levels for every method.
3. Train each configuration to the effective frame budget, retaining complete episodes in replay and sampling windows without crossing episode boundaries.
4. Evaluate the final checkpoint with exploration disabled, using one episode for each seed from `61001` through `61010` and the same seeds for all methods.
5. Report all evaluation episodes, paired differences, and the six decision metrics; report RMSE separately as a diagnostic.
6. Remove the auxiliary prediction modules for deployment and retain the same sliding-window input convention used in evaluation.

These are the experimental settings reported in the manuscript. Executable installation commands, repository entry points, checkpoint paths, and a complete dependency lockfile are not specified in the manuscript.

</details>

## Embedded Deployment

The deployed policy was executed at batch size one on an **NVIDIA Jetson Orin NX**, running **JetPack 6.2 / Linux for Tegra 36.4.3** in **MAXN SUPER** mode.

| Board-monitor measurement | Reported value |
| --- | ---: |
| Policy process memory | 303 MB |
| Policy process GPU memory | 105 MB |
| Policy process CPU share | 32.0% |
| Integrated GPU utilization | 21% at 306 MHz |
| Aggregate CPU load | 21.6% |
| Whole-board input power | 6.8 W |

*Source: manuscript Section 5.4 and Figure 6. These are board-monitor readings during execution, not an inference-latency benchmark or isolated policy power measurements. The simulation control rate does not establish an embedded latency guarantee.*

## Citation

If you use ACP-Mamba in your research, please cite the manuscript. The entry below uses the available manuscript metadata; update it when final publication details are available.

```bibtex
@misc{zhang_acpmamba,
  title  = {{ACP-Mamba}: Value-Based Thrust Control of Unmanned Underwater Vehicles through Action-Conditioned Predictive State-Space Beliefs},
  author = {Zhang, Tianze and Tao, Xiaowen and Wu, Lei and Wang, Yangyang and Qin, Hongde},
  note   = {Manuscript},
  url    = {https://github.com/Albert-Zhang-UPC/ACP-Mamba}
}
```
