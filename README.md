# MAPPO-RIS-assisted HAPS-ISAC Systems

## Overview
The goal of this research is to develop an advanced policy-based deep reinforcement learning (DRL) framework that can jointly optimize beamforming, power allocation, and reconfigurable intelligent surface (RIS) 
phase shifts in a **High-Altitude Platform Station (HAPS) multiple-input multiple-output (MIMO) Integrated Sensing and Communication (ISAC) system**. By deploying a RIS alongside a HAPS transmitter, our method aims 
to enhance *sum-rate*, *beampattern gain*, and *sensing performance* in dynamic network environments. The optimization framework uses a **multi-agent proximal policy optimization (MAPPO) algorithm** with centralized 
**training and decentralized execution (CTDE)**, addressing communication beamforming, power control, and RIS configuration as cooperative sub-problems.

---

## Contents
- [System Model](#system-model)
- [Deep Reinforcement Learning Solution](#deep-reinforcement-learning-solution)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

---

## 1. System Model

The proposed system model comprises a **RIS-assisted HAPS MIMO ISAC downlink network** with a *Uniform Planar Array (UPA)* transmitter at an altitude of **20km**, a passive *RIS* deployed at **50m** elevation, 
**multi-antenna users**, and **radar-sensing targets**. We implemented the environment in *Mahool.py (HAPS_RIS_MIMO_ISAC)*.

### Main Components
**HAPS Transmitter** The UPA consists of $\( M = S_w \times S_l \)$ with a 64-antenna UPA at 20km altitude, operating at 2.5 GHz with 100 MHz bandwidth and up to 50dBW transmit power.  \
**Reconfigurable Intelligent Surface (RIS)** An RIS with 64 elements using 4-bit quantized phase shifts (16 levels) and mutual coupling modeling. Implements a corrected two-hop channel model 
where large-scale path loss and small-scale Rician fading are separated, ensuring physically meaningful RIS array gain. \
**Dynamic Environment** The users are placed closed to the RIS to maximize reflected path gain. Sensing targets (drones, in this case) follow random walk trajectories.  \
**Channel Model** We assume Rician fading (K=10) with separated path-loss amplitude $\\beta = \left(\frac{d_0}{d}\right)^{\alpha/2}\$ and unit-variance small-scale fading, where the path loss exponent $\( \alpha = 2.2 \)$.\ 
Channel estimation errors are modeled with additive noise $\(\sigma = 0.05 \)$.\
**ISAC Waveform Design** We assume the Null-space radar beamforming to ensure sensing waveforms do not interfere with communication users, with a fixed sensing/communication power split $(\eta=0.5$).\
**Baseline Schemes** To demonstrate the effectiveness of the proposed approach (MAPPO with RIS), we compared it with HAPS-ISAC (Genetic Algorithm, no RIS), and UAV-ISAC (MRT w/o RIS). 

---

## 2. Deep Reinforcement Learning Solution

The proposed DRL approach depends on multiple agents, with the main solution to the proposed MAPPO being the joint optimization of three cooperative sub-problems, each managed by a dedicated actor and critic.

### Problem Dimensions
- $\( M \)$: Antennas  
- $\( K \)$: Users  
- $\( N \)$: RIS elements  

### Agents

| Agent | Action Space | Responsibility |
|------|-------------|---------------|
| Comm | $\(2 \times M \times K\)$ (beamformer real + imaginary) | Communication beamformer design |
| Power | $\(K\)$ values | Per-user power allocation |
| RIS | $\(N\)$ values | RIS phase shift configuration |

Each agent has its own critic conditioned on the global state (CTDE paradigm), preventing gradient interference between sub-problems. The global state is broken down into agent-specific observations 
using *decompose_state()*. State decomposition handled by `decompose_state()`.

---

### Proposed algorithm highlights (`MAPPO.py`):
PPO is an on-policy DRL algorithm derived from the trust region policy optimization (TRPO) framework, offering lower computational complexity while maintaining comparable stability. 
The basic architecture of PPO is based on Actor and critic networks. Such that\
- **Actor:** Gaussian policy with tanh-bounded mean, learnable log-std, orthogonal initialization, and conservative initial std ($\approx 0.61)$.\
- **Critic:** Per-agent value function $V(s)$ using the global state with LayerNorm stabilization.\
- **Training:** Generalized Advantage Estimation (GAE, $\lambda =0.95)$, clipped surrogate objective ($\epsilon=0.15$), entropy regularisation, running reward normalization (EMA), and cosine annealing learning rate (LR) schedulers.\
- **Stability:** Gradient clipping $(max norm=0.5)$, input clamping (±10), action clamping (±1), and NaN/Inf guards throughout.
## Curriculum and schedules (`Train.py`):
- The SINR threshold is gradually increased from 5 dB to 15 dB during training to guide exploration.
-	The entropy coefficient decreases from 0.05 to 0.005, shifting from exploration to exploitation.
-	The warm-start blending for the ZF beamformer is phased out over the first zf_blend_episodes episodes.

---

## 3. Repository Structure
1.	`Mahool.py` (HAPS-RIS-MIMO-ISAC environment (channel model, reward, state)) \
2.	`MAPPO.py` (Multi-Agent PPO: ActorNetwork, AgentCritic, MAPPO_Agent, MAPPO_ISAC)\ 
3.	`Train.py` ( Training loop, baseline policies (GA, MRT), evaluation, figure generation)\
4.	`MAPPO_ISAC`/ # Output directory \
5.	agent_N.pt  (Model checkpoints (saved every 50 episodes)) \ 
6.	metrics_N.pt  (Training metrics snapshots)\
8.	Results  \
9.	README.md # code working description

---
## 4.	Installation
Clone the repository and install the dependencies:

pip <span style="color:blue">install</span>  -r requirements.txt\
**Requirements** (used packages for this project):\

torch==1.13.0\
numpy==1.17.0\
scipy==1.2.1\
matplotlib== 3.1.0\
tqdm==4.32.1\
gym==0.26.2

## 5.	Usage
Training the MAPPO agent
Please run the main file with default parameters (500 episodes, HAPS 64 antennas, RIS 8×8):\
**Key Findings:** \
•	*RIS gain*: The improved two-hop channel model produces approximately +8 to 13 dB reflected path gain over the direct channel with $N=64$ elements, providing the proposed scheme with a fundamental physical advantage.\
•	*Sensing performance*: Null-space radar beamforming attains higher minimum beampattern gain across all targets compared to non-cooperative baselines.\
•	*Communication*: MAPPO jointly optimizes beamforming and power allocation, achieving higher per-user rates and a better Jain fairness index than GA (which only operates on the direct channel) and MRT (which ignores inter-user interference).\
•	*Power scaling*: The performance gap between Proposed and HAPS-GA widens with increasing transmit power because of the super-linear $(N^²)$ RIS coherent combining gain.

---

## 6.	Results
Paper figures are generated automatically after training via `generate_paper_figures()`. To regenerate from a saved checkpoint.\
**Acknowledgment:**
This work is fully sponsored by university of Manitoba.\
Please cite “A. Iqbal and P. Hu, “Sensing and Communication for RIS-assisted HAPS-ISAC Systems with MIMO Beamforming,” Accepted for presentation in 2026, IEEE GLOBECOM Macau, China.” 


