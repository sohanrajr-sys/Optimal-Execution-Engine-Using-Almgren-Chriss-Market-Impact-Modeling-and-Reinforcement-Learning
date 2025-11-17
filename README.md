📘 Optimal Execution Engine
Almgren–Chriss • Market Impact Modeling • Reinforcement Learning

A full-stack quantitative trading research project implementing:

Almgren–Chriss optimal execution (closed-form solution)

Stochastic price simulation (GBM + OU processes)

Permanent & temporary market impact models

Reinforcement Learning execution agent (PPO / DQN)

Backtesting & evaluation suite

This project demonstrates stochastic calculus, microstructure modeling, optimal control, and machine learning for trading — ideal for quant interviews and research roles.

📌 Table of Contents

Motivation

Key Concepts

System Architecture

Project Structure

Installation & Usage

Detailed Module Explanation

Evaluation & Metrics

Extensions & Future Work

References

💡 Motivation

Executing a large order without moving the market is one of the most important problems in quant trading.
A naive execution can lead to:

high implementation shortfall

excessive market impact

liquidity-driven slippage

risk from price volatility

This project builds an Optimal Execution Engine that:

models market dynamics using SDEs,

analytically computes the optimal execution trajectory using Almgren–Chriss,

trains an RL agent to beat the analytical strategy in richer market environments.

🧠 Key Concepts

This section covers the math and intuition behind the project.

1️⃣ Price Dynamics — Stochastic Differential Equations (SDEs)

The mid-price process 
𝑆
𝑡
S
t
	​

 can follow:

Geometric Brownian Motion (GBM)
𝑑
𝑆
𝑡
=
𝜇
𝑆
𝑡
𝑑
𝑡
+
𝜎
𝑆
𝑡
𝑑
𝑊
𝑡
dS
t
	​

=μS
t
	​

dt+σS
t
	​

dW
t
	​

Arithmetic Brownian Motion (ABM)
𝑑
𝑆
𝑡
=
𝜇
𝑑
𝑡
+
𝜎
𝑑
𝑊
𝑡
dS
t
	​

=μdt+σdW
t
	​


Why?
Short-horizon intraday prices behave nearly linearly (ABM) but longer intraday periods sometimes fit GBM.

2️⃣ Microstructure Alpha — Ornstein–Uhlenbeck (OU) Process

Models short-term mean-reversion in order flow:

𝑑
𝑋
𝑡
=
−
𝜃
𝑋
𝑡
𝑑
𝑡
+
𝜂
𝑑
𝑊
𝑡
dX
t
	​

=−θX
t
	​

dt+ηdW
t
	​


This provides RL with an exploitable alpha signal.

3️⃣ Market Impact Modeling

Trading affects the market in two ways:

Permanent Impact
𝑆
𝑡
perm
=
𝑆
𝑡
−
1
+
𝛾
𝑣
𝑡
S
t
perm
	​

=S
t−1
	​

+γv
t
	​

Temporary Impact
𝐶
𝑡
=
𝑆
𝑡
+
𝜖
𝑣
𝑡
C
t
	​

=S
t
	​

+ϵv
t
	​


Where:

𝑣
𝑡
v
t
	​

 = shares traded at time t

𝛾
γ = permanent impact coefficient

𝜖
ϵ = temporary impact coefficient

Modeling impact is essential for realistic execution.

4️⃣ Almgren–Chriss Optimal Execution

The classical solution solves:

min
⁡
𝑣
𝑡
𝐸
[
Cost
]
+
𝜆
⋅
Risk
v
t
	​

min
	​

E[Cost]+λ⋅Risk

Closed-form optimal trading trajectory:

𝑥
𝑡
=
𝑋
0
⋅
sinh
⁡
(
𝑘
(
𝑇
−
𝑡
)
)
sinh
⁡
(
𝑘
𝑇
)
x
t
	​

=X
0
	​

⋅
sinh(kT)
sinh(k(T−t))
	​


Where:

𝑋
0
X
0
	​

: total shares

𝑘
k: derived from volatility & impact

𝑇
T: trading horizon

This provides a baseline to compare RL vs optimal control.

5️⃣ Reinforcement Learning Execution Agent

The agent observes:

mid-price

OU signal

volatility

liquidity

remaining inventory

remaining time

Objective: minimize implementation shortfall.

RL Algorithms Supported:

PPO (default)

DQN

A2C

The agent often outperforms Almgren–Chriss in markets with stochastic alpha or liquidity shocks.

🏗️ System Architecture
                ┌──────────────────────────────┐
                │       Market Simulator        │
                │  (GBM + OU + Impact + Liquidity)
                └──────────────┬──────────────┘
                               │
                ┌──────────────▼───────────────┐
                │   Execution Environment       │
                │   (Gym-style RL environment)  │
                └──────────────┬───────────────┘
                               │
    ┌──────────────────────────┼──────────────────────────┐
    │                          │                          │
┌───▼──────────┐      ┌────────▼──────────┐      ┌────────▼─────────┐
│  TWAP/VWAP   │      │ Almgren–Chriss    │      │ RL Agent (PPO)   │
└──────────────┘      └───────────────────┘      └───────────────────┘
                               │
                    ┌──────────▼─────────┐
                    │ Backtesting Engine │
                    └──────────┬─────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Metrics & Plots   │
                    └──────────────────────┘

📁 Project Structure
Optimal-Execution-Engine/
│
├── env/                # Market simulator + Gym environment
├── market_simulator/   # SDEs, impact models, liquidity models
├── rl_agent/           # PPO / DQN RL agents
├── evaluation/         # Backtesting, metrics, plotting
├── utils/              # Helper functions
├── main.py             # Train + evaluate pipeline
└── README.md

⚙️ Installation & Usage
1. Clone the repo
git clone https://github.com/sohanrajr-sys/Optimal-Execution-Engine-Using-Almgren-Chriss-Market-Impact-Modeling-and-Reinforcement-Learning
cd Optimal-Execution-Engine-Using-Almgren-Chriss-Market-Impact-Modeling-and-Reinforcement-Learning

2. Install requirements
pip install -r requirements.txt

3. Run baseline + RL training
python main.py

4. View results

Plots and logs will appear in:

/results

📘 Detailed Module Explanation
📂 market_simulator/

Implements:

GBM / ABM price SDE

OU alpha signal

temporary & permanent impact

stochastic liquidity

Monte Carlo simulation

Generates realistic trajectories for execution.

📂 env/

A Gym-like environment where the RL agent interacts with the market.

State includes:

price

alpha

remaining shares

remaining time

market depth

Actions: number of shares to execute.

Reward = negative execution cost.

📂 rl_agent/

Implementation of:

PPO

DQN

A2C

with:

policy networks

replay buffers

training loops

exploration strategies

📂 evaluation/

Measures:

implementation shortfall

realized cost

variance of cost

PnL distribution

Sharpe-style risk-adjusted measures

Also provides plotting utilities for:

execution paths

price trajectories

policy comparison

📊 Evaluation & Metrics

Metrics include:

1. Implementation Shortfall (IS)
IS
=
∑
(
𝑝
𝑡
−
𝑝
0
)
𝑣
𝑡
IS=∑(p
t
	​

−p
0
	​

)v
t
	​

2. Trading Cost Decomposition

Temporary impact cost

Permanent impact cost

Drift cost

Volatility risk

3. Strategy Comparison

TWAP

VWAP

Almgren–Chriss

RL agent

4. Monte Carlo Backtesting

Thousands of simulated paths for robust statistics.

🚀 Extensions & Future Work

You can extend this project into deeper quant research by adding:

Heston volatility model

queue-reactive limit order book simulator

Jump-diffusion price process

Deep RL with attention networks

Multi-asset execution

Adversarial market maker simulation

📚 References

Almgren, R., & Chriss, N. (2000). Optimal execution of portfolio transactions.

Gatheral, J. The Volatility Surface.

Cartea, Á., Jaimungal, S., & Penalva, J. Algorithmic and High-Frequency Trading.

Bertsimas & Lo. Optimal control of execution costs.

Sutton & Barto. Reinforcement Learning: An Introduction.
