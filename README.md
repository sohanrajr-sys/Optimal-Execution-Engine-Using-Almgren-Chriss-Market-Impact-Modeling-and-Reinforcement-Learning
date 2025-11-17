
---

# 📁 Project Structure

# 📘 Optimal Execution Engine  
### **Almgren–Chriss • Market Impact Modeling • Reinforcement Learning**

A full-stack quantitative trading research project implementing:

- **Almgren–Chriss optimal execution (closed-form solution)**  
- **Stochastic price simulation (GBM + OU processes)**  
- **Permanent & temporary market impact models**  
- **Reinforcement Learning execution agent (PPO / DQN)**  
- **Backtesting & evaluation suite**  

This project demonstrates **stochastic calculus, microstructure modeling, optimal control, and machine learning for trading** — ideal for quant interviews and research roles.

---

# 📌 Table of Contents  
1. [Motivation](#-motivation)  
2. [Key Concepts](#-key-concepts)  
3. [System Architecture](#-system-architecture)  
4. [Project Structure](#-project-structure)  
5. [Installation & Usage](#-installation--usage)  
6. [Detailed Module Explanation](#-detailed-module-explanation)  
7. [Evaluation & Metrics](#-evaluation--metrics)  
8. [Extensions & Future Work](#-extensions--future-work)  
9. [References](#-references)  

---

# 💡 Motivation

Executing a large order without moving the market is one of the most important problems in **quant trading**.  
A naive execution can lead to:

- high implementation shortfall  
- excessive market impact  
- liquidity-driven slippage  
- risk from price volatility  

This project builds an **Optimal Execution Engine** that:

1. models market dynamics using **SDEs**,  
2. analytically computes the **optimal execution trajectory** using **Almgren–Chriss**,  
3. trains an RL agent to **beat the analytical strategy** in richer market environments.

---

# 🧠 Key Concepts

This section covers the math and intuition behind the project.

---

## 1️⃣ Price Dynamics — Stochastic Differential Equations (SDEs)

The mid-price process \( S_t \) can follow:

### **Geometric Brownian Motion (GBM)**  
\[
dS_t = \mu S_t dt + \sigma S_t dW_t
\]

### **Arithmetic Brownian Motion (ABM)**  
\[
dS_t = \mu dt + \sigma dW_t
\]

**Why?**  
Short-horizon intraday prices behave nearly linearly (ABM) but longer intraday periods sometimes fit GBM.

---

## 2️⃣ Microstructure Alpha — Ornstein–Uhlenbeck (OU) Process

Models short-term mean-reversion in order flow:

\[
dX_t = -\theta X_t dt + \eta dW_t
\]

This provides RL with an exploitable **alpha signal**.

---

## 3️⃣ Market Impact Modeling

Trading affects the market in two ways:

### **Permanent Impact**
\[
S^{\text{perm}}_{t} = S_{t-1} + \gamma v_t
\]

### **Temporary Impact**
\[
C_{t} = S_t + \epsilon v_t
\]

Where:

- \( v_t \) = shares traded at time t  
- \( \gamma \) = permanent impact coefficient  
- \( \epsilon \) = temporary impact coefficient  

Modeling impact is essential for realistic execution.

---

## 4️⃣ Almgren–Chriss Optimal Execution

The classical solution solves:

\[
\min_{v_t} \quad \mathbb{E}[\text{Cost}] + \lambda \cdot \text{Risk}
\]

Closed-form optimal trading trajectory:

\[
x_t = X_0 \cdot \frac{\sinh(k(T - t))}{\sinh(kT)}
\]

Where:

- \( X_0 \): total shares  
- \( k \): derived from volatility & impact  
- \( T \): trading horizon  

This provides a **baseline** to compare RL vs optimal control.

---

## 5️⃣ Reinforcement Learning Execution Agent

The agent observes:

- mid-price  
- OU signal  
- volatility  
- liquidity  
- remaining inventory  
- remaining time  

Objective: minimize **implementation shortfall**.

### RL Algorithms Supported:
- **PPO (default)**
- **DQN**
- **A2C**

The agent often **outperforms Almgren–Chriss** in markets with stochastic alpha or liquidity shocks.

---

# 🏗️ System Architecture

