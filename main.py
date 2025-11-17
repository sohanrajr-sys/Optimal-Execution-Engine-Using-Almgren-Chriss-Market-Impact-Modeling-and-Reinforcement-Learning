from market_simulator import MarketSimulator
from env import ExecutionEnv
from rl_agent import train_rl_agent
from evaluation import evaluate_policy, evaluate_ac
from utils import plot_results

def main():
    sim = MarketSimulator(
        init_price=100,
        sigma=0.002,
        ou_theta=10,
        ou_sigma=0.001,
        temporary_impact=1e-4,
        permanent_impact=1e-5,
        spread=0.02,
        seed=42
    )

    T_steps = 20
    Q = 1000

    env = ExecutionEnv(sim, T_steps=T_steps, target_shares=Q)

    print("Training RL agent...")
    model = train_rl_agent(env, total_steps=50000)

    print("Evaluating RL agent...")
    rl_vals = evaluate_policy(env, model, n_paths=300)

    print("Evaluating Almgren-Chriss baseline...")
    ac_vals = evaluate_ac(sim, T_steps=T_steps, Q=Q, n_paths=300)

    plot_results(rl_vals, ac_vals)

if __name__ == "__main__":
    main()
