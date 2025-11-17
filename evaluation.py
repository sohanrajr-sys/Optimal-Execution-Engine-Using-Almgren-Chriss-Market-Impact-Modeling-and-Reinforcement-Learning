import numpy as np
from almgren_chriss import almgren_chriss_schedule

def evaluate_policy(env, model, n_paths=200):
    results = []
    for _ in range(n_paths):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
        results.append(info["final_value"])
    return np.array(results)

def evaluate_ac(sim, T_steps, Q, n_paths=200):
    ac_results = []
    for _ in range(n_paths):
        sim_path = sim.simulate_path(T_steps)
        v = almgren_chriss_schedule(
            Q=Q,
            T=T_steps * sim.dt,
            N=T_steps,
            sigma=sim.sigma,
            eta=sim.tmp,
            gamma=sim.perm
        )
        result = sim.simulate_path(T_steps, actions=v)
        ac_results.append(result["final_value"])
    return np.array(ac_results)
