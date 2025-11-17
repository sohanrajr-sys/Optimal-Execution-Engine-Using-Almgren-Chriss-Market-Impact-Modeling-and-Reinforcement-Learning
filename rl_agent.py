from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

def train_rl_agent(env, total_steps=50000):
    vec_env = DummyVecEnv([lambda: env])

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_epochs=10,
        batch_size=64,
        verbose=1,
        policy_kwargs=dict(net_arch=[dict(pi=[64,64], vf=[64,64])])
    )

    model.learn(total_timesteps=total_steps)
    return model
