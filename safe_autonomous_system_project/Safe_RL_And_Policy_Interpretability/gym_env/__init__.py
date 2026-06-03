import safety_gymnasium
from stable_baselines3.common.env_util import make_atari_env
#from stable_baselines3.common.monitor import Monitor
from monitor import Monitor
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

def make_env(level, num_environment): #args, test_viper=False):
    # load in environment
    return DummyVecEnv([lambda: Monitor(safety_gymnasium.make(f"SafetyCarGoal{level}-v0")) for _ in range(num_environment)])
