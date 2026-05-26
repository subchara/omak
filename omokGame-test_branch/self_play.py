from sb3_contrib import MaskablePPO

from ai.env import GomokuEnv
from ai.hybrid_agent import HybridAgent


def self_play():
    previous = "checkpoints/stage3.zip"

    for generation in range(5):
        opponent = HybridAgent(previous)

        env = GomokuEnv(opponent=opponent)

        model = MaskablePPO.load(previous, env=env)

        model.learn(total_timesteps=300000)

        save_path = f"checkpoints/selfplay_{generation}.zip"

        model.save(save_path)

        previous = save_path


if __name__ == "__main__":
    self_play()
