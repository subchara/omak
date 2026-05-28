# self_play.py (강화 버전)

import os

from sb3_contrib import MaskablePPO

from ai.env import GomokuEnv
from ai.hybrid_agent import HybridAgent


SAVE_DIR = "/content/drive/MyDrive/omok_checkpoints"

LATEST_MODEL = f"{SAVE_DIR}/latest_model.zip"


def self_play():

    previous = LATEST_MODEL

    for generation in range(10):

        print(f"========== Self Play Generation {generation} ==========")

        opponent = HybridAgent(previous)

        env = GomokuEnv(opponent=opponent)

        model = MaskablePPO.load(
            previous,
            env=env
        )

        model.learn(
            total_timesteps=300000,
            progress_bar=True,
            reset_num_timesteps=False,
        )

        save_path = f"{SAVE_DIR}/selfplay_{generation}.zip"

        model.save(save_path)

        previous = save_path

        print(f"저장 완료: {save_path}")


if __name__ == "__main__":
    self_play()

