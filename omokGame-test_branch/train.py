# train.py (중급 대응 강화 최종 버전)
import os

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from stable_baselines3.common.callbacks import CheckpointCallback

from ai.env import GomokuEnv
from ai.model import GomokuCNN

from ai.rule_based_ai import RandomAI
from ai.rule_based_ai import WeakRuleAI
from ai.rule_based_ai import StrongRuleAI


# =========================================
# 저장 경로
# =========================================

SAVE_DIR = "/content/drive/MyDrive/omok_checkpoints"

os.makedirs(SAVE_DIR, exist_ok=True)

LATEST_MODEL = f"{SAVE_DIR}/latest_model.zip"


# =========================================
# CNN Policy 설정
# =========================================

policy_kwargs = dict(
    features_extractor_class=GomokuCNN,
    features_extractor_kwargs=dict(features_dim=256),
)


# =========================================
# 체크포인트 저장
# =========================================

checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path=SAVE_DIR,
    name_prefix="gomoku_checkpoint",
)


# =========================================
# 학습 함수
# =========================================

def train_stage(opponent, total_timesteps):

    env = GomokuEnv(opponent=opponent)

    # =====================================
    # 이어서 학습
    # =====================================

    if os.path.exists(LATEST_MODEL):

        print("기존 모델 불러오는 중...")

        model = MaskablePPO.load(
            LATEST_MODEL,
            env=env
        )

    else:

        print("새 모델 생성")

        model = MaskablePPO(
            MaskableActorCriticPolicy,
            env,

            learning_rate=3e-4,
            gamma=0.99,

            batch_size=256,
            n_steps=2048,

            clip_range=0.2,
            ent_coef=0.01,

            verbose=1,

            tensorboard_log="./logs/",

            policy_kwargs=policy_kwargs,

            device="auto"
        )

    # =====================================
    # 학습
    # =====================================

    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
        reset_num_timesteps=False,
    )

    # =====================================
    # 최종 저장
    # =====================================

    model.save(LATEST_MODEL)

    print("모델 저장 완료")
    print(LATEST_MODEL)


# =========================================
# 메인 학습 루프
# =========================================

if __name__ == "__main__":

    print("========== Stage 1 ==========")

    train_stage(
        RandomAI(),
        total_timesteps=100000
    )

    print("========== Stage 2 ==========")

    train_stage(
        WeakRuleAI(),
        total_timesteps=300000
    )

    print("========== Stage 3 ==========")

    train_stage(
        StrongRuleAI(),
        total_timesteps=800000
    )

    print("========== 학습 완료 ==========")
