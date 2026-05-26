from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from ai.env import GomokuEnv
from ai.model import GomokuCNN
from ai.rule_based_ai import RandomAI
from ai.rule_based_ai import WeakRuleAI
from ai.rule_based_ai import StrongRuleAI


policy_kwargs = dict(
    features_extractor_class=GomokuCNN,
    features_extractor_kwargs=dict(features_dim=256),
)


def train_stage(opponent, steps, save_path):
    env = GomokuEnv(opponent=opponent)

    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        learning_rate=3e-4,
        gamma=0.99,
        batch_size=256,
        n_steps=2048,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./logs/",
        policy_kwargs=policy_kwargs,
    )

    model.learn(total_timesteps=steps)

    model.save(save_path)


if __name__ == "__main__":
    train_stage(RandomAI(), 200000, "checkpoints/stage1")
    train_stage(WeakRuleAI(), 300000, "checkpoints/stage2")
    train_stage(StrongRuleAI(), 500000, "checkpoints/stage3")
