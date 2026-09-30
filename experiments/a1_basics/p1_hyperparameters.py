from modal_train import launch_training_jobs
from train import TrainConfig

WARMUP_PERCENTS_NOT = (0.04, 0.08, 0.12, 0.16, 0.2)
WARMUP_PERCENTS = (0.01, 0.02, 0.04, 0.08, 0.16)
BATCH_SIZES = (16, 32, 64, 128, 256)
WEIGHT_DECAYS = (0.0025, 0.005, 0.01, 0.02, 0.04)
LEARNING_RATES = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)
SCHEDULER = ("linear", "cos", "const", "wsd0.1")

warmup = 0.02
batch_size = 32
lr = 0.003 # 0.001
wd = 0.0025

RUNS = [
    TrainConfig(
        warmup_percent=0.02,
        batch_size=32,
        weight_decay=0.0025,
        learning_rate=0.001,
        lr_schedule="wsd0.2",
    )
    # for warmup_percent in WARMUP_PERCENTS
    # for batch_size in BATCH_SIZES
    # for weight_decay in WEIGHT_DECAYS
    # for learning_rate in LEARNING_RATES
    # for batch_size, learning_rate in zip(BATCH_SIZES[:3], LEARNING_RATES[:3])
    # for schedule in SCHEDULER
]


# TODO: Design individual sweeps, paired sweeps, and scheduler comparisons.


def main():
    launch_training_jobs(RUNS)


if __name__ == "__main__":
    main()
