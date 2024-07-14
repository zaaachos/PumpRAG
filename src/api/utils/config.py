import os


class Config:
    TEXT_DATASET_NAME = "wikipedia"
    TEXT_DATASET_VERSION = "20220301.simple"
    NUM_SAMPLES = 30_000
    BATCH_SIZE = 100
    NUM_WORKERS = 0  # os.cpu_count() ERROR with my machine
    MAX_MEMORY_TOKENS = 2000
    MAX_MEMORY_SIZE = 5
    GYM_DATASET_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.getcwd())), "assets/megaGymDataset.txt"
    )
