import os

class Config:
    TEXT_DATASET_NAME="wikipedia"
    TEXT_DATASET_VERSION="20220301.simple"
    NUM_SAMPLES=30_000
    BATCH_SIZE = 500
    NUM_WORKERS = 0    # os.cpu_count() ERROR with my machine
    MAX_MEMORY_TOKENS = 2000
    MAX_MEMORY_SIZE = 5