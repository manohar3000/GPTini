"""
Configuration settings for the language model.

This module contains all hyperparameters and settings used throughout
the training and inference processes.
"""

import torch

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# File paths
INPUT_FILE = "input.txt"
MODEL_SAVE_PATH = "model_checkpoints"

# Model architecture - Optimized for RTX 4050 GPU
VOCAB_SIZE = None  # Will be set after processing the input file
N_EMBED = 384      # Reduced for RTX 4050
N_HEADS = 6        # Reduced for RTX 4050
N_LAYERS = 6       # Reduced for RTX 4050
HEAD_SIZE = N_EMBED // N_HEADS
BLOCK_SIZE = 512   # Reduced for RTX 4050

# Training parameters
BATCH_SIZE = 16    # Reduced for RTX 4050
LEARNING_RATE = 3e-4
MAX_ITERATIONS = 10000
EVAL_INTERVAL = 1000
TRAIN_SPLIT = 0.9  # 90% training, 10% validation

# Generation parameters
DEFAULT_GEN_LENGTH = 100
TEMPERATURE = 1.0
TOP_K = 0
TOP_P = 0.9

# Checkpointing
SAVE_CHECKPOINT_INTERVAL = 1000
CHECKPOINT_TO_LOAD = None  # Set to path of checkpoint if resuming training