import os
from pathlib import Path

# Resolve project root by going up from current file
PROJECT_ROOT = Path(__file__).resolve().parents[0]  # Adjust the number based on depth

# NIST_ROOT_PATH=os.getenv("NIST_ROOT_PATH","/home/payman/Downloads/by_write")
NIST_ROOT_PATH="/mnt/lustre-grete/projects/LLMticketsummarization/payman/nist_data"
# BY_CLASS_LOG="/mnt/lustre-grete/projects/LLMticketsummarization/payman/nist_data/by_class/by_class_md5.log"
# BY_WRITE_LOG="/mnt/lustre-grete/projects/LLMticketsummarization/payman/nist_data/by_write/by_write_md5.log"
# DIGITS_HASHES="/mnt/lustre-grete/projects/LLMticketsummarization/payman/nist_data/by_write/digits_hashes"
# JSON_DIGIT_LABELS="/mnt/lustre-grete/projects/LLMticketsummarization/payman/nist_data/by_write/digits_labels.json"
NUMBER_EPOCHS=3
LEARNING_RATE=1e-3
WEIGHT_DECAY=1e-4
CLIENTS_NUMBER_EPOCHS=10
BATCH_SIZE=16

BEST_EWC_LAMBDA=8.0
BEST_KD_T=8.0
BEST_KD_ALPHA=0.95
BEST_PROX_LAMBDA=0.9
BEST_LOGIT_LAMBDA=0.1
BEST_FEATURE_BETA=0.1
