"""
Split existing HuggingFace train-only datasets into train/validation/test splits
and push back to the hub.

Takes 500 samples from each config for validation and 500 for test.
The remainder stays as train.

Usage:
    python split_and_push.py
"""

import os
from datasets import load_dataset, DatasetDict

HUB_REPO = "RaghaRao314159/transcription-dataset"
CONFIGS = ["librispeech", "mls_eng"]
VAL_SIZE = 500
TEST_SIZE = 500
SEED = 42


def split_and_push(config_name: str):
    print(f"\n{'='*60}")
    print(f"Processing config: {config_name}")
    print(f"{'='*60}")

    # Load the full train split
    ds = load_dataset(HUB_REPO, config_name, split="train")
    print(f"Loaded {len(ds)} samples from train split")

    total_holdout = VAL_SIZE + TEST_SIZE
    assert len(ds) > total_holdout, (
        f"Dataset {config_name} has only {len(ds)} samples, "
        f"need at least {total_holdout} for val+test"
    )

    # Shuffle deterministically then carve out val/test from the end
    ds = ds.shuffle(seed=SEED)

    test_ds = ds.select(range(TEST_SIZE))
    val_ds = ds.select(range(TEST_SIZE, TEST_SIZE + VAL_SIZE))
    train_ds = ds.select(range(TEST_SIZE + VAL_SIZE, len(ds)))

    print(f"  train: {len(train_ds)}")
    print(f"  validation: {len(val_ds)}")
    print(f"  test: {len(test_ds)}")

    dataset_dict = DatasetDict({
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds,
    })

    print(f"Pushing {config_name} to {HUB_REPO} ...")
    dataset_dict.push_to_hub(HUB_REPO, config_name=config_name)
    print(f"Done pushing {config_name}!")


def main():
    for config in CONFIGS:
        split_and_push(config)
    print(f"\nAll configs pushed to {HUB_REPO} with train/validation/test splits.")


if __name__ == "__main__":
    main()
