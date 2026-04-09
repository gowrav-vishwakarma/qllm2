"""Auto-generated training configuration."""

config = {
    "project_name": "v7_experiment",
    "output_dir": "./experiments/v7_experiment",
    "checkpoint_dir": "./experiments/v7_experiment/checkpoints",
    "log_dir": "./experiments/v7_experiment/logs",
    "dataset": "tinystories",
    "tokenizer": "gpt2",
    "seq_len": 512,
    "batch_size": 32,
    "optimizer": "AdamW",
    "lr": 0.0003,
    "weight_decay": 0.1,
    "betas": [
        0.9,
        0.95
    ],
    "scheduler": "cosine",
    "warmup_steps": 500,
    "epochs": 50,
    "grad_clip": 1,
    "amp": true,
    "compile": false,
    "grad_accumulation": 1
}
