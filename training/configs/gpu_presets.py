# GPU-specific training presets for cognitive-core
# Each preset targets ~80% VRAM utilization for the 1B model.

PRESETS = {
    "t4": {
        "max_len": 8192,
        "train_cap": 8192,
        "accum": 24,
        "neftune": 5,
        "lr": 1e-5,
        "bsz": 1,
        "notes": "T4 (16GB): 8K context, effective batch 24"
    },
    "l4": {
        "max_len": 16384,
        "train_cap": 16384,
        "accum": 24,
        "neftune": 5,
        "lr": 1e-5,
        "bsz": 1,
        "notes": "L4 (24GB): 16K context, effective batch 24"
    },
    "a10g": {
        "max_len": 4096,
        "train_cap": 4096,
        "accum": 6,
        "neftune": 5,
        "lr": 1e-5,
        "bsz": 4,
        "notes": "A10G (24GB): 4K context, batch 4, fastest"
    },
    "l40s": {
        "max_len": 24576,
        "train_cap": 24576,
        "accum": 12,
        "neftune": 5,
        "lr": 1e-5,
        "bsz": 2,
        "notes": "L40S (46GB): full context, batch 2, fast"
    },
    "a10g-full": {
        "max_len": 24576,
        "train_cap": 24576,
        "accum": 24,
        "neftune": 5,
        "lr": 1e-5,
        "bsz": 1,
        "notes": "A10G (24GB): full 24K context, tight on VRAM"
    },
    "a100-40": {
        "max_len": 24576,
        "train_cap": 24576,
        "accum": 24,
        "neftune": 5,
        "lr": 1e-5,
        "bsz": 2,
        "notes": "A100-40GB: full context, batch 2"
    },
    "a100-80": {
        "max_len": 24576,
        "train_cap": 24576,
        "accum": 12,
        "neftune": 5,
        "lr": 1e-5,
        "bsz": 4,
        "notes": "A100-80GB: full context, batch 4, fastest"
    },
}

GPU_MAP = {
    "Tesla T4": "t4",
    "NVIDIA T4": "t4",
    "NVIDIA L4": "l4",
    "NVIDIA L40S": "l40s",
    "NVIDIA A10G": "a10g",
    "NVIDIA A100-SXM4-40GB": "a100-40",
    "NVIDIA A100 80GB PCIe": "a100-80",
    "NVIDIA A100-SXM4-80GB": "a100-80",
}

DEFAULT = "a10g"
