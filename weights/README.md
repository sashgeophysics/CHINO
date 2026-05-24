# Pre-trained Weights

Pre-trained CHINO v1.0 weights are hosted on HuggingFace:

**https://huggingface.co/hier-majumder/CHINO**

## Download

```python
from huggingface_hub import hf_hub_download

ckpt_path = hf_hub_download(
    repo_id="hier-majumder/CHINO",
    filename="chino_best.pt",
    local_dir="checkpoints/",
)
```

Or via the command line:

```bash
pip install huggingface_hub
huggingface-cli download hier-majumder/CHINO chino_best.pt --local-dir checkpoints/
```

## Checkpoint details

| File | Epochs | L2 (t=2.0) | Description |
|---|---|---|---|
| `chino_best.pt` | 550 | 0.25 | Best validation loss checkpoint |
| `chino_final.pt` | 550 | — | Final epoch checkpoint |

## Training configuration

The weights were produced using the configuration in `config.yaml` at the
repository root. Training was conducted in two phases:

- Phase 1 (epochs 1-300): standard curriculum as defined in `config.yaml`
- Phase 2 (epochs 301-550): resumed with `w_phys=0.3`, gradient
  checkpointing disabled, using `scripts/resume.py`

Hardware: NVIDIA RTX PRO 6000 Blackwell (95 GB HBM3)
Total training time: approximately 13 hours
