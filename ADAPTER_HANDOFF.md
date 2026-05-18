# Retrieval Adapter Handoff

Run these from the repo root on the FontDiffuser machine.

## 0. Environment

See:

```text
ENV_SETUP_ADAPTER.md
environment-fontdiffuser-cu117.yml
```

## 1. Smoke Tests

```powershell
python tests/test_retrieval_adapter.py
python tests/test_adapter_integration.py
python scripts/inspect_retrieval_pack.py --path-map /d/htt/data=D:/htt/data
```

If the Wang Xizhi image root is different, change the right side of
`--path-map`.

By default, `inspect_retrieval_pack.py` checks every case in `case_manifest.csv`.
To inspect a subset, pass comma-separated targets:

```powershell
python scripts/inspect_retrieval_pack.py --targets 璨,霆 --path-map /d/htt/data=D:/htt/data
```

Expected:

- `test_retrieval_adapter: OK`
- `test_adapter_integration: OK`
- `target_gt_leakage_count: 0`
- `missing_valid_ref_images: 0` if the image root is mapped correctly

## 2. Attach Adapter In Code

```python
from src.modules.retrieval_adapter import (
    attach_retrieval_adapter,
    freeze_backbone_train_adapter,
)

adapter = attach_retrieval_adapter(model.unet, up_block_index=2)
freeze_backbone_train_adapter(model, up_block_index=2)
```

## 3. Model Forward Retrieval Inputs

`FontDiffuserModel.forward` now accepts:

```python
retrieval_inputs = {
    "ref_images": ref_images,          # [B, 5, 3, 96, 96]
    "slot_ids": slot_ids,              # [B, 5]
    "role_ids": role_ids,              # [B, 5]
    "target_struct": target_struct,    # [B]
    "mask": mask,                      # [B, 5]
}
```

The model runs `ref_images` through the frozen `content_encoder` and passes
`[B, 5, 64, 48, 48]` ref features to `UNet.up_blocks[2]`.

## 4. Current Limitation

DPM sampling has not been wired for retrieval yet. Do training/tiny-overfit
checks through `FontDiffuserModel.forward` first. DPM classifier-free retrieval
conditioning needs a separate design decision for uncond vs cond retrieval
inputs.
