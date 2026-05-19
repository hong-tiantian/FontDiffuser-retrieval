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

## 5. Tiny Overfit

Create a small CSV like:

```text
target_char,content_image_path,style_image_path,target_image_path
璨,D:/.../content/璨.jpg,D:/.../style/ref.jpg,D:/.../gt/璨.jpg
```

Then run:

```powershell
python scripts/train_adapter_tiny_overfit.py `
  --ckpt-dir D:/path/to/fontdiffuser_ckpt `
  --tiny-manifest D:/path/to/tiny_overfit_manifest.csv `
  --plan-a-root D:/htt/callirag `
  --path-map /d/htt/data=D:/htt/data `
  --steps 100 `
  --adapter-scale 50 `
  --direct-scale 1 `
  --device cuda:0 `
  --save-checkpoint
```

For the 15-case manifest run, the equivalent entrypoint is:

```powershell
python scripts/train_adapter_manifest_cases.py `
  --ckpt-dir D:/path/to/fontdiffuser_ckpt `
  --manifest D:/htt/FontDiffuser-retrieval/examples/tiny_overfit_manifest.csv `
  --plan-a-root D:/htt/callirag `
  --steps 1000 `
  --adapter-scale 10 `
  --offset-scale 0 `
  --direct-scale 1 `
  --device cuda:0 `
  --save-checkpoint `
  --output-dir outputs/adapter_manifest_direct_s10
```

By default the tiny-overfit script fixes the diffusion noise and timestep so the
loss curve is interpretable. Add `--resample-noise` only when you want a noisier
training-liveness check.

Watch:

```text
alpha
loss
final_delta_abs_mean
alpha_zero_mean_abs_diff
shuffle_refs_mean_abs_diff
zero_refs_mean_abs_diff
random_refs_mean_abs_diff
```

For C0, the important signal is that `alpha` moves away from zero and changing
refs produces a nonzero output difference after training.

Use `--adapter-scale` only as a diagnostic. If `--adapter-scale 50` makes loss
move and ref ablation grow, the adapter path is valid but too weak at normal
scale.

Use `--direct-scale` to test the second injection path. The default `0` keeps
the original offset-path-only behavior. A useful diagnostic command is
`--adapter-scale 50 --offset-scale 0 --direct-scale 1`, which isolates direct
skip-feature injection.

When `--save-checkpoint` is used, two files are written:

```text
retrieval_adapter.pth  # adapter only, kept for debugging
retrieval_bundle.pth   # adapter + direct 1x1 projections + scale settings
```

Use `retrieval_bundle.pth` for direct-skip runs.
