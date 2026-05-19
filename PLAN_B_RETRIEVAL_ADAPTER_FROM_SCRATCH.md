# Plan B2: Retrieval-Conditioned Adapter From Clean FontDiffuser

## 0. Position

This repository is treated as the clean FontDiffuser working copy. The earlier
`callirag/adapter_dev` implementation is reference material only; do not copy it
blindly. All new adapter work happens in this repo.

Plan A retrieval assets remain external and read-only:

```text
D:/00_project/callirag/retrieval_data_prepare/
  bank/
    wxz_bank.json
    bank_enriched.json
    non_trivial_components.json
  outputs/
    case_manifest.csv
    sim_layer.json
    retrieval_topk_manifest.csv
    train_retrieval_cache.jsonl
    train_retrieval_cache_raw.jsonl
  pipeline/
    6retrieve.py
```

The goal of this phase is not to improve visual beauty. The goal is to prove
that a frozen FontDiffuser can accept retrieval-conditioned structural features
through a small trainable adapter, and that the output is sensitive to the
retrieved refs.

## 1. Research Question

Can a trainable adapter, injected into FontDiffuser's high-resolution offset
path while the original model is frozen, use 5-slot structure-aware retrieval
refs to improve or alter rare/complex character structure?

This phase validates technical viability:

- bit-exact baseline preservation when retrieval is disabled
- adapter-only trainability
- nonzero adapter gradients
- output sensitivity to retrieved refs and masks
- no target GT leakage from Plan A refs

Do not claim general performance improvement until the ablation stage supports
it.

## 2. Scope

Initial scope:

- single character generation
- Wang Xizhi only
- 15 Plan A cases from `case_manifest.csv`
- traditional/simplified status follows existing Plan A metadata
- FontDiffuser backbone frozen
- only retrieval adapter parameters trainable
- primary injection: `UNet.up_blocks[2]`, `StyleRSIUpBlock2D`
- no mid-block injection in the first implementation
- no new retrieval ranking work in this repo

Out of scope for the first implementation:

- rewriting Plan A retrieval
- learned retriever
- multi-calligrapher training
- page-level generation
- rarity-tiered injection
- component aux loss
- DINO/OCR metric engineering beyond simple sanity scripts

## 3. Key Correction From Earlier A-Phase

The earlier locked decision said:

```text
out_proj zero-init + alpha init 0
```

This must be changed. If both `out_proj` and `alpha` are initialized to zero,
the adapter can enter a double-zero gradient trap:

```text
delta = alpha * out_proj(attention_output)
alpha = 0
out_proj(...) = 0
=> grad(alpha) = 0
=> grad(out_proj) = 0
```

New rule:

```text
alpha init = 0
out_proj weight = normal init, bias = 0
```

This preserves bit-exact output at initialization because `alpha=0`, while
keeping a valid gradient path into `alpha`. Once `alpha` moves away from zero,
the rest of the adapter can learn.

One additional implementation constraint is now explicit: ref feature maps are
accepted at `48 x 48`, but the adapter pools each ref slot to a smaller token
grid before cross-attention. The default is `12 x 12` tokens per slot. Full
query-to-5x48x48 attention is possible in principle, but it is unnecessarily
expensive for the first engineering validation and can make tiny overfit runs
memory-bound.

## 4. Data Contract From Plan A

The adapter consumes 5 retrieval slots per target:

```text
2 anchor slots
3 coverage slots
```

Each slot must carry:

```text
ref_image_path
bank_id
ref_char
role              # anchor / coverage / empty
matched_comp
slot_id
role_id
valid_mask
```

Model-facing tensor contract:

```python
retrieval_inputs = {
    "ref_images": Tensor[B, 5, 3, 96, 96],
    "slot_ids": Tensor[B, 5],          # long, vocab size 37
    "role_ids": Tensor[B, 5],          # long, anchor/coverage/empty
    "target_struct": Tensor[B],        # long, layout/structure id
    "mask": Tensor[B, 5],              # bool, at least one valid slot per row
    "meta": Optional[list[dict]],      # analysis only, not used by model
}
```

UNet-facing tensor contract after frozen `content_encoder` extraction:

```python
retrieval_unet_inputs = {
    "refs": Tensor[B, 5, 64, 48, 48],
    "slot_ids": Tensor[B, 5],
    "role_ids": Tensor[B, 5],
    "target_struct": Tensor[B],
    "mask": Tensor[B, 5],
}
```

The feature level `64 x 48 x 48` matches `StyleRSIUpBlock2D` at
`up_blocks[2]`, where `style_content_feat = style_structure_features[-4]`.

Invariant:

```text
Every batch item must have at least one valid slot.
```

If Plan A retrieval cannot find coverage refs, it must fall back to anchors
before training data is packed.

## 5. Architecture

### 5.1 New Module

Add:

```text
src/modules/retrieval_adapter.py
```

Class:

```python
class RetrievalAdapter(nn.Module):
    def forward(
        self,
        h_q,
        refs,
        slot_ids,
        role_ids,
        target_struct,
        mask,
        return_gate=False,
    ):
        ...
```

Input/output:

```text
h_q:   [B, 64, 48, 48]
refs:  [B, 5, 64, 48, 48]
delta: [B, 64, 48, 48]
```

Adapter design:

- flatten `h_q` into query tokens
- flatten all valid ref slots into key/value tokens
- add metadata bias to ref tokens:
  - slot embedding, vocab 37
  - role embedding, vocab 3
  - target structure embedding, vocab 12
- masked attention over ref tokens
- pre-softmax mask with additive `-1e4`
- no post-softmax renormalization
- output projection
- scalar gate `alpha`

Initialization:

```text
alpha = 0
out_proj.weight = N(0, 0.02)
out_proj.bias = 0
```

### 5.2 Injection Point

Modify:

```text
src/modules/unet_blocks.py
```

Target class:

```text
StyleRSIUpBlock2D
```

Injection location:

```python
style_content_feat = style_structure_features[-self.upblock_index - 2]
style_content_feat = style_content_feat + delta_h
```

The adapter call happens once before the loop over `sc_interpreter_offsets`,
not inside every layer.

Only `up_blocks[2]` receives an adapter. Do not add mid-block injection in the
first implementation.

### 5.3 UNet Routing

Modify:

```text
src/modules/unet.py
```

Add optional argument:

```python
retrieval_inputs: Optional[dict] = None
```

When iterating up blocks, pass retrieval inputs only when:

```python
i == 2 and hasattr(upsample_block, "retrieval_adapter")
```

All other blocks must be bit-exact unchanged.

### 5.4 Model-Level Ref Feature Extraction

Modify:

```text
src/model.py
```

Extend `FontDiffuserModel.forward` to accept optional `retrieval_inputs`.

If present:

1. read `ref_images` as `[B, 5, 3, 96, 96]`
2. flatten to `[B*5, 3, 96, 96]`
3. run frozen `content_encoder`
4. take the `64 x 48 x 48` residual feature
5. reshape to `[B, 5, 64, 48, 48]`
6. pass `retrieval_unet_inputs` to `self.unet`

Do not use `style_encoder` for retrieved refs.

For phase C inference, also extend `FontDiffuserModelDPM.forward` and
`src/dpm_solver/pipeline_dpm_solver.py`, but this can wait until adapter
training has passed C0.

## 6. Freeze Policy

Training mode must enforce:

```text
requires_grad=False for:
  unet original parameters
  style_encoder
  content_encoder

requires_grad=True only for:
  unet.up_blocks[2].retrieval_adapter.*
```

The frozen `content_encoder` may run forward on refs, but its parameters must
not receive gradients. Prefer `torch.no_grad()` around reference feature
extraction unless later evidence shows ref feature gradients are needed.

## 7. Source Hygiene

Because this repo is the clean working copy, use git as the baseline.

Rules:

- New files do not need marker comments.
- Every edit inside an existing FontDiffuser source file must be wrapped:

```python
# --- CALLI-RAG BEGIN: short reason ---
...
# --- CALLI-RAG END ---
```

- Maintain:

```text
CALLIRAG_CHANGELOG.md
```

Each entry must include:

```text
file
approx line
change type
phase
reason
verification
```

- Before moving phases, inspect:

```powershell
git diff -- src src/modules dataset configs train.py sample.py
```

Every diff hunk in existing source files must have a marker and changelog
entry.

## 8. Implementation Phases

### P0: Clean Recon

Goal: record the exact baseline architecture before edits.

Tasks:

- inspect `src/model.py`
- inspect `src/modules/unet.py`
- inspect `src/modules/unet_blocks.py`
- confirm `up_blocks[2]` style-content feature shape is `[B, 64, 48, 48]`
- confirm baseline forward works with dummy tensors
- write recon notes into this plan or `CALLIRAG_CHANGELOG.md`

Exit criteria:

- no source edits except documentation
- injection point confirmed against current repo

### P1: Adapter Module

Add:

```text
src/modules/retrieval_adapter.py
tests/test_retrieval_adapter.py
```

Tests:

- output shape equals `h_q`
- initial output is exactly zero
- all-masked batch raises assertion/error
- partial mask does not NaN
- `return_gate=True` returns `alpha` and `pregate_norm`
- `alpha.grad` is nonzero after backward from nonzero downstream loss

Exit criteria:

- adapter smoke test passes on CPU
- no FontDiffuser source files modified yet

### P2: UNet Integration With Bit-Exact Invariance

Modify:

```text
src/modules/unet_blocks.py
src/modules/unet.py
```

Add:

```text
tests/test_adapter_integration.py
```

Tests:

- baseline UNet output without retrieval equals output with adapter attached and
  `alpha=0`
- compare both `noise_pred` and `offset_out_sum`
- adapter receives nonzero gradient path
- only adapter parameters are trainable after freeze helper is applied

Exit criteria:

- bit-exact check passes with `torch.equal`
- if `torch.equal` fails, report max abs diff and stop

### P3: Model-Level Retrieval Inputs

Modify:

```text
src/model.py
```

Add helper code to convert Plan A ref images into content-encoder features.

Tests:

- `retrieval_inputs=None` keeps original behavior
- `retrieval_inputs` with `alpha=0` remains bit-exact
- ref feature tensor shape is `[B, 5, 64, 48, 48]`
- `content_encoder` params have no gradients

Exit criteria:

- full `FontDiffuserModel.forward` supports optional retrieval inputs
- no changes to existing callers are required when retrieval is disabled

### P4: Plan A Ref-Pack Loader

Add:

```text
dataset/retrieval_ref_pack.py
scripts/inspect_retrieval_pack.py
```

Responsibilities:

- read Plan A outputs from `D:/00_project/callirag/retrieval_data_prepare`
- build 5-slot packs for every target case in `case_manifest.csv`
- load and normalize ref images the same way FontDiffuser loads style/content
  images
- enforce self-GT exclusion
- enforce at least one valid slot
- preserve `matched_comp`, `role`, `bank_id`, and `ref_char` in metadata

Exit criteria:

- script prints one valid ref pack per target
- no target character appears as its own ref
- all image paths exist or are explicitly reported missing

### P5: Tiny Overfit Sanity

Add:

```text
scripts/train_adapter_tiny_overfit.py
```

Scope:

- 1 or 2 cases
- frozen FontDiffuser
- adapter only
- diffusion loss only
- 50 to 300 steps
- fixed generation seed

Required logs:

```text
loss
alpha
pregate_norm
adapter grad norm
number of trainable params
ref ablation output diff
```

Exit criteria:

- loss can move downward on the tiny set
- `alpha` moves away from zero
- adapter gradients are nonzero
- backbone gradients are absent
- changing refs or mask changes output after training

If P5 fails, do not add mid-block injection. Diagnose adapter, data contract,
or loss flow first.

### P6: Manifest-Case Adapter Training

Add:

```text
scripts/train_adapter_manifest_cases.py
scripts/run_adapter_manifest_cases_inference.py
```

Training:

- all Plan A cases in `case_manifest.csv` unless `--targets` selects a subset
- frozen backbone
- adapter-only checkpoint
- no aux loss in V1
- default diagnostic architecture: direct skip residual injection at `up_blocks[2]`
- recommended starting config: `adapter_scale=10`, `offset_scale=0`, `direct_scale=1`

Inference groups:

```text
baseline_no_adapter
adapter_correct_refs
adapter_shuffled_refs
adapter_random_refs
adapter_all_anchor
adapter_coverage_only
adapter_alpha_zero
```

Exit criteria:

- outputs generated for every case/group
- all manifests saved
- adapter checkpoint saved separately from FontDiffuser weights

### P7: Ablation-First Evaluation

The first evaluation question is not "is it beautiful?" The first question is:

```text
Does the trained adapter use retrieval-specific information?
```

Minimum metrics:

- output pixel difference against `alpha_zero`
- OCR top-1 / confidence if available
- manual pairwise review on the manifest case set
- compare correct refs vs shuffled/random refs
- compare all-anchor vs coverage-only

Interpretation:

- correct refs > shuffled/random: retrieval signal is being used
- correct refs ~= shuffled/random but > alpha_zero: adapter learned generic bias
- all groups ~= alpha_zero: adapter is inactive
- all-anchor > coverage-only: anchor dominates, coverage weak
- coverage-only helps specific cases: coverage slots are meaningful

## 9. Mid-Block Injection Decision

Current C0 result:

```text
offset-path-only injection is too weak.
direct skip residual injection at up_blocks[2] can overfit and changes noise_pred.
```

Do not add mid-block injection yet. First train/evaluate the direct skip path on
the manifest cases.

Do not implement mid-block injection until P5 and P7 provide evidence.

Add mid-block only if:

- adapter is trainable
- high-resolution injection affects output
- correct refs do not sufficiently influence coarse structure

If needed, define a separate smaller adapter for mid hidden states. Do not reuse
the 48x48 offset-path adapter without a new shape/semantic review.

## 10. Files Expected To Change

New files:

```text
src/modules/retrieval_adapter.py
tests/test_retrieval_adapter.py
tests/test_adapter_integration.py
dataset/retrieval_ref_pack.py
scripts/inspect_retrieval_pack.py
scripts/train_adapter_tiny_overfit.py
scripts/train_adapter_manifest_cases.py
scripts/run_adapter_manifest_cases_inference.py
CALLIRAG_CHANGELOG.md
```

Modified files:

```text
src/modules/unet_blocks.py
src/modules/unet.py
src/model.py
```

Later, only after P5:

```text
src/dpm_solver/pipeline_dpm_solver.py
sample.py
train.py
```

## 11. Execution Discipline

- Do not run `pip install`.
- Do not modify Plan A files under `D:/00_project/callirag` unless explicitly
  requested.
- Do not change retrieval ranking during adapter development.
- Do not modify unrelated FontDiffuser modules.
- If a bug is discovered outside the planned files, record it first. Fix it
  only if it blocks the current phase.
- Prefer:

```text
D:/htt/miniconda3/envs/FontDiffuser/python.exe
```

over `conda run`, because the earlier run path swallowed stdout.

## 12. Current Next Step

Start with P1, not with training:

```text
1. Add RetrievalAdapter with corrected initialization.
2. Add CPU smoke tests for shape, mask, zero output, and gradient liveness.
3. Only then wire it into UNet.
```

The main engineering risk is not the injection location anymore. The main risk
is silently building a bit-exact but dead adapter. The corrected gate
initialization and gradient-liveness tests are mandatory.
