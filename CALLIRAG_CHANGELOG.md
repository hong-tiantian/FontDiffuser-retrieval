# CALLI-RAG Changelog

## 2026-05-15 - P1/P2/P3 Retrieval Adapter Bootstrap

- `src/modules/retrieval_adapter.py` - new file - P1 - adds `RetrievalAdapter` with 5-slot masked attention, metadata embeddings, scalar `alpha` gate initialized to zero, nonzero `out_proj` initialization to avoid the double-zero gradient trap, plus attach/freeze helpers. Verification: `tests/test_retrieval_adapter.py`.
- `tests/test_retrieval_adapter.py` - new file - P1 - standalone CPU smoke test for shape, zero initial output, all-masked failure, `return_gate`, and nonzero `alpha` gradient.
- `src/modules/unet_blocks.py` - approx L1/L520/L545/L552 - P2 - adds optional `retrieval_adapter` attachment point and applies `delta_h` to `style_content_feat` before the `StyleRSIUpBlock2D` offset loop. Verification: integration test to be run after `tests/test_adapter_integration.py` is added.
- `src/modules/unet.py` - approx L205/L285 - P2 - adds optional `retrieval_inputs` argument and routes it only into `up_blocks[2]` when a retrieval adapter is attached. Verification: integration test to be run after `tests/test_adapter_integration.py` is added.
- `src/model.py` - approx L25/L30/L55 - P3 - adds optional model-level `retrieval_inputs`, converts `[B,5,3,H,W]` ref images into frozen `content_encoder` residual features, and passes `[B,5,64,48,48]` features to UNet. Verification: integration test to be run after `tests/test_adapter_integration.py` is added.
- `tests/test_adapter_integration.py` - new file - P2 - standalone CPU integration test for UNet bit-exact invariance, adapter gradient liveness, and adapter-only trainable parameter counting.
- `dataset/retrieval_ref_pack.py` - new file - P4 - reads Plan A `case_manifest.csv` and `sim_layer.json`, builds 5-slot retrieval packs, checks target leakage, resolves `/d/...` paths on Windows, and can load normalized ref image tensors.
- `scripts/inspect_retrieval_pack.py` - new file - P4 - command-line inspection utility for Plan A retrieval packs, including slot ids, role ids, masks, missing image paths, and leakage checks.
- `scripts/inspect_retrieval_pack.py` - approx loader defaults - P4 hygiene - keeps inspection default aligned with every case in `case_manifest.csv`; `--targets` can be used for manual subset checks.
- `scripts/train_adapter_tiny_overfit.py` - new file - P5 - trains only the retrieval adapter on a tiny CSV manifest using frozen FontDiffuser checkpoints, logs loss/alpha/gradient norm, and reports a shuffled-ref ablation difference.
- `examples/tiny_overfit_manifest.example.csv` - new file - P5 - documents the minimal CSV schema required by the tiny overfit script.
