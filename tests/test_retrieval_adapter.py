import importlib.util
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_adapter_class():
    adapter_path = PROJECT_ROOT / "src" / "modules" / "retrieval_adapter.py"
    spec = importlib.util.spec_from_file_location("retrieval_adapter_module", adapter_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.RetrievalAdapter


def make_inputs(batch_size=2, n_slots=5, channels=64, height=8, width=8):
    h_q = torch.randn(batch_size, channels, height, width, requires_grad=True)
    refs = torch.randn(batch_size, n_slots, channels, height, width)
    slot_ids = torch.randint(0, 37, (batch_size, n_slots), dtype=torch.long)
    role_ids = torch.randint(0, 3, (batch_size, n_slots), dtype=torch.long)
    target_struct = torch.randint(0, 12, (batch_size,), dtype=torch.long)
    mask = torch.randint(0, 2, (batch_size, n_slots), dtype=torch.bool)
    for idx in range(batch_size):
        if not mask[idx].any():
            mask[idx, 0] = True
    return h_q, refs, slot_ids, role_ids, target_struct, mask


def main():
    torch.manual_seed(1234)
    RetrievalAdapter = load_adapter_class()
    adapter = RetrievalAdapter(
        feat_channels=64,
        ref_channels=64,
        n_slots=5,
        n_heads=4,
        ref_token_size=4,
    )

    h_q, refs, slot_ids, role_ids, target_struct, mask = make_inputs()
    delta_h, gate = adapter(
        h_q,
        refs,
        slot_ids,
        role_ids,
        target_struct,
        mask,
        return_gate=True,
    )
    assert delta_h.shape == h_q.shape
    assert torch.equal(delta_h, torch.zeros_like(delta_h))
    assert gate["alpha"].ndim == 0
    assert gate["pregate_norm"].ndim == 0
    assert torch.isfinite(gate["pregate_norm"])

    all_masked = torch.zeros_like(mask)
    try:
        adapter(h_q, refs, slot_ids, role_ids, target_struct, all_masked)
        raise AssertionError("all-masked input should fail")
    except ValueError:
        pass

    loss = adapter(
        h_q,
        refs,
        slot_ids,
        role_ids,
        target_struct,
        mask,
    ).sum()
    loss.backward()
    alpha_grad = adapter.alpha.grad
    assert alpha_grad is not None
    assert torch.isfinite(alpha_grad)
    assert alpha_grad.abs().item() > 0.0

    print("test_retrieval_adapter: OK")
    print(f"alpha_grad: {alpha_grad.item():.8f}")
    print(f"pregate_norm: {gate['pregate_norm'].item():.8f}")


if __name__ == "__main__":
    main()
