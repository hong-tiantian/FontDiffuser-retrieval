import importlib
import importlib.util
import random
import sys
import types
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def make_default_args():
    from configs.fontdiffuser import get_parser

    args = get_parser().parse_args([])
    args.style_image_size = (args.style_image_size, args.style_image_size)
    args.content_image_size = (args.content_image_size, args.content_image_size)
    return args


def load_build_unet_without_src_init():
    src_pkg = types.ModuleType("src")
    src_pkg.__path__ = [str(PROJECT_ROOT / "src")]
    sys.modules["src"] = src_pkg

    modules_pkg = types.ModuleType("src.modules")
    modules_pkg.__path__ = [str(PROJECT_ROOT / "src" / "modules")]
    sys.modules["src.modules"] = modules_pkg

    unet_mod = importlib.import_module("src.modules.unet")
    content_mod = importlib.import_module("src.modules.content_encoder")
    style_mod = importlib.import_module("src.modules.style_encoder")
    scr_mod = importlib.import_module("src.modules.scr")

    src_pkg.UNet = unet_mod.UNet
    src_pkg.ContentEncoder = content_mod.ContentEncoder
    src_pkg.StyleEncoder = style_mod.StyleEncoder
    src_pkg.SCR = scr_mod.SCR

    build_path = PROJECT_ROOT / "src" / "build.py"
    spec = importlib.util.spec_from_file_location("fontdiffuser_build", build_path)
    build_mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(build_mod)
    return build_mod.build_unet


def load_retrieval_adapter_class():
    adapter_path = PROJECT_ROOT / "src" / "modules" / "retrieval_adapter.py"
    spec = importlib.util.spec_from_file_location("retrieval_adapter_module", adapter_path)
    adapter_mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(adapter_mod)
    return adapter_mod.RetrievalAdapter


def make_dummy_encoder_hidden_states(batch_size: int, device: torch.device):
    style_img_feature = torch.randn(batch_size, 1024, 3, 3, device=device)
    style_hidden_states = style_img_feature.permute(0, 2, 3, 1).reshape(batch_size, 9, 1024)

    content_residual_features = [
        torch.randn(batch_size, 3, 96, 96, device=device),
        torch.randn(batch_size, 64, 48, 48, device=device),
        torch.randn(batch_size, 128, 24, 24, device=device),
        torch.randn(batch_size, 256, 12, 12, device=device),
    ]
    content_feature = torch.randn(batch_size, 512, 6, 6, device=device)
    content_residual_features.append(content_feature)

    style_content_res_features = [
        torch.randn(batch_size, 3, 96, 96, device=device),
        torch.randn(batch_size, 64, 48, 48, device=device),
        torch.randn(batch_size, 128, 24, 24, device=device),
        torch.randn(batch_size, 256, 12, 12, device=device),
    ]
    style_content_feature = torch.randn(batch_size, 512, 6, 6, device=device)
    style_content_res_features.append(style_content_feature)

    return [
        style_img_feature,
        content_residual_features,
        style_hidden_states,
        style_content_res_features,
    ]


def make_retrieval_inputs(batch_size: int, n_slots: int, device: torch.device):
    refs = torch.randn(batch_size, n_slots, 64, 48, 48, device=device)
    slot_ids = torch.randint(0, 37, (batch_size, n_slots), dtype=torch.long, device=device)
    role_ids = torch.randint(0, 3, (batch_size, n_slots), dtype=torch.long, device=device)
    target_struct = torch.randint(0, 12, (batch_size,), dtype=torch.long, device=device)
    mask = torch.randint(0, 2, (batch_size, n_slots), dtype=torch.bool, device=device)
    for idx in range(batch_size):
        if not mask[idx].any():
            mask[idx, 0] = True
    return {
        "refs": refs,
        "slot_ids": slot_ids,
        "role_ids": role_ids,
        "target_struct": target_struct,
        "mask": mask,
    }


def main():
    torch.manual_seed(1234)
    random.seed(1234)
    device = torch.device("cpu")
    batch_size = 1
    n_slots = 5

    args = make_default_args()
    build_unet = load_build_unet_without_src_init()
    unet = build_unet(args)
    unet.to(device)
    unet.eval()

    RetrievalAdapter = load_retrieval_adapter_class()
    adapter = RetrievalAdapter(
        feat_channels=64,
        ref_channels=64,
        n_slots=n_slots,
        ref_token_size=4,
    )
    adapter.to(device)
    adapter.eval()

    sample = torch.randn(batch_size, 3, 96, 96, device=device)
    timesteps = torch.randint(0, 1000, (batch_size,), dtype=torch.long, device=device)
    encoder_hidden_states = make_dummy_encoder_hidden_states(batch_size, device)
    retrieval_inputs = make_retrieval_inputs(batch_size, n_slots, device)

    with torch.no_grad():
        out_baseline = unet(
            sample=sample,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            content_encoder_downsample_size=args.content_encoder_downsample_size,
        )
        unet.up_blocks[2].retrieval_adapter = adapter
        out_with_adapter = unet(
            sample=sample,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            content_encoder_downsample_size=args.content_encoder_downsample_size,
            retrieval_inputs=retrieval_inputs,
        )

    noise_equal = torch.equal(out_baseline[0], out_with_adapter[0])
    offset_equal = torch.equal(out_baseline[1], out_with_adapter[1])
    if not noise_equal or not offset_equal:
        print(f"noise_equal: {noise_equal}")
        print(f"offset_equal: {offset_equal}")
        print(f"max_noise_diff: {(out_baseline[0] - out_with_adapter[0]).abs().max().item()}")
        print(f"max_offset_diff: {(out_baseline[1] - out_with_adapter[1]).abs().max().item()}")
        raise AssertionError("bit-exact invariance failed")

    unet.train()
    unet.zero_grad(set_to_none=True)
    out_train = unet(
        sample=sample,
        timestep=timesteps,
        encoder_hidden_states=encoder_hidden_states,
        content_encoder_downsample_size=args.content_encoder_downsample_size,
        retrieval_inputs=retrieval_inputs,
    )
    loss = out_train[0].sum() + out_train[1]
    loss.backward()
    alpha_grad = unet.up_blocks[2].retrieval_adapter.alpha.grad
    assert alpha_grad is not None
    assert alpha_grad.abs().item() > 0.0

    for param in unet.parameters():
        param.requires_grad_(False)
    for param in unet.up_blocks[2].retrieval_adapter.parameters():
        param.requires_grad_(True)
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    adapter_params = sum(p.numel() for p in unet.up_blocks[2].retrieval_adapter.parameters())
    assert trainable_params == adapter_params

    print("test_adapter_integration: OK")
    print(f"alpha_grad: {alpha_grad.item():.8f}")
    print(f"trainable_params: {trainable_params}")


if __name__ == "__main__":
    main()
