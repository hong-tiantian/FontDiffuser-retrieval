import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.fontdiffuser import get_parser
from dataset.retrieval_ref_pack import (
    DEFAULT_PLAN_A_ROOT,
    load_retrieval_packs,
    pack_to_model_inputs,
    parse_path_maps,
)
from src import (
    FontDiffuserModel,
    build_content_encoder,
    build_ddpm_scheduler,
    build_style_encoder,
    build_unet,
)
from src.modules.retrieval_adapter import (
    attach_retrieval_adapter,
    freeze_backbone_train_adapter,
)


def build_args(cli_args):
    parser = get_parser()
    args = parser.parse_args([])
    args.style_image_size = (args.style_image_size, args.style_image_size)
    args.content_image_size = (args.content_image_size, args.content_image_size)
    args.phase_1_ckpt_dir = cli_args.ckpt_dir
    return args


def make_transform(size):
    return transforms.Compose(
        [
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )


def load_image(path, transform):
    image = Image.open(path).convert("RGB")
    return transform(image)


def load_tiny_manifest(path):
    path = Path(path)
    rows = []
    if path.suffix.lower() == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    return rows


def load_model(args, device, adapter_scale, offset_scale, direct_scale):
    unet = build_unet(args=args)
    style_encoder = build_style_encoder(args=args)
    content_encoder = build_content_encoder(args=args)

    if args.phase_1_ckpt_dir is None:
        raise ValueError("--ckpt-dir is required.")
    ckpt_dir = Path(args.phase_1_ckpt_dir)
    unet.load_state_dict(torch.load(ckpt_dir / "unet.pth", map_location="cpu"))
    style_encoder.load_state_dict(torch.load(ckpt_dir / "style_encoder.pth", map_location="cpu"))
    content_encoder.load_state_dict(torch.load(ckpt_dir / "content_encoder.pth", map_location="cpu"))

    model = FontDiffuserModel(
        unet=unet,
        style_encoder=style_encoder,
        content_encoder=content_encoder,
    )
    attach_retrieval_adapter(
        model.unet,
        up_block_index=2,
        residual_scale=adapter_scale,
        offset_scale=offset_scale,
        direct_scale=direct_scale,
    )
    freeze_backbone_train_adapter(model, up_block_index=2)
    model.to(device)
    return model


def make_batch(rows, packs, path_maps, image_transform, device):
    content_images = []
    style_images = []
    target_images = []
    retrieval_items = []

    for row in rows:
        target_char = row["target_char"]
        if target_char not in packs:
            raise KeyError(f"target_char {target_char} not found in retrieval packs.")

        content_images.append(load_image(row["content_image_path"], image_transform))
        style_images.append(load_image(row["style_image_path"], image_transform))
        target_images.append(load_image(row["target_image_path"], image_transform))
        retrieval_items.append(pack_to_model_inputs(packs[target_char], path_maps=path_maps))

    retrieval_inputs = {
        "ref_images": torch.stack([item["ref_images"] for item in retrieval_items], dim=0),
        "slot_ids": torch.stack([item["slot_ids"] for item in retrieval_items], dim=0),
        "role_ids": torch.stack([item["role_ids"] for item in retrieval_items], dim=0),
        "target_struct": torch.stack([item["target_struct"] for item in retrieval_items], dim=0),
        "mask": torch.stack([item["mask"] for item in retrieval_items], dim=0),
    }

    batch = {
        "content_images": torch.stack(content_images, dim=0).to(device),
        "style_images": torch.stack(style_images, dim=0).to(device),
        "target_images": torch.stack(target_images, dim=0).to(device),
        "retrieval_inputs": retrieval_inputs,
    }
    for key, value in retrieval_inputs.items():
        retrieval_inputs[key] = value.to(device)
    return batch


def clone_retrieval_inputs(retrieval_inputs):
    return {key: value.clone() for key, value in retrieval_inputs.items()}


def tensor_mean_abs_diff(a, b):
    return (a - b).abs().mean().item()


def model_noise_pred(model, batch, noisy_target_images, timesteps, args, retrieval_inputs):
    return model(
        x_t=noisy_target_images,
        timesteps=timesteps,
        style_images=batch["style_images"],
        content_images=batch["content_images"],
        content_encoder_downsample_size=args.content_encoder_downsample_size,
        retrieval_inputs=retrieval_inputs,
    )[0]


def sample_noise_and_timesteps(target_images, noise_scheduler, fixed_noise, fixed_timesteps):
    if fixed_noise is not None and fixed_timesteps is not None:
        return fixed_noise, fixed_timesteps
    noise = torch.randn_like(target_images)
    timesteps = torch.randint(
        0,
        noise_scheduler.config.num_train_timesteps,
        (target_images.shape[0],),
        device=target_images.device,
    ).long()
    return noise, timesteps


def main():
    parser = argparse.ArgumentParser(description="Tiny overfit sanity check for retrieval adapter.")
    parser.add_argument("--ckpt-dir", type=str, required=True)
    parser.add_argument("--tiny-manifest", type=str, required=True)
    parser.add_argument("--plan-a-root", type=Path, default=DEFAULT_PLAN_A_ROOT)
    parser.add_argument("--path-map", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/adapter_tiny_overfit"))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--adapter-scale",
        type=float,
        default=1.0,
        help="Temporary residual multiplier for C0 diagnostics. Default preserves normal adapter behavior.",
    )
    parser.add_argument(
        "--offset-scale",
        type=float,
        default=1.0,
        help="Scale for the original offset-path retrieval injection.",
    )
    parser.add_argument(
        "--direct-scale",
        type=float,
        default=0.0,
        help="Scale for direct retrieval residual injection into StyleRSI skip features.",
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument(
        "--resample-noise",
        action="store_true",
        help="Resample diffusion noise/timestep every step. Default keeps them fixed for true tiny-overfit diagnostics.",
    )
    cli_args = parser.parse_args()

    random.seed(cli_args.seed)
    torch.manual_seed(cli_args.seed)
    device = torch.device(cli_args.device if torch.cuda.is_available() else "cpu")
    cli_args.output_dir.mkdir(parents=True, exist_ok=True)

    args = build_args(cli_args)
    model = load_model(
        args,
        device=device,
        adapter_scale=cli_args.adapter_scale,
        offset_scale=cli_args.offset_scale,
        direct_scale=cli_args.direct_scale,
    )
    model.train()
    noise_scheduler = build_ddpm_scheduler(args)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=cli_args.lr,
        weight_decay=0.0,
    )

    rows = load_tiny_manifest(cli_args.tiny_manifest)
    if not rows:
        raise ValueError("tiny manifest is empty.")
    path_maps = parse_path_maps(cli_args.path_map)
    packs = load_retrieval_packs(cli_args.plan_a_root)
    image_transform = make_transform(args.resolution)
    batch = make_batch(rows, packs, path_maps, image_transform, device)

    adapter = model.unet.up_blocks[2].retrieval_adapter
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    adapter_params = sum(param.numel() for param in adapter.parameters())
    direct_params = sum(
        param.numel()
        for param in model.unet.up_blocks[2].retrieval_res_projs.parameters()
        if param.requires_grad
    )
    expected_trainable_params = adapter_params + direct_params
    if trainable_params != expected_trainable_params:
        raise AssertionError(
            "freeze check failed: "
            f"trainable={trainable_params}, expected={expected_trainable_params}"
        )
    print(f"device: {device}")
    print(f"num_samples: {len(rows)}")
    print(f"trainable_params: {trainable_params}")
    print(f"resample_noise: {cli_args.resample_noise}")
    print(f"adapter_scale: {cli_args.adapter_scale}")
    print(f"offset_scale: {cli_args.offset_scale}")
    print(f"direct_scale: {cli_args.direct_scale}")

    fixed_noise = None
    fixed_timesteps = None
    if not cli_args.resample_noise:
        fixed_noise = torch.randn_like(batch["target_images"])
        fixed_timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (batch["target_images"].shape[0],),
            device=batch["target_images"].device,
        ).long()

    last_loss = None
    for step in range(1, cli_args.steps + 1):
        target_images = batch["target_images"]
        noise, timesteps = sample_noise_and_timesteps(
            target_images,
            noise_scheduler,
            fixed_noise=fixed_noise,
            fixed_timesteps=fixed_timesteps,
        )
        noisy_target_images = noise_scheduler.add_noise(target_images, noise, timesteps)

        noise_pred, offset_out_sum = model(
            x_t=noisy_target_images,
            timesteps=timesteps,
            style_images=batch["style_images"],
            content_images=batch["content_images"],
            content_encoder_downsample_size=args.content_encoder_downsample_size,
            retrieval_inputs=batch["retrieval_inputs"],
        )
        diff_loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        loss = diff_loss + 0.5 * (offset_out_sum / 2)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        alpha_grad = None if adapter.alpha.grad is None else adapter.alpha.grad.detach().item()
        grad_norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
        optimizer.step()
        last_loss = loss.item()

        if step == 1 or step % cli_args.log_every == 0 or step == cli_args.steps:
            stats = getattr(adapter, "last_stats", {})
            pregate_norm = float(stats.get("pregate_norm", torch.tensor(0.0)).detach().cpu())
            delta_abs_mean = float(stats.get("delta_abs_mean", torch.tensor(0.0)).detach().cpu())
            print(
                f"step={step} loss={loss.item():.6f} diff={diff_loss.item():.6f} "
                f"offset={float(offset_out_sum.detach().cpu()):.6f} "
                f"alpha={adapter.alpha.item():.8f} "
                f"alpha_grad={alpha_grad:.6e} grad_norm={float(grad_norm):.6e} "
                f"pregate_norm={pregate_norm:.6e} delta_abs_mean={delta_abs_mean:.6e}"
            )

    model.eval()
    with torch.no_grad():
        target_images = batch["target_images"]
        noise, timesteps = sample_noise_and_timesteps(
            target_images,
            noise_scheduler,
            fixed_noise=fixed_noise,
            fixed_timesteps=fixed_timesteps,
        )
        noisy_target_images = noise_scheduler.add_noise(target_images, noise, timesteps)
        out_correct = model_noise_pred(
            model, batch, noisy_target_images, timesteps, args, batch["retrieval_inputs"]
        )

        old_alpha = adapter.alpha.detach().clone()
        adapter.alpha.zero_()
        out_alpha_zero = model_noise_pred(
            model, batch, noisy_target_images, timesteps, args, batch["retrieval_inputs"]
        )
        adapter.alpha.copy_(old_alpha)

        shuffled_inputs = clone_retrieval_inputs(batch["retrieval_inputs"])
        shuffled_inputs["ref_images"] = torch.roll(shuffled_inputs["ref_images"], shifts=1, dims=1)
        out_shuffled = model_noise_pred(
            model, batch, noisy_target_images, timesteps, args, shuffled_inputs
        )

        zero_ref_inputs = clone_retrieval_inputs(batch["retrieval_inputs"])
        zero_ref_inputs["ref_images"] = torch.zeros_like(zero_ref_inputs["ref_images"])
        out_zero_refs = model_noise_pred(
            model, batch, noisy_target_images, timesteps, args, zero_ref_inputs
        )

        random_ref_inputs = clone_retrieval_inputs(batch["retrieval_inputs"])
        random_ref_inputs["ref_images"] = torch.randn_like(random_ref_inputs["ref_images"])
        out_random_refs = model_noise_pred(
            model, batch, noisy_target_images, timesteps, args, random_ref_inputs
        )

        ref_ablation_diff = tensor_mean_abs_diff(out_correct, out_shuffled)
        alpha_zero_diff = tensor_mean_abs_diff(out_correct, out_alpha_zero)
        zero_refs_diff = tensor_mean_abs_diff(out_correct, out_zero_refs)
        random_refs_diff = tensor_mean_abs_diff(out_correct, out_random_refs)
        final_stats = getattr(adapter, "last_stats", {})
        final_pregate_norm = float(final_stats.get("pregate_norm", torch.tensor(0.0)).detach().cpu())
        final_delta_abs_mean = float(final_stats.get("delta_abs_mean", torch.tensor(0.0)).detach().cpu())

    print(f"final_loss: {last_loss:.6f}")
    print(f"final_alpha: {adapter.alpha.item():.8f}")
    print(f"final_pregate_norm: {final_pregate_norm:.8e}")
    print(f"final_delta_abs_mean: {final_delta_abs_mean:.8e}")
    print(f"shuffle_refs_mean_abs_diff: {ref_ablation_diff:.8f}")
    print(f"alpha_zero_mean_abs_diff: {alpha_zero_diff:.8f}")
    print(f"zero_refs_mean_abs_diff: {zero_refs_diff:.8f}")
    print(f"random_refs_mean_abs_diff: {random_refs_diff:.8f}")

    metrics = {
        "steps": cli_args.steps,
        "resample_noise": cli_args.resample_noise,
        "adapter_scale": cli_args.adapter_scale,
        "offset_scale": cli_args.offset_scale,
        "direct_scale": cli_args.direct_scale,
        "final_loss": last_loss,
        "final_alpha": adapter.alpha.item(),
        "final_pregate_norm": final_pregate_norm,
        "final_delta_abs_mean": final_delta_abs_mean,
        "shuffle_refs_mean_abs_diff": ref_ablation_diff,
        "alpha_zero_mean_abs_diff": alpha_zero_diff,
        "zero_refs_mean_abs_diff": zero_refs_diff,
        "random_refs_mean_abs_diff": random_refs_diff,
        "trainable_params": trainable_params,
        "adapter_params": adapter_params,
        "num_samples": len(rows),
    }
    (cli_args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if cli_args.save_checkpoint:
        torch.save(adapter.state_dict(), cli_args.output_dir / "retrieval_adapter.pth")


if __name__ == "__main__":
    main()
