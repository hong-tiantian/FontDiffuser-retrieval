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


def load_model(args, device):
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
    attach_retrieval_adapter(model.unet, up_block_index=2)
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
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-checkpoint", action="store_true")
    cli_args = parser.parse_args()

    random.seed(cli_args.seed)
    torch.manual_seed(cli_args.seed)
    device = torch.device(cli_args.device if torch.cuda.is_available() else "cpu")
    cli_args.output_dir.mkdir(parents=True, exist_ok=True)

    args = build_args(cli_args)
    model = load_model(args, device=device)
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
    if trainable_params != adapter_params:
        raise AssertionError(
            f"freeze check failed: trainable={trainable_params}, adapter={adapter_params}"
        )
    print(f"device: {device}")
    print(f"num_samples: {len(rows)}")
    print(f"trainable_params: {trainable_params}")

    last_loss = None
    for step in range(1, cli_args.steps + 1):
        target_images = batch["target_images"]
        noise = torch.randn_like(target_images)
        timesteps = torch.randint(
            0,
            noise_scheduler.num_train_timesteps,
            (target_images.shape[0],),
            device=target_images.device,
        ).long()
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
        grad_norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
        optimizer.step()
        last_loss = loss.item()

        if step == 1 or step % cli_args.log_every == 0 or step == cli_args.steps:
            print(
                f"step={step} loss={loss.item():.6f} diff={diff_loss.item():.6f} "
                f"offset={float(offset_out_sum.detach().cpu()):.6f} "
                f"alpha={adapter.alpha.item():.8f} grad_norm={float(grad_norm):.6f}"
            )

    model.eval()
    with torch.no_grad():
        target_images = batch["target_images"]
        noise = torch.randn_like(target_images)
        timesteps = torch.randint(
            0,
            noise_scheduler.num_train_timesteps,
            (target_images.shape[0],),
            device=target_images.device,
        ).long()
        noisy_target_images = noise_scheduler.add_noise(target_images, noise, timesteps)
        out_correct = model(
            x_t=noisy_target_images,
            timesteps=timesteps,
            style_images=batch["style_images"],
            content_images=batch["content_images"],
            content_encoder_downsample_size=args.content_encoder_downsample_size,
            retrieval_inputs=batch["retrieval_inputs"],
        )[0]
        shuffled_inputs = clone_retrieval_inputs(batch["retrieval_inputs"])
        shuffled_inputs["ref_images"] = torch.roll(shuffled_inputs["ref_images"], shifts=1, dims=1)
        out_shuffled = model(
            x_t=noisy_target_images,
            timesteps=timesteps,
            style_images=batch["style_images"],
            content_images=batch["content_images"],
            content_encoder_downsample_size=args.content_encoder_downsample_size,
            retrieval_inputs=shuffled_inputs,
        )[0]
        ref_ablation_diff = (out_correct - out_shuffled).abs().mean().item()

    print(f"final_loss: {last_loss:.6f}")
    print(f"final_alpha: {adapter.alpha.item():.8f}")
    print(f"ref_ablation_mean_abs_diff: {ref_ablation_diff:.8f}")

    metrics = {
        "steps": cli_args.steps,
        "final_loss": last_loss,
        "final_alpha": adapter.alpha.item(),
        "ref_ablation_mean_abs_diff": ref_ablation_diff,
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
