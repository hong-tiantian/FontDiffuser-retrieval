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
    unet_state = torch.load(ckpt_dir / "unet.pth", map_location="cpu")
    missing, unexpected = unet.load_state_dict(unet_state, strict=False)
    if missing:
        print(
            f"[load] unet.pth: {len(missing)} missing key(s) use random init "
            f"(e.g. retrieval_res_projs for direct-scale). First: {missing[0]}"
        )
    if unexpected:
        print(f"[load] unet.pth: {len(unexpected)} unexpected key(s) ignored.")
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


def make_retrieval_inputs_for_mode(retrieval_inputs, mode):
    inputs = clone_retrieval_inputs(retrieval_inputs)
    if mode == "correct":
        return inputs
    if mode == "shuffled":
        batch_size = inputs["ref_images"].shape[0]
        if batch_size > 1:
            for key in ("ref_images", "slot_ids", "role_ids", "mask"):
                inputs[key] = torch.roll(inputs[key], shifts=1, dims=0)
        else:
            inputs["ref_images"] = torch.roll(inputs["ref_images"], shifts=1, dims=1)
            inputs["slot_ids"] = torch.roll(inputs["slot_ids"], shifts=1, dims=1)
            inputs["role_ids"] = torch.roll(inputs["role_ids"], shifts=1, dims=1)
            inputs["mask"] = torch.roll(inputs["mask"], shifts=1, dims=1)
        return inputs
    if mode == "zero":
        inputs["ref_images"] = torch.zeros_like(inputs["ref_images"])
        return inputs
    if mode == "random":
        inputs["ref_images"] = torch.randn_like(inputs["ref_images"])
        return inputs
    raise ValueError(f"unknown retrieval mode: {mode}")


def sample_retrieval_mode(args):
    weights = [
        ("correct", args.p_correct),
        ("shuffled", args.p_shuffled),
        ("zero", args.p_zero),
        ("random", args.p_random),
    ]
    if any(weight < 0 for _, weight in weights):
        raise ValueError("retrieval mode probabilities must be non-negative.")
    total = sum(weight for _, weight in weights)
    if total <= 0:
        raise ValueError("retrieval mode probabilities must sum to a positive value.")
    threshold = random.random() * total
    cumulative = 0.0
    for mode, weight in weights:
        if weight == 0:
            continue
        cumulative += weight
        if threshold < cumulative:
            return mode
    return weights[-1][0]


def trainable_retrieval_parameters(model):
    return [param for param in model.parameters() if param.requires_grad]


def retrieval_state_dict(model):
    block = model.unet.up_blocks[2]
    return {
        "retrieval_adapter": block.retrieval_adapter.state_dict(),
        "retrieval_res_projs": block.retrieval_res_projs.state_dict(),
        "retrieval_offset_scale": block.retrieval_offset_scale,
        "retrieval_direct_scale": block.retrieval_direct_scale,
    }


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


def model_noise_pred_alpha_zero(model, batch, noisy_target_images, timesteps, args, retrieval_inputs):
    adapter = model.unet.up_blocks[2].retrieval_adapter
    old_alpha = adapter.alpha.detach().clone()
    with torch.no_grad():
        try:
            adapter.alpha.zero_()
            noise_pred = model_noise_pred(
                model,
                batch,
                noisy_target_images,
                timesteps,
                args,
                retrieval_inputs,
            ).detach()
        finally:
            adapter.alpha.copy_(old_alpha)
    return noise_pred


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
    parser.add_argument("--tiny-manifest", "--manifest", dest="tiny_manifest", type=str, required=True)
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
    parser.add_argument(
        "--retrieval-mode-augmentation",
        action="store_true",
        help="Sample correct/shuffled/zero/random retrieval refs during training.",
    )
    parser.add_argument("--p-correct", type=float, default=0.70)
    parser.add_argument("--p-shuffled", type=float, default=0.10)
    parser.add_argument("--p-zero", type=float, default=0.10)
    parser.add_argument("--p-random", type=float, default=0.10)
    parser.add_argument(
        "--lambda-delta",
        type=float,
        default=0.0,
        help="L1 penalty on retrieval delta_h mean absolute value.",
    )
    parser.add_argument(
        "--lambda-alpha",
        type=float,
        default=0.0,
        help="L1 penalty on adapter alpha.",
    )
    parser.add_argument(
        "--wrong-ref-target",
        choices=("diffusion", "alpha_zero"),
        default="diffusion",
        help=(
            "Training target for shuffled/zero/random refs. "
            "'diffusion' preserves the old behavior; 'alpha_zero' pushes wrong refs "
            "back to the frozen baseline prediction."
        ),
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
    trainable_parameters = trainable_retrieval_parameters(model)
    optimizer = torch.optim.AdamW(
        trainable_parameters,
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
    print(f"retrieval_mode_augmentation: {cli_args.retrieval_mode_augmentation}")
    print(
        "mode_probabilities: "
        f"correct={cli_args.p_correct} shuffled={cli_args.p_shuffled} "
        f"zero={cli_args.p_zero} random={cli_args.p_random}"
    )
    print(f"lambda_delta: {cli_args.lambda_delta}")
    print(f"lambda_alpha: {cli_args.lambda_alpha}")
    print(f"wrong_ref_target: {cli_args.wrong_ref_target}")

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
    mode_totals = {
        mode: {
            "count": 0,
            "loss": 0.0,
            "objective": 0.0,
            "diff": 0.0,
            "baseline": 0.0,
            "delta": 0.0,
            "alpha": 0.0,
        }
        for mode in ("correct", "shuffled", "zero", "random")
    }
    for step in range(1, cli_args.steps + 1):
        target_images = batch["target_images"]
        noise, timesteps = sample_noise_and_timesteps(
            target_images,
            noise_scheduler,
            fixed_noise=fixed_noise,
            fixed_timesteps=fixed_timesteps,
        )
        noisy_target_images = noise_scheduler.add_noise(target_images, noise, timesteps)
        retrieval_mode = (
            sample_retrieval_mode(cli_args)
            if cli_args.retrieval_mode_augmentation
            else "correct"
        )
        step_retrieval_inputs = make_retrieval_inputs_for_mode(
            batch["retrieval_inputs"],
            retrieval_mode,
        )

        noise_pred, offset_out_sum = model(
            x_t=noisy_target_images,
            timesteps=timesteps,
            style_images=batch["style_images"],
            content_images=batch["content_images"],
            content_encoder_downsample_size=args.content_encoder_downsample_size,
            retrieval_inputs=step_retrieval_inputs,
        )
        diff_loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        forward_regularization = getattr(adapter, "last_regularization", {})
        delta_reg = forward_regularization.get(
            "delta_abs_mean",
            torch.zeros((), device=target_images.device, dtype=diff_loss.dtype),
        )
        forward_stats = {
            key: value.detach().clone()
            for key, value in getattr(adapter, "last_stats", {}).items()
            if torch.is_tensor(value)
        }
        baseline_loss = torch.zeros((), device=target_images.device, dtype=diff_loss.dtype)
        if retrieval_mode != "correct" and cli_args.wrong_ref_target == "alpha_zero":
            baseline_pred = model_noise_pred_alpha_zero(
                model,
                batch,
                noisy_target_images,
                timesteps,
                args,
                step_retrieval_inputs,
            )
            objective_loss = F.mse_loss(
                noise_pred.float(),
                baseline_pred.float(),
                reduction="mean",
            )
            baseline_loss = objective_loss
        else:
            objective_loss = diff_loss
        alpha_reg = adapter.alpha.abs()
        loss = (
            objective_loss
            + 0.5 * (offset_out_sum / 2)
            + cli_args.lambda_delta * delta_reg
            + cli_args.lambda_alpha * alpha_reg
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        alpha_grad = None if adapter.alpha.grad is None else adapter.alpha.grad.detach().item()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
        optimizer.step()
        last_loss = loss.item()
        mode_totals[retrieval_mode]["count"] += 1
        mode_totals[retrieval_mode]["loss"] += loss.item()
        mode_totals[retrieval_mode]["objective"] += objective_loss.item()
        mode_totals[retrieval_mode]["diff"] += diff_loss.item()
        mode_totals[retrieval_mode]["baseline"] += baseline_loss.item()
        mode_totals[retrieval_mode]["delta"] += float(delta_reg.detach().cpu())
        mode_totals[retrieval_mode]["alpha"] += float(alpha_reg.detach().cpu())

        if step == 1 or step % cli_args.log_every == 0 or step == cli_args.steps:
            pregate_norm = float(forward_stats.get("pregate_norm", torch.tensor(0.0)).detach().cpu())
            delta_abs_mean = float(forward_stats.get("delta_abs_mean", torch.tensor(0.0)).detach().cpu())
            print(
                f"step={step} mode={retrieval_mode} loss={loss.item():.6f} "
                f"objective={objective_loss.item():.6f} diff={diff_loss.item():.6f} "
                f"baseline={baseline_loss.item():.6f} "
                f"offset={float(offset_out_sum.detach().cpu()):.6f} "
                f"delta_reg={float(delta_reg.detach().cpu()):.6e} "
                f"alpha_reg={float(alpha_reg.detach().cpu()):.6e} "
                f"alpha={adapter.alpha.item():.8f} "
                f"alpha_grad={alpha_grad:.6e} grad_norm={float(grad_norm):.6e} "
                f"pregate_norm={pregate_norm:.6e} delta_abs_mean={delta_abs_mean:.6e}"
            )

    mode_summary = {}
    for mode, totals in mode_totals.items():
        count = totals["count"]
        if count == 0:
            mode_summary[mode] = {
                "count": 0,
                "mean_loss": None,
                "mean_objective": None,
                "mean_diff": None,
                "mean_baseline": None,
                "mean_delta_reg": None,
                "mean_alpha_reg": None,
            }
            continue
        mode_summary[mode] = {
            "count": count,
            "mean_loss": totals["loss"] / count,
            "mean_objective": totals["objective"] / count,
            "mean_diff": totals["diff"] / count,
            "mean_baseline": totals["baseline"] / count,
            "mean_delta_reg": totals["delta"] / count,
            "mean_alpha_reg": totals["alpha"] / count,
        }

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

        shuffled_inputs = make_retrieval_inputs_for_mode(batch["retrieval_inputs"], "shuffled")
        out_shuffled = model_noise_pred(
            model, batch, noisy_target_images, timesteps, args, shuffled_inputs
        )

        zero_ref_inputs = make_retrieval_inputs_for_mode(batch["retrieval_inputs"], "zero")
        out_zero_refs = model_noise_pred(
            model, batch, noisy_target_images, timesteps, args, zero_ref_inputs
        )

        random_ref_inputs = make_retrieval_inputs_for_mode(batch["retrieval_inputs"], "random")
        out_random_refs = model_noise_pred(
            model, batch, noisy_target_images, timesteps, args, random_ref_inputs
        )

        eval_mode_metrics = {
            "correct": {
                "diff_loss": F.mse_loss(out_correct.float(), noise.float(), reduction="mean").item(),
                "mean_abs_diff_vs_correct": 0.0,
                "mean_abs_diff_vs_alpha_zero": tensor_mean_abs_diff(out_correct, out_alpha_zero),
            },
            "shuffled": {
                "diff_loss": F.mse_loss(out_shuffled.float(), noise.float(), reduction="mean").item(),
                "mean_abs_diff_vs_correct": tensor_mean_abs_diff(out_correct, out_shuffled),
                "mean_abs_diff_vs_alpha_zero": tensor_mean_abs_diff(out_shuffled, out_alpha_zero),
            },
            "zero": {
                "diff_loss": F.mse_loss(out_zero_refs.float(), noise.float(), reduction="mean").item(),
                "mean_abs_diff_vs_correct": tensor_mean_abs_diff(out_correct, out_zero_refs),
                "mean_abs_diff_vs_alpha_zero": tensor_mean_abs_diff(out_zero_refs, out_alpha_zero),
            },
            "random": {
                "diff_loss": F.mse_loss(out_random_refs.float(), noise.float(), reduction="mean").item(),
                "mean_abs_diff_vs_correct": tensor_mean_abs_diff(out_correct, out_random_refs),
                "mean_abs_diff_vs_alpha_zero": tensor_mean_abs_diff(out_random_refs, out_alpha_zero),
            },
        }
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
    for mode, summary in eval_mode_metrics.items():
        print(
            f"eval_mode[{mode}]: diff_loss={summary['diff_loss']:.6f} "
            f"mean_abs_diff_vs_correct={summary['mean_abs_diff_vs_correct']:.8f} "
            f"mean_abs_diff_vs_alpha_zero={summary['mean_abs_diff_vs_alpha_zero']:.8f}"
        )
    for mode, summary in mode_summary.items():
        if summary["count"] == 0:
            print(f"mode_summary[{mode}]: count=0")
        else:
            print(
                f"mode_summary[{mode}]: count={summary['count']} "
                f"mean_loss={summary['mean_loss']:.6f} "
                f"mean_objective={summary['mean_objective']:.6f} "
                f"mean_diff={summary['mean_diff']:.6f} "
                f"mean_baseline={summary['mean_baseline']:.6f} "
                f"mean_delta_reg={summary['mean_delta_reg']:.6e} "
                f"mean_alpha_reg={summary['mean_alpha_reg']:.6e}"
            )

    metrics = {
        "steps": cli_args.steps,
        "resample_noise": cli_args.resample_noise,
        "adapter_scale": cli_args.adapter_scale,
        "offset_scale": cli_args.offset_scale,
        "direct_scale": cli_args.direct_scale,
        "retrieval_mode_augmentation": cli_args.retrieval_mode_augmentation,
        "mode_probabilities": {
            "correct": cli_args.p_correct,
            "shuffled": cli_args.p_shuffled,
            "zero": cli_args.p_zero,
            "random": cli_args.p_random,
        },
        "lambda_delta": cli_args.lambda_delta,
        "lambda_alpha": cli_args.lambda_alpha,
        "wrong_ref_target": cli_args.wrong_ref_target,
        "mode_summary": mode_summary,
        "eval_mode_metrics": eval_mode_metrics,
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
        torch.save(retrieval_state_dict(model), cli_args.output_dir / "retrieval_bundle.pth")


if __name__ == "__main__":
    main()
