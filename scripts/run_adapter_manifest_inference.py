import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.retrieval_ref_pack import DEFAULT_PLAN_A_ROOT, load_retrieval_packs, parse_path_maps
from scripts.train_adapter_tiny_overfit import (
    build_args,
    clone_retrieval_inputs,
    load_model,
    load_tiny_manifest,
    make_batch,
    make_transform,
)
from src import build_ddpm_scheduler
from src.dpm_solver.dpm_solver_pytorch import DPM_Solver, NoiseScheduleVP


def load_retrieval_bundle(model, bundle_path):
    bundle = torch.load(bundle_path, map_location="cpu")
    block = model.unet.up_blocks[2]
    if "retrieval_adapter" in bundle:
        block.retrieval_adapter.load_state_dict(bundle["retrieval_adapter"])
        if "retrieval_res_projs" in bundle:
            block.retrieval_res_projs.load_state_dict(bundle["retrieval_res_projs"])
        if "retrieval_offset_scale" in bundle:
            block.retrieval_offset_scale = bundle["retrieval_offset_scale"]
        if "retrieval_direct_scale" in bundle:
            block.retrieval_direct_scale = bundle["retrieval_direct_scale"]
    else:
        block.retrieval_adapter.load_state_dict(bundle)
    return block


def save_tensor_image(tensor, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = (tensor / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(1, 2, 0).numpy()
    image = (image * 255).round().astype("uint8")
    from PIL import Image

    Image.fromarray(image).save(path)


def model_time_from_continuous(t_continuous, total_n):
    return (t_continuous - 1.0 / total_n) * 1000.0


def noise_pred_cfg(
    model,
    x,
    t_continuous,
    batch,
    args,
    retrieval_inputs,
    guidance_scale,
    total_n,
):
    t_input = model_time_from_continuous(t_continuous, total_n)
    if guidance_scale == 1.0:
        return model(
            x_t=x,
            timesteps=t_input,
            style_images=batch["style_images"],
            content_images=batch["content_images"],
            content_encoder_downsample_size=args.content_encoder_downsample_size,
            retrieval_inputs=retrieval_inputs,
        )[0]

    ones_content = torch.ones_like(batch["content_images"])
    ones_style = torch.ones_like(batch["style_images"])
    noise_uncond = model(
        x_t=x,
        timesteps=t_input,
        style_images=ones_style,
        content_images=ones_content,
        content_encoder_downsample_size=args.content_encoder_downsample_size,
        retrieval_inputs=None,
    )[0]
    noise_cond = model(
        x_t=x,
        timesteps=t_input,
        style_images=batch["style_images"],
        content_images=batch["content_images"],
        content_encoder_downsample_size=args.content_encoder_downsample_size,
        retrieval_inputs=retrieval_inputs,
    )[0]
    return noise_uncond + guidance_scale * (noise_cond - noise_uncond)


def make_group_retrieval_inputs(group, base_inputs):
    if group == "baseline_no_adapter":
        return None
    inputs = clone_retrieval_inputs(base_inputs)
    if group == "adapter_correct_refs" or group == "adapter_alpha_zero":
        return inputs
    if group == "adapter_shuffled_refs":
        inputs["ref_images"] = torch.roll(inputs["ref_images"], shifts=1, dims=1)
        return inputs
    if group == "adapter_zero_refs":
        inputs["ref_images"] = torch.zeros_like(inputs["ref_images"])
        return inputs
    if group == "adapter_random_refs":
        inputs["ref_images"] = torch.randn_like(inputs["ref_images"])
        return inputs
    raise ValueError(f"Unknown inference group: {group}")


def sample_one(
    model,
    train_scheduler,
    batch,
    args,
    retrieval_inputs,
    x_t,
    num_steps,
    order,
    guidance_scale,
):
    noise_schedule = NoiseScheduleVP(schedule="discrete", betas=train_scheduler.betas)
    total_n = noise_schedule.total_N

    def model_fn(x, t_continuous):
        return noise_pred_cfg(
            model=model,
            x=x,
            t_continuous=t_continuous,
            batch=batch,
            args=args,
            retrieval_inputs=retrieval_inputs,
            guidance_scale=guidance_scale,
            total_n=total_n,
        )

    solver = DPM_Solver(
        model_fn=model_fn,
        noise_schedule=noise_schedule,
        algorithm_type="dpmsolver++",
        correcting_x0_fn=None,
    )
    return solver.sample(
        x=x_t,
        steps=num_steps,
        order=order,
        skip_type="time_uniform",
        method="multistep",
    )


def set_alpha(model, value):
    adapter = model.unet.up_blocks[2].retrieval_adapter
    old_value = adapter.alpha.detach().clone()
    adapter.alpha.data.fill_(value)
    return old_value


def restore_alpha(model, value):
    model.unet.up_blocks[2].retrieval_adapter.alpha.data.copy_(value)


def main():
    parser = argparse.ArgumentParser(description="Run retrieval-adapter manifest inference.")
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--retrieval-bundle", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--plan-a-root", type=Path, default=DEFAULT_PLAN_A_ROOT)
    parser.add_argument("--path-map", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/adapter_manifest_inference"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--adapter-scale", type=float, default=10.0)
    parser.add_argument("--offset-scale", type=float, default=0.0)
    parser.add_argument("--direct-scale", type=float, default=1.0)
    parser.add_argument(
        "--groups",
        default="baseline_no_adapter,adapter_correct_refs,adapter_alpha_zero,adapter_shuffled_refs,adapter_zero_refs,adapter_random_refs",
    )
    cli_args = parser.parse_args()

    device = torch.device(cli_args.device if torch.cuda.is_available() else "cpu")
    model_args = build_args(cli_args)
    model = load_model(
        model_args,
        device=device,
        adapter_scale=cli_args.adapter_scale,
        offset_scale=cli_args.offset_scale,
        direct_scale=cli_args.direct_scale,
    )
    load_retrieval_bundle(model, cli_args.retrieval_bundle)
    block = model.unet.up_blocks[2]
    block.retrieval_offset_scale = cli_args.offset_scale
    block.retrieval_direct_scale = cli_args.direct_scale
    model.eval()

    rows = load_tiny_manifest(cli_args.manifest)
    path_maps = parse_path_maps(cli_args.path_map)
    packs = load_retrieval_packs(cli_args.plan_a_root)
    image_transform = make_transform(model_args.resolution)
    train_scheduler = build_ddpm_scheduler(model_args)
    groups = [group.strip() for group in cli_args.groups.split(",") if group.strip()]

    manifest_records = []
    with torch.no_grad():
        for idx, row in enumerate(rows):
            target_char = row["target_char"]
            batch = make_batch([row], packs, path_maps, image_transform, device)
            generator = torch.Generator(device=device)
            generator.manual_seed(cli_args.seed + idx)
            x_t = torch.randn(
                (1, 3, model_args.resolution, model_args.resolution),
                generator=generator,
                device=device,
            )

            for group in groups:
                retrieval_inputs = make_group_retrieval_inputs(group, batch["retrieval_inputs"])
                old_alpha = None
                if group == "adapter_alpha_zero":
                    old_alpha = set_alpha(model, 0.0)
                sample = sample_one(
                    model=model,
                    train_scheduler=train_scheduler,
                    batch=batch,
                    args=model_args,
                    retrieval_inputs=retrieval_inputs,
                    x_t=x_t.clone(),
                    num_steps=cli_args.num_steps,
                    order=cli_args.order,
                    guidance_scale=cli_args.guidance_scale,
                )
                if old_alpha is not None:
                    restore_alpha(model, old_alpha)

                out_path = cli_args.output_dir / group / f"{idx:03d}_{target_char}.png"
                save_tensor_image(sample[0], out_path)
                manifest_records.append(
                    {
                        "index": idx,
                        "target_char": target_char,
                        "group": group,
                        "output_path": str(out_path),
                        "content_image_path": row["content_image_path"],
                        "style_image_path": row["style_image_path"],
                        "target_image_path": row["target_image_path"],
                    }
                )
                print(f"[saved] {group} {target_char}: {out_path}")

    cli_args.output_dir.mkdir(parents=True, exist_ok=True)
    (cli_args.output_dir / "inference_manifest.json").write_text(
        json.dumps(manifest_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    config = vars(cli_args).copy()
    config["device_resolved"] = str(device)
    (cli_args.output_dir / "inference_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
