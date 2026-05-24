import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


GROUPS_4 = [
    ("GT", None),
    ("baseline", "baseline_no_adapter"),
    ("correct", "adapter_correct_refs"),
    ("alpha=0", "adapter_alpha_zero"),
]

GROUPS_7 = [
    ("GT", None),
    ("baseline", "baseline_no_adapter"),
    ("correct", "adapter_correct_refs"),
    ("alpha=0", "adapter_alpha_zero"),
    ("shuffled", "adapter_shuffled_refs"),
    ("zero", "adapter_zero_refs"),
    ("random", "adapter_random_refs"),
]


def load_manifest_records(infer_dir):
    manifest_path = infer_dir / "inference_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing inference manifest: {manifest_path}")
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_index = {}
    for record in records:
        idx = record["index"]
        by_index.setdefault(idx, {"target_char": record["target_char"], "groups": {}})
        by_index[idx]["target_image_path"] = record["target_image_path"]
        by_index[idx]["groups"][record["group"]] = Path(record["output_path"])
    return by_index


def resolve_image_path(path, infer_dir):
    path = Path(path)
    if path.is_file():
        return path
    candidate = infer_dir.parent / path
    if candidate.is_file():
        return candidate
    candidate = infer_dir / path.name
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(path)


def load_cell_image(path, cell_size):
    image = Image.open(path).convert("RGB")
    return image.resize((cell_size, cell_size), Image.Resampling.BILINEAR)


def build_comparison_grid(cases, group_defs, cell_size, header_h, row_label_w):
    n_rows = len(cases)
    n_cols = len(group_defs)
    width = row_label_w + n_cols * cell_size
    height = header_h + n_rows * cell_size
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    for col_idx, (label, _) in enumerate(group_defs):
        x = row_label_w + col_idx * cell_size + 4
        draw.text((x, 4), label, fill=(0, 0, 0), font=font)

    for row_idx, case in enumerate(cases):
        y = header_h + row_idx * cell_size
        draw.text((4, y + cell_size // 2 - 8), f"{case['index']:02d}", fill=(0, 0, 0), font=font)
        for col_idx, (_, group_name) in enumerate(group_defs):
            x = row_label_w + col_idx * cell_size
            if group_name is None:
                image_path = case["target_image_path"]
            else:
                image_path = case["groups"][group_name]
            cell = load_cell_image(image_path, cell_size)
            canvas.paste(cell, (x, y))

    return canvas


def main():
    parser = argparse.ArgumentParser(description="Build inference comparison collages.")
    parser.add_argument(
        "--infer-dir",
        type=Path,
        required=True,
        help="Inference output directory containing group subfolders.",
    )
    parser.add_argument("--cell-size", type=int, default=96)
    parser.add_argument("--header-h", type=int, default=24)
    parser.add_argument("--row-label-w", type=int, default=28)
    parser.add_argument("--max-cases", type=int, default=None)
    cli_args = parser.parse_args()

    infer_dir = cli_args.infer_dir.resolve()
    by_index = load_manifest_records(infer_dir)
    indices = sorted(by_index)
    if cli_args.max_cases is not None:
        indices = indices[: cli_args.max_cases]

    cases = []
    for idx in indices:
        entry = by_index[idx]
        groups = {}
        for _, group_name in GROUPS_7:
            if group_name is None:
                continue
            groups[group_name] = resolve_image_path(entry["groups"][group_name], infer_dir)
        cases.append(
            {
                "index": idx,
                "target_char": entry["target_char"],
                "target_image_path": Path(entry["target_image_path"]),
                "groups": groups,
            }
        )

    for suffix, group_defs in (
        ("4groups", GROUPS_4),
        ("7groups", GROUPS_7),
    ):
        out_path = infer_dir / f"comparison_{len(cases)}cases_{suffix}.png"
        grid = build_comparison_grid(
            cases,
            group_defs,
            cell_size=cli_args.cell_size,
            header_h=cli_args.header_h,
            row_label_w=cli_args.row_label_w,
        )
        grid.save(out_path)
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
