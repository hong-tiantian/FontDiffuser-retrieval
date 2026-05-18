import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dataset.retrieval_ref_pack import (
    DEFAULT_PLAN_A_ROOT,
    load_retrieval_packs,
    parse_path_maps,
    resolve_ref_path,
)


def main():
    parser = argparse.ArgumentParser(description="Inspect Plan A 5-slot retrieval packs.")
    parser.add_argument("--plan-a-root", type=Path, default=DEFAULT_PLAN_A_ROOT)
    parser.add_argument(
        "--targets",
        type=str,
        default=None,
        help="Optional comma-separated target chars. Defaults to every case in case_manifest.csv.",
    )
    parser.add_argument(
        "--path-map",
        action="append",
        default=[],
        help="Optional path remap in OLD=NEW form, e.g. /d/htt/data=D:/htt/data",
    )
    args = parser.parse_args()

    path_maps = parse_path_maps(args.path_map)
    if args.targets:
        target_chars = [item.strip() for item in args.targets.split(",") if item.strip()]
    else:
        target_chars = None
    packs = load_retrieval_packs(args.plan_a_root, target_chars=target_chars)
    missing = 0
    leakage = 0

    for target_char, pack in packs.items():
        print(f"{target_char}: target_struct={pack.target_struct}")
        print(f"  slot_ids={[slot.slot_id for slot in pack.slots]}")
        print(f"  role_ids={[slot.role_id for slot in pack.slots]}")
        print(f"  mask={[slot.valid for slot in pack.slots]}")
        for idx, slot in enumerate(pack.slots):
            resolved = resolve_ref_path(slot.ref_path, path_maps)
            exists = bool(resolved and resolved.is_file())
            if slot.valid and not exists:
                missing += 1
            if slot.ref_char == target_char:
                leakage += 1
            print(
                "  "
                f"[{idx}] role={slot.role} rank={slot.rank} ref={slot.ref_char} "
                f"bank_id={slot.bank_id} comp={slot.matched_comp} exists={exists} path={resolved}"
            )

    print(f"packs: {len(packs)}")
    print(f"missing_valid_ref_images: {missing}")
    print(f"target_gt_leakage_count: {leakage}")
    if leakage:
        raise SystemExit("target GT leakage detected")


if __name__ == "__main__":
    main()
