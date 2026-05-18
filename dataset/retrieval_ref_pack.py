import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image


DEFAULT_PLAN_A_ROOT = Path("D:/00_project/callirag/retrieval_data_prepare")

LAYOUT_TO_TARGET_STRUCT = {
    "": 0,
    "⿰": 1,
    "⿱": 2,
    "⿲": 3,
    "⿳": 4,
    "⿴": 5,
    "⿵": 6,
    "⿶": 7,
    "⿷": 8,
    "⿸": 9,
    "⿹": 10,
    "⿺": 11,
    "⿻": 12,
}

ROLE_TO_ID = {
    "anchor": 0,
    "coverage": 1,
    "empty": 2,
}

@dataclass
class RetrievalSlot:
    slot_id: int
    role_id: int
    role: str
    valid: bool
    bank_id: Optional[str]
    ref_char: Optional[str]
    ref_path: Optional[str]
    matched_comp: Optional[str]
    rank: Optional[int]


@dataclass
class RetrievalPack:
    target_char: str
    target_struct: int
    slots: List[RetrievalSlot]

    def as_tensors(self):
        import torch

        return {
            "slot_ids": torch.tensor([slot.slot_id for slot in self.slots], dtype=torch.long),
            "role_ids": torch.tensor([slot.role_id for slot in self.slots], dtype=torch.long),
            "target_struct": torch.tensor(self.target_struct, dtype=torch.long),
            "mask": torch.tensor([slot.valid for slot in self.slots], dtype=torch.bool),
        }


def parse_path_maps(path_maps: Optional[Iterable[str]]) -> List[Tuple[str, str]]:
    maps = []
    for item in path_maps or []:
        if "=" not in item:
            raise ValueError(f"path map must be OLD=NEW, got: {item}")
        old, new = item.split("=", 1)
        maps.append((old, new))
    return maps


def resolve_ref_path(path_text: Optional[str], path_maps: Optional[List[Tuple[str, str]]] = None) -> Optional[Path]:
    if not path_text:
        return None

    candidates = [path_text]
    for old, new in path_maps or []:
        if path_text.startswith(old):
            candidates.append(new + path_text[len(old):])

    if path_text.startswith("/d/"):
        candidates.append("D:/" + path_text[len("/d/"):])

    for candidate in candidates:
        path = Path(candidate)
        if path.is_file():
            return path

    return Path(candidates[-1])


def load_case_manifest(path: Path) -> Dict[str, dict]:
    cases = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            cases[row["target_char"]] = row
    return cases


def load_sim_layer(path: Path) -> Dict[str, list]:
    return json.loads(path.read_text(encoding="utf-8"))


def _slot_from_candidate(target_char: str, candidate: dict, index: int) -> RetrievalSlot:
    ref_char = candidate.get("character")
    if ref_char == target_char:
        raise ValueError(f"target GT leakage detected for {target_char}")

    role = "anchor" if index < 2 else "coverage"
    shared = candidate.get("shared_components") or []
    matched_comp = shared[0] if shared else None
    rank = int(candidate.get("rank", index + 1))
    return RetrievalSlot(
        slot_id=min(rank - 1, 36),
        role_id=ROLE_TO_ID[role],
        role=role,
        valid=True,
        bank_id=candidate.get("bank_id"),
        ref_char=ref_char,
        ref_path=candidate.get("wxz_path"),
        matched_comp=matched_comp,
        rank=rank,
    )


def _empty_slot() -> RetrievalSlot:
    return RetrievalSlot(
        slot_id=36,
        role_id=ROLE_TO_ID["empty"],
        role="empty",
        valid=False,
        bank_id=None,
        ref_char=None,
        ref_path=None,
        matched_comp=None,
        rank=None,
    )


def build_pack_for_target(
    target_char: str,
    case_row: dict,
    sim_layer: Dict[str, list],
    n_slots: int = 5,
) -> RetrievalPack:
    candidates = list(sim_layer.get(target_char, []))
    slots = [_slot_from_candidate(target_char, cand, idx) for idx, cand in enumerate(candidates[:n_slots])]
    while len(slots) < n_slots:
        slots.append(_empty_slot())
    if not any(slot.valid for slot in slots):
        raise ValueError(f"{target_char} has no valid retrieval slot.")

    layout = case_row.get("layout", "")
    target_struct = LAYOUT_TO_TARGET_STRUCT.get(layout, 0)
    return RetrievalPack(target_char=target_char, target_struct=target_struct, slots=slots)


def load_retrieval_packs(
    plan_a_root: Path = DEFAULT_PLAN_A_ROOT,
    n_slots: int = 5,
    target_chars: Optional[Iterable[str]] = None,
) -> Dict[str, RetrievalPack]:
    plan_a_root = Path(plan_a_root)
    cases = load_case_manifest(plan_a_root / "outputs" / "case_manifest.csv")
    sim_layer = load_sim_layer(plan_a_root / "outputs" / "sim_layer.json")
    if target_chars is not None:
        target_set = set(target_chars)
        cases = {target_char: row for target_char, row in cases.items() if target_char in target_set}
    return {
        target_char: build_pack_for_target(target_char, row, sim_layer, n_slots=n_slots)
        for target_char, row in cases.items()
    }


def make_ref_image_transform(resolution: int = 96):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )


def load_ref_images(
    pack: RetrievalPack,
    resolution: int = 96,
    path_maps: Optional[List[Tuple[str, str]]] = None,
):
    import torch

    transform = make_ref_image_transform(resolution)
    tensors = []
    for slot in pack.slots:
        path = resolve_ref_path(slot.ref_path, path_maps)
        if slot.valid and path is not None and path.is_file():
            image = Image.open(path).convert("RGB")
            tensors.append(transform(image))
        else:
            tensors.append(torch.ones(3, resolution, resolution))
    return torch.stack(tensors, dim=0)


def pack_to_model_inputs(
    pack: RetrievalPack,
    resolution: int = 96,
    path_maps: Optional[List[Tuple[str, str]]] = None,
):
    tensor_inputs = pack.as_tensors()
    tensor_inputs["ref_images"] = load_ref_images(pack, resolution=resolution, path_maps=path_maps)
    return tensor_inputs
