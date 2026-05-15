# Environment Setup For Adapter Work

This repo follows the original FontDiffuser recommendation:

```text
Python 3.9
PyTorch 1.13.1
CUDA 11.7
```

## Option A: One-Step Conda Env

From the repo root:

```powershell
conda env create -f environment-fontdiffuser-cu117.yml
conda activate fontdiffuser
```

Then verify:

```powershell
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); print(torch.cuda.is_available())"
```

`torch.cuda.is_available()` should be `True` on the training machine.

## Option B: Manual Install

```powershell
conda create -n fontdiffuser python=3.9 -y
conda activate fontdiffuser
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
pip install "numpy<2" pillow tqdm
```

## Adapter Smoke Checks

```powershell
python tests/test_retrieval_adapter.py
python tests/test_adapter_integration.py
python scripts/inspect_retrieval_pack.py --path-map /d/htt/data=D:/htt/data
```

If your Wang Xizhi image root is not `D:/htt/data`, change the right side of
the path map.

Expected:

```text
test_retrieval_adapter: OK
test_adapter_integration: OK
target_gt_leakage_count: 0
missing_valid_ref_images: 0
```

## Current Machine Note

This local machine has conda but the FontDiffuser environment is not installed,
so only syntax checks were run here. Run the smoke checks above on the machine
where the environment and data are installed.
