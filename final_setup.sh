#!/usr/bin/env bash
# =============================================================================
# final_setup.sh — Full Hunyuan3D-2 (hanyuan) + TripoSG + MV-Adapter setup
#
# Usage:
#   bash /workspace/TripoSG-VGGT_Omega-3D/final_setup.sh
#
# What it does (in order):
#   1.  Creates the "hanyuan" conda env from final_env.yml
#   2.  Installs PyTorch 2.5.1 (cu121 wheels — torch>=2.4 needed for RMSNorm / Flux2Klein)
#   3.  Installs BEN2 at pinned commit
#   4.  Installs hy3dgen from the already-cloned local source (editable)
#   5.  Compiles & installs PyTorch3D (CUDA, --no-build-isolation)
#   6.  Compiles & installs custom_rasterizer (CUDA, --no-build-isolation)
#   7.  Compiles & installs differentiable_renderer (CUDA, --no-build-isolation)
#   8.  Installs system OpenGL/EGL libraries via apt
#   9.  Persists PYOPENGL_PLATFORM=egl to /workspace/.env
#   10. Installs VGGT-Omega (camera pose estimation backbone)
#   11. Installs TripoSG (mesh backbone — geometry generator)
#   12. Installs MV-Adapter (texturizing model — geometry-guided UV texturing)
#       a. Clones the MV-Adapter repo
#       b. Installs it (editable) + nvdiffrast (CUDA, --no-build-isolation)
#       c. Installs cvcuda_cu12 (needed for the UV texture pipeline)
#       d. Downloads RealESRGAN and LaMa checkpoints for the texture pipeline
#   13. Prints a quick sanity-check
# =============================================================================
set -euo pipefail

# ── colour helpers ────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
section() { echo -e "\n${GREEN}══════ $* ══════${NC}\n"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_YML="${SCRIPT_DIR}/final_env.yml"
CONDA_ENV="hanyuan"
CONDA_BIN="/opt/miniforge3/bin/conda"
HY3DGEN_SRC="${SCRIPT_DIR}/src/hy3dgen"
VGGT_DIR="vggt-repo"
VGGT_REPO="https://github.com/facebookresearch/vggt.git"
VGGT_OMEGA_DIR="vggt-omega"
VGGT_OMEGA_REPO="https://github.com/facebookresearch/vggt-omega.git"
# ── init conda in this shell ──────────────────────────────────────────────────
eval "$("${CONDA_BIN}" shell.bash hook)"

# =============================================================================
section "1. Creating conda environment: ${CONDA_ENV}"
# =============================================================================
if conda env list | grep -qE "^${CONDA_ENV}\s"; then
    warn "Environment '${CONDA_ENV}' already exists — skipping creation."
    warn "To recreate it: conda env remove -n ${CONDA_ENV} && rerun this script."
else
    conda env create -n "${CONDA_ENV}" -f "${ENV_YML}"
    info "Environment '${CONDA_ENV}' created."
fi
conda activate "${CONDA_ENV}"

# =============================================================================
section "2. Installing PyTorch 2.5.1 + cu121 wheels"
# =============================================================================
# torch>=2.4 required for torch.nn.RMSNorm (used by Flux2KleinPipeline).
# 2.5.1 is the latest stable cu121 wheel available.
pip install \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
pip install peft jaxtyping typeguard

python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; \
    print(f'  torch {torch.__version__}  CUDA {torch.version.cuda}  GPU: {torch.cuda.get_device_name(0)}')"

# =============================================================================
section "3. Installing BEN2 (background removal, pinned commit)"
# =============================================================================
pip install \
    git+https://github.com/PramaLLC/BEN2.git@41efc6966a6fcaf77dfd90ecd4cf06a1edcd1745

# =============================================================================
section "4. Installing hy3dgen (Hunyuan3D-2) — editable, from local source"
# =============================================================================
# info "Source: ${HY3DGEN_SRC}"
# pip install -e "${HY3DGEN_SRC}"
pip install -e git+https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git@e2df0625bda7efd5f1daba8f3f2a4cb3d9ac85f8#egg=hy3dgen
# =============================================================================
section "5. Compiling & installing PyTorch3D (takes 10–20 min)"
# =============================================================================
pip install --no-build-isolation \
    git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47

# =============================================================================
section "6. Compiling & installing custom_rasterizer"
# =============================================================================
RAST_DIR="${HY3DGEN_SRC}/hy3dgen/texgen/custom_rasterizer"
[ -d "${RAST_DIR}" ] || error "custom_rasterizer not found at ${RAST_DIR}"
pip install --no-build-isolation "${RAST_DIR}"

# =============================================================================
section "7. Compiling & installing differentiable_renderer"
# =============================================================================
DIFF_DIR="${HY3DGEN_SRC}/hy3dgen/texgen/differentiable_renderer"
[ -d "${DIFF_DIR}" ] || error "differentiable_renderer not found at ${DIFF_DIR}"
pip install --no-build-isolation "${DIFF_DIR}"
pip install --no-build-isolation diso
# =============================================================================
section "8. Installing system OpenGL / EGL libraries"
# =============================================================================
apt-get install -y libgl1 libegl1 libopengl0 libgles2

# =============================================================================
section "9. Persisting PYOPENGL_PLATFORM=egl"
# =============================================================================
WORKSPACE_ENV="/workspace/.env"
if grep -q "PYOPENGL_PLATFORM" "${WORKSPACE_ENV}" 2>/dev/null; then
    warn "PYOPENGL_PLATFORM already set in ${WORKSPACE_ENV} — skipping."
else
    echo 'export PYOPENGL_PLATFORM=egl' >> "${WORKSPACE_ENV}"
    info "Added PYOPENGL_PLATFORM=egl to ${WORKSPACE_ENV}"
fi
export PYOPENGL_PLATFORM=egl

# =============================================================================
section "10. Installing VGGT-Omega"
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
if [ ! -d "${VGGT_OMEGA_DIR}" ]; then
    info "Cloning VGGT-Omega repository..."
    git clone "${VGGT_OMEGA_REPO}" "${VGGT_OMEGA_DIR}"
else
    info "VGGT-Omega directory already exists — skipping clone."
fi

cd "${SCRIPT_DIR}/${VGGT_OMEGA_DIR}"

if [ -f "requirements.txt" ]; then
    info "Installing VGGT-Omega requirements (requirements.txt)..."
    pip install -r requirements.txt
fi

info "Installing VGGT-Omega package in editable mode..."
pip install -e .

cd "${SCRIPT_DIR}"
info "VGGT-Omega (vggt_omega) setup complete."

section "10a. Downloading VGGT-Omega checkpoint"
CKPT_DIR="${SCRIPT_DIR}/checkpoints/vggt-omega"
CKPT_FILE="${CKPT_DIR}/vggt_omega_1b_512.pt"
mkdir -p "${CKPT_DIR}"

CKPT_OK=0
if [ -f "${CKPT_FILE}" ]; then
    info "Checkpoint already present — ${CKPT_FILE}"
    CKPT_OK=1
else
    # NOTE: facebook/VGGT-Omega is gated — run `huggingface-cli login` after access approval.
    info "Attempting to download vggt_omega_1b_512.pt into ${CKPT_DIR} ..."
    if command -v hf >/dev/null 2>&1; then
        HF_DL=(hf download)
    elif command -v huggingface-cli >/dev/null 2>&1; then
        HF_DL=(huggingface-cli download)
    else
        HF_DL=(python -m huggingface_hub.cli download)
    fi
    if "${HF_DL[@]}" facebook/VGGT-Omega vggt_omega_1b_512.pt \
            --local-dir "${CKPT_DIR}"; then
        info "Checkpoint downloaded to ${CKPT_FILE}"
        CKPT_OK=1
    else
        warn "Checkpoint download failed (gated repo / not logged in)."
        warn "Run 'huggingface-cli login' after requesting access, then:"
        warn "  hf download facebook/VGGT-Omega vggt_omega_1b_512.pt --local-dir ${CKPT_DIR}"
    fi
fi
# =============================================================================
section "10b. Installing VGGT (Visual Geometry Grounded Transformer)"
# =============================================================================
# VGGT: feed-forward 3D scene reconstruction (cameras, depth, point maps).
# Repo:  https://github.com/facebookresearch/vggt
# Paper: CVPR 2025 Best Paper
#
# Has a pyproject.toml → installs as a proper package.
# Installed with --no-deps to avoid the numpy<2 constraint conflicting with
# the numpy==2.x already in this env (VGGT is compatible with numpy 2.x in
# practice; the pin is conservative).

cd "${SCRIPT_DIR}"
if [ ! -d "${VGGT_DIR}/.git" ]; then
    info "Cloning VGGT → ${VGGT_DIR} ..."
    git clone "${VGGT_REPO}" "${VGGT_DIR}"
    info "Clone complete."
else
    info "VGGT already cloned at ${VGGT_DIR} — skipping."
fi

info "Installing VGGT package (editable, --no-deps to preserve numpy 2.x) ..."
pip install --no-build-isolation --no-deps -e "${SCRIPT_DIR}/${VGGT_DIR}"

python -c "
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
print('  vggt OK')
" || warn "vggt import check failed — verify the install at ${SCRIPT_DIR}/${VGGT_DIR}."

info "VGGT package installed."

# =============================================================================
section "10c. Downloading VGGT checkpoint"
# =============================================================================
# Two checkpoints available on HuggingFace:
#   facebook/VGGT-1B            — research / non-commercial licence
#   facebook/VGGT-1B-Commercial — commercial licence (gated, requires form)
# We download the public one by default; swap the repo ID below for commercial.

VGGT_CKPT_REPO="${VGGT_CKPT_REPO:-facebook/VGGT-1B}"
VGGT_CKPT_DIR="${SCRIPT_DIR}/checkpoints/vggt-1b"
VGGT_CKPT_FILE="${VGGT_CKPT_DIR}/model.pt"
mkdir -p "${VGGT_CKPT_DIR}"

if [ -f "${VGGT_CKPT_FILE}" ]; then
    info "VGGT checkpoint already present — ${VGGT_CKPT_FILE}"
else
    info "Downloading VGGT model.pt from ${VGGT_CKPT_REPO} ..."
    if command -v hf >/dev/null 2>&1; then
        HF_DL=(hf download)
    elif command -v huggingface-cli >/dev/null 2>&1; then
        HF_DL=(huggingface-cli download)
    else
        HF_DL=(python -m huggingface_hub.cli download)
    fi
    if "${HF_DL[@]}" "${VGGT_CKPT_REPO}" model.pt \
            --local-dir "${VGGT_CKPT_DIR}"; then
        info "VGGT checkpoint → ${VGGT_CKPT_FILE}"
    else
        warn "VGGT checkpoint download failed."
        warn "The model can also auto-download at runtime via:"
        warn "  VGGT.from_pretrained('${VGGT_CKPT_REPO}')"
        warn "Or download manually:"
        warn "  hf download ${VGGT_CKPT_REPO} model.pt --local-dir ${VGGT_CKPT_DIR}"
        warn "For commercial use, request access to facebook/VGGT-1B-Commercial then:"
        warn "  set VGGT_CKPT_REPO=facebook/VGGT-1B-Commercial and rerun section 10c."
    fi
fi
# =============================================================================
section "11. Installing TripoSG (official guide: clone + requirements)"
# =============================================================================
# Following https://github.com/VAST-AI-Research/TripoSG :
#   TripoSG has NO setup.py / pyproject.toml — pip install git+... does NOT work.
#   All Python deps from its requirements.txt are already declared in
#   final_env.yml (peft, jaxtyping, typeguard, diso, diffusers, trimesh, …)
#   so no separate `pip install -r requirements.txt` is needed here.
#
#   Steps done below:
#     1. git clone the repo   (provides the `triposg` Python package directory)
#     2. add repo root to PYTHONPATH  (so `from triposg.* import ...` works)
#   Model weights auto-download on first inference run.

TRIPOSG_REPO_DIR="${SCRIPT_DIR}/triposg-repo"
TRIPOSG_REPO_URL="https://github.com/VAST-AI-Research/TripoSG.git"
MVADAPTER_REPO_DIR="${SCRIPT_DIR}/mvadapter-repo"
MVADAPTER_REPO_URL="https://github.com/huanngzh/MV-Adapter.git"

# ── 1. clone ──────────────────────────────────────────────────────────────────
if [ ! -d "${TRIPOSG_REPO_DIR}/.git" ]; then
    info "Cloning TripoSG → ${TRIPOSG_REPO_DIR} ..."
    git clone "${TRIPOSG_REPO_URL}" "${TRIPOSG_REPO_DIR}"
    info "Clone complete."
else
    info "TripoSG already cloned at ${TRIPOSG_REPO_DIR} — skipping."
fi

# ── 2. add repo root to PYTHONPATH so `import triposg` works outside the repo ─
# All Python deps from TripoSG's requirements.txt are already declared in
# final_env.yml (peft, jaxtyping, typeguard, diso, diffusers, trimesh, etc.)
# and were installed when the conda env was created — no extra pip install needed.
if ! python -c "import triposg" 2>/dev/null; then
    export PYTHONPATH="${TRIPOSG_REPO_DIR}:${PYTHONPATH:-}"
    if ! grep -q "triposg-repo" "${WORKSPACE_ENV}" 2>/dev/null; then
        echo "export PYTHONPATH=\"${TRIPOSG_REPO_DIR}:\${PYTHONPATH}\"" >> "${WORKSPACE_ENV}"
        info "Persisted TripoSG PYTHONPATH to ${WORKSPACE_ENV}"
    fi
else
    info "triposg already importable — skipping PYTHONPATH update."
fi

# ── verify ────────────────────────────────────────────────────────────────────
python -c "
from triposg.pipelines.pipeline_triposg import TripoSGPipeline
print('  triposg.pipelines.pipeline_triposg  OK')
" || error "triposg import failed — check ${TRIPOSG_REPO_DIR} and PYTHONPATH."

info "TripoSG ready."
info "  Weights auto-download on first run into:"
info "  ${TRIPOSG_REPO_DIR}/pretrained_weights/TripoSG  (default)"
info "  or pass --triposg_model_path to use a custom location."

# =============================================================================
section "12. Installing MV-Adapter (texturizing model)"
# =============================================================================
# MV-Adapter: geometry-guided multi-view texturizing for existing 3D meshes.
# Repo: https://github.com/huanngzh/MV-Adapter
#
# Steps:
#   a. Clone the repo
#   b. pip install editable (installs mvadapter Python package)
#   c. Compile & install nvdiffrast (required for mesh rendering inside MVAdapter)
#   d. Install cvcuda_cu12 (required for the UV texture pipeline)
#   e. Download RealESRGAN and LaMa checkpoints for the texture pipeline

# ── 12a. Clone ────────────────────────────────────────────────────────────────
if [ ! -d "${MVADAPTER_REPO_DIR}/.git" ]; then
    info "Cloning MV-Adapter → ${MVADAPTER_REPO_DIR} ..."
    git clone "${MVADAPTER_REPO_URL}" "${MVADAPTER_REPO_DIR}"
    info "Clone complete."
else
    info "MV-Adapter already cloned at ${MVADAPTER_REPO_DIR} — skipping."
fi

# ── 12b. Install editable ─────────────────────────────────────────────────────
info "Installing MV-Adapter in editable mode ..."
# Install without optional heavy deps first (cvcuda handled separately below)
pip install -e "${MVADAPTER_REPO_DIR}" --no-deps
# Install the non-CUDA runtime requirements explicitly
pip install \
    controlnet_aux \
    kornia \
    open3d \
    "spandrel==0.4.1" \
    pytorch-lightning \
    gltflib
info "MV-Adapter editable install complete."

# ── 12c. Compile & install nvdiffrast ─────────────────────────────────────────
# nvdiffrast provides differentiable rasterization; MVAdapter uses it for
# rendering mesh views during texture generation.
info "Compiling & installing nvdiffrast (CUDA, --no-build-isolation) ..."
pip install --no-build-isolation \
    git+https://github.com/NVlabs/nvdiffrast.git
info "nvdiffrast installed."

# ── 12d. Install cvcuda_cu12 ─────────────────────────────────────────────────
# CV-CUDA is needed for the TexturePipeline view-upscaling step.
# It is tightly tied to the CUDA version — install the cu12 wheel which covers
# CUDA 12.x toolkit (this image ships PyTorch cu121).
info "Installing cvcuda_cu12 ..."
pip install cvcuda_cu12 || warn "cvcuda_cu12 install failed — texture pipeline upscaling may be unavailable."

# ── 12e. Download texture-pipeline checkpoints ───────────────────────────────
CKPT_DIR="${SCRIPT_DIR}/checkpoints"
mkdir -p "${CKPT_DIR}"

REALESRGAN_CKPT="${CKPT_DIR}/RealESRGAN_x2plus.pth"
LAMA_CKPT="${CKPT_DIR}/big-lama.pt"

if [ ! -f "${REALESRGAN_CKPT}" ]; then
    info "Downloading RealESRGAN_x2plus.pth ..."
    wget -q --show-progress \
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth" \
        -O "${REALESRGAN_CKPT}" \
    && info "RealESRGAN checkpoint → ${REALESRGAN_CKPT}" \
    || warn "RealESRGAN download failed — MVAdapterTexturizer will auto-download on first use."
else
    info "RealESRGAN checkpoint already present — ${REALESRGAN_CKPT}"
fi

if [ ! -f "${LAMA_CKPT}" ]; then
    info "Downloading big-lama.pt ..."
    wget -q --show-progress \
        "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt" \
        -O "${LAMA_CKPT}" \
    && info "LaMa checkpoint → ${LAMA_CKPT}" \
    || warn "LaMa download failed — MVAdapterTexturizer will auto-download on first use."
else
    info "LaMa checkpoint already present — ${LAMA_CKPT}"
fi

# SD2.1 base model for MV-Adapter sd21 variant is auto-downloaded on first use
# by MVAdapterTexturizer (see camera/models/hanyuan.py).

# ── add mvadapter-repo to PYTHONPATH so `from scripts.inference_*` works ──────
if ! python -c "import mvadapter" 2>/dev/null; then
    export PYTHONPATH="${MVADAPTER_REPO_DIR}:${PYTHONPATH:-}"
    if ! grep -q "mvadapter-repo" "${WORKSPACE_ENV}" 2>/dev/null; then
        echo "export PYTHONPATH=\"${MVADAPTER_REPO_DIR}:\${PYTHONPATH}\"" >> "${WORKSPACE_ENV}"
        info "Persisted MV-Adapter PYTHONPATH to ${WORKSPACE_ENV}"
    fi
else
    info "mvadapter already importable — skipping PYTHONPATH update."
fi

python -c "
import mvadapter
print('  mvadapter OK')
from mvadapter.pipelines.pipeline_texture import TexturePipeline
print('  mvadapter.pipelines.pipeline_texture OK')
" || warn "mvadapter import check failed — verify the install at ${MVADAPTER_REPO_DIR}."

info "MV-Adapter ready."
info "  Checkpoints: ${CKPT_DIR}/RealESRGAN_x2plus.pth"
info "              ${CKPT_DIR}/big-lama.pt"
info "  Texture backends now available: 'hunyuan', 'mvadapter', 'triposg'"

# =============================================================================
section "13. Sanity check"
# =============================================================================
python -c "
import torch
print(f'  torch {torch.__version__}  CUDA {torch.version.cuda}  GPU ok: {torch.cuda.is_available()}')

import hy3dgen
print(f'  hy3dgen OK')

try:
    import custom_rasterizer
    print(f'  custom_rasterizer OK')
except Exception as e:
    print(f'  custom_rasterizer FAIL: {e}')

try:
    from hy3dgen.texgen.differentiable_renderer import mesh_processor
    print(f'  differentiable_renderer OK')
except Exception as e:
    print(f'  differentiable_renderer FAIL: {e}')

import pytorch3d
print(f'  pytorch3d {pytorch3d.__version__} OK')

try:
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    print(f'  vggt (VGGT model) OK')
except Exception as e:
    print(f'  vggt FAIL: {e}')

try:
    import vggt_omega
    print(f'  vggt_omega OK')
except Exception as e:
    print(f'  vggt_omega FAIL: {e}')

try:
    from triposg.pipelines.pipeline_triposg import TripoSGPipeline
    print(f'  triposg backbone (TripoSGPipeline) OK')
except Exception as e:
    print(f'  triposg FAIL: {e}')

try:
    import mvadapter
    from mvadapter.pipelines.pipeline_texture import TexturePipeline
    print(f'  mvadapter (TexturePipeline) OK')
except Exception as e:
    print(f'  mvadapter FAIL: {e}')

try:
    import nvdiffrast
    print(f'  nvdiffrast OK')
except Exception as e:
    print(f'  nvdiffrast FAIL: {e}')

import os
print(f'  PYOPENGL_PLATFORM = {os.environ.get(\"PYOPENGL_PLATFORM\", \"NOT SET\")}')
"

echo ""
info "All done. To use the environment:"
info "  eval \"\$(/opt/miniforge3/bin/conda shell.bash hook)\""
info "  conda activate ${CONDA_ENV}"
info "  export PYOPENGL_PLATFORM=egl   # (or source /workspace/.env)"
info ""
info "Texture backends:"
info "  hunyuan   — python mesh/inference.py --image photo.png --texture_backend hunyuan"
info "  mvadapter — python mesh/inference.py --image photo.png --texture_backend mvadapter"
info "  triposg   — python mesh/inference.py --image photo.png --texture_backend triposg"
