#!/usr/bin/env bash
# =============================================================================
# final_setup.sh — Full Hunyuan3D-2 (hanyuan) environment setup
#
# Usage:
#   bash /workspace/SIA3D-Camera-inversion/final_setup.sh
#
# What it does (in order):
#   1.  Creates the "hanyuan" conda env from final_env.yml
#   2.  Installs PyTorch 2.3.1 (cu121 wheels — compatible with driver 12.6)
#   3.  Installs BEN2 at pinned commit
#   4.  Installs hy3dgen from the already-cloned local source (editable)
#   5.  Compiles & installs PyTorch3D (CUDA, --no-build-isolation)
#   6.  Compiles & installs custom_rasterizer (CUDA, --no-build-isolation)
#   7.  Compiles & installs differentiable_renderer (CUDA, --no-build-isolation)
#   8.  Installs system OpenGL/EGL libraries via apt
#   9.  Persists PYOPENGL_PLATFORM=egl to /workspace/.env
#   10. Prints a quick sanity-check
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
section "2. Installing PyTorch 2.3.1 + cu121 wheels"
# =============================================================================
pip install \
    torch==2.3.1 \
    torchvision==0.18.1 \
    torchaudio==2.3.1 \
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

section "7. Downloading VGGT-Omega checkpoint"
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
section "12. Sanity check"
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
    from triposg.pipelines.pipeline_triposg import TripoSGPipeline
    print(f'  triposg (TripoSGPipeline) OK')
except Exception as e:
    print(f'  triposg FAIL: {e}')

import os
print(f'  PYOPENGL_PLATFORM = {os.environ.get(\"PYOPENGL_PLATFORM\", \"NOT SET\")}')
"

echo ""
info "All done. To use the environment:"
info "  eval \"\$(/opt/miniforge3/bin/conda shell.bash hook)\""
info "  conda activate ${CONDA_ENV}"
info "  export PYOPENGL_PLATFORM=egl   # (or source /workspace/.env)"
