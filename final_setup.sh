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

# ── sanity checks ─────────────────────────────────────────────────────────────
[ -f "${ENV_YML}" ]       || error "final_env.yml not found at ${ENV_YML}"
[ -f "${CONDA_BIN}" ]     || error "conda not found at ${CONDA_BIN}"
[ -d "${HY3DGEN_SRC}" ]   || error "hy3dgen source not found at ${HY3DGEN_SRC} — clone it first"

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
info "Source: ${HY3DGEN_SRC}"
pip install -e "${HY3DGEN_SRC}"

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
section "10. Sanity check"
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

import os
print(f'  PYOPENGL_PLATFORM = {os.environ.get(\"PYOPENGL_PLATFORM\", \"NOT SET\")}')
"

echo ""
info "All done. To use the environment:"
info "  eval \"\$(/opt/miniforge3/bin/conda shell.bash hook)\""
info "  conda activate ${CONDA_ENV}"
info "  export PYOPENGL_PLATFORM=egl   # (or source /workspace/.env)"
