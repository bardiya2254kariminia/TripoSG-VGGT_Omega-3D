set -euo pipefail

# ── configurable variables ────────────────────────────────────────────────────
CONDA_ENV_NAME="SIA32"
ENV_FILE="hanyuan_vggt-omega_environment.yaml"

HUNYUAN_REPO="https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git"
HUNYUAN_DIR="Hunyuan3D-2"

VGGT_OMEGA_REPO="https://github.com/facebookresearch/vggt-omega.git"
VGGT_OMEGA_DIR="vggt-omega"
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
section() { echo -e "\n${GREEN}========== $* ==========${NC}\n"; }

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

# ── 7. download the VGGT-Omega checkpoint ─────────────────────────────────────
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
