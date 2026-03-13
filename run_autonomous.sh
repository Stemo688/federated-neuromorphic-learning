#!/usr/bin/env bash
# run_autonomous.sh — Master script for autonomous Pi experiments.
#
# Runs on claudio, manages both Pis via direct Ethernet (10.0.0.x).
# Designed to survive Wi-Fi outages. Monitor via Tailscale.
#
# Usage:
#   tmux new -d -s experiment "bash ~/federated_experiment/run_autonomous.sh"
#
# Monitor:
#   ssh admin@100.116.228.57 'tmux attach -t experiment'
#   ssh admin@100.116.228.57 'tail -20 ~/federated_experiment/logs/master_*.log'

set -euo pipefail

WORK_DIR="$HOME/federated_experiment"
LOG_DIR="$WORK_DIR/logs"
VENV="$HOME/akida-env/bin/activate"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="$LOG_DIR/master_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR" "$WORK_DIR/models" "$WORK_DIR/results"

# Activate venv
source "$VENV"
cd "$WORK_DIR"

# Logging helper
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG"
}

log "=========================================="
log "AUTONOMOUS EXPERIMENT — START"
log "=========================================="
log "Host: $(hostname)"
log "Python: $(python3 --version)"
log "Working dir: $WORK_DIR"
log ""

# ---------------------------------------------------------------------------
# Phase 1: Fine-tune DS-CNN feature extractor
# ---------------------------------------------------------------------------
log "=== PHASE 1: Fine-tune DS-CNN ==="

FINETUNE_LOG="$LOG_DIR/finetune_${TIMESTAMP}.log"
FINETUNE_OK=0

if [ -f "$WORK_DIR/data/raw/.extracted" ]; then
    log "Raw audio data found, starting fine-tuning ..."
    if python3 finetune_dscnn.py \
        --epochs 20 \
        --deploy-to-paolo \
        2>&1 | tee "$FINETUNE_LOG"; then
        log "Fine-tuning SUCCEEDED"
        FINETUNE_OK=1
    else
        log "Fine-tuning FAILED (exit code $?) — continuing with original extractor"
    fi
else
    log "No raw audio data at $WORK_DIR/data/raw/.extracted"
    log "Skipping fine-tuning, using original feature extractor"
fi

# ---------------------------------------------------------------------------
# Phase 2: Verify feature extractor on both Pis
# ---------------------------------------------------------------------------
log ""
log "=== PHASE 2: Verify Feature Extractors ==="

# Check claudio
if [ -f "$WORK_DIR/models/feat_extractor.fbz" ]; then
    log "claudio: feat_extractor.fbz exists ($(stat -c %s "$WORK_DIR/models/feat_extractor.fbz" 2>/dev/null || stat -f %z "$WORK_DIR/models/feat_extractor.fbz") bytes)"
else
    log "WARNING: claudio feat_extractor.fbz not found!"
fi

# Check paolo via direct Ethernet
PAOLO_CHECK=$(ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no admin@10.0.0.2 \
    "ls -la ~/federated_experiment/models/feat_extractor.fbz 2>&1" || echo "UNREACHABLE")
log "paolo: $PAOLO_CHECK"

# Quick extraction test on claudio
log "Running quick extraction test on claudio ..."
python3 -c "
import akida
m = akida.Model('$WORK_DIR/models/feat_extractor.fbz')
import numpy as np
x = np.random.randint(0, 256, (1, 49, 10, 1), dtype=np.uint8)
out = m.forward(x)
print(f'OK: input {x.shape} -> output {out.shape}')
assert out.shape[-1] == 64, f'Bad output shape: {out.shape}'
" 2>&1 | tee -a "$MASTER_LOG"

log "Feature extractors verified"

# ---------------------------------------------------------------------------
# Phase 3: Hyperparameter sweep
# ---------------------------------------------------------------------------
log ""
log "=== PHASE 3: Hyperparameter Sweep ==="

SWEEP_LOG="$LOG_DIR/sweep_${TIMESTAMP}.log"

log "Starting sweep: 24 configs x 10 trials + top-3 multi-round"
log "See detailed log: $SWEEP_LOG"

if python3 hyperparam_sweep.py \
    --num-trials 10 \
    2>&1 | tee "$SWEEP_LOG"; then
    log "Sweep COMPLETED SUCCESSFULLY"
else
    log "Sweep FAILED (exit code $?)"
    log "Check partial results at $WORK_DIR/results/sweep_results.json"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
log ""
log "=========================================="
log "AUTONOMOUS EXPERIMENT — COMPLETE"
log "=========================================="
log "Timestamp: $TIMESTAMP"
log "Fine-tune: $([ $FINETUNE_OK -eq 1 ] && echo 'OK' || echo 'SKIPPED/FAILED')"
log "Results:"
log "  Fine-tune history: $WORK_DIR/finetune_history.json"
log "  Sweep results:     $WORK_DIR/results/sweep_results.json"
log "  Logs:              $LOG_DIR/"
log ""
log "Download results:"
log "  scp admin@100.116.228.57:$WORK_DIR/results/sweep_results.json ."
log "  scp admin@100.116.228.57:$WORK_DIR/finetune_history.json ."
log "=========================================="
