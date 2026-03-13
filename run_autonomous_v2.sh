#!/usr/bin/env bash
# =============================================================================
# Comprehensive Phase 2 Experiment — Autonomous Launcher
#
# Runs on claudio (192.168.1.52) in a tmux session. Orchestrates all phases:
#   Phase A: Fine-tune 4 feature extractors
#   Phase B: Full sweep (42 configs × 10 trials) + baselines
#   Phase C: Binarization comparison
#   Phase D: Disjoint extractor sweep
#   Phase E: Wide features sweep
#   Phase F: Multi-round federation
#
# Usage:
#   tmux new -d -s experiment "bash ~/federated_experiment/run_autonomous_v2.sh"
#
# Monitor:
#   tmux attach -t experiment
#   tail -f ~/federated_experiment/logs/comprehensive_sweep.log
# =============================================================================

set -euo pipefail

WORK_DIR="$HOME/federated_experiment"
LOG_DIR="$WORK_DIR/logs"
VENV="$HOME/akida-env/bin/activate"
PAOLO_IP="10.0.0.2"

mkdir -p "$LOG_DIR"

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/comprehensive_sweep_${TIMESTAMP}.log"

echo "=== Comprehensive Phase 2 Experiment ===" | tee "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
echo "Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Activate venv
source "$VENV"
cd "$WORK_DIR"

# ---------------------------------------------------------------------------
# Step 1: Kill any leftover workers
# ---------------------------------------------------------------------------
echo "[$(date +%H:%M:%S)] Cleaning up old workers ..." | tee -a "$LOG_FILE"
pkill -f "node_worker.py" 2>/dev/null || true
ssh -o StrictHostKeyChecking=no admin@${PAOLO_IP} "pkill -f 'node_worker.py' 2>/dev/null" || true
sleep 2

# ---------------------------------------------------------------------------
# Step 2: Deploy latest code to paolo
# ---------------------------------------------------------------------------
echo "[$(date +%H:%M:%S)] Deploying code to paolo ..." | tee -a "$LOG_FILE"
for f in node_worker.py baselines_pi.py; do
    scp -o StrictHostKeyChecking=no "$WORK_DIR/$f" "admin@${PAOLO_IP}:~/federated_experiment/" 2>&1 | tee -a "$LOG_FILE"
done

# ---------------------------------------------------------------------------
# Step 3: Run comprehensive sweep (all phases)
# ---------------------------------------------------------------------------
echo "[$(date +%H:%M:%S)] Starting comprehensive sweep ..." | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

python3 comprehensive_sweep.py \
    --phase ALL \
    --num-trials 10 \
    --finetune-epochs 20 \
    --output-dir "$WORK_DIR/results" \
    2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo "" | tee -a "$LOG_FILE"
echo "=== Experiment complete ===" | tee -a "$LOG_FILE"
echo "Finished: $(date)" | tee -a "$LOG_FILE"
echo "Exit code: $EXIT_CODE" | tee -a "$LOG_FILE"
echo "Results: $WORK_DIR/results/comprehensive_sweep.json" | tee -a "$LOG_FILE"

# Create a symlink to latest log
ln -sf "$LOG_FILE" "$LOG_DIR/comprehensive_sweep.log"

exit $EXIT_CODE
