#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# ILGAN Training Monitor
# ──────────────────────────────────────────────────────────────────────────────
# Usage:
#   chmod +x monitor_training.sh
#   ./monitor_training.sh <INSTANCE_IP> <PEM_PATH> [REFRESH_SECONDS]
#
# Example:
#   ./monitor_training.sh 123.456.789.0 ~/my-key.pem 5
#
# What this does:
#   Continuously polls the remote GPU instance and displays:
#   - Last 10 log lines (losses, metrics)
#   - Latest checkpoint sample images (generated + real with boxes)
#   - GPU utilisation and memory
#   - Training speed (images/sec)
#   - Epoch progress
#
# The script refreshes every REFRESH_SECONDS (default: 10).
# It downloads the latest sample images to ./monitor_output/ for local viewing.
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────────────
INSTANCE_IP="${1:?Usage: $0 <INSTANCE_IP> <PEM_PATH> [REFRESH_SECONDS]}"
PEM_PATH="${2:?Usage: $0 <INSTANCE_IP> <PEM_PATH> [REFRESH_SECONDS]}"
REFRESH="${3:-10}"
SSH_USER="ubuntu"
REMOTE_DIR="~/ilgan"
LOCAL_OUTPUT="./monitor_output"

# ── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'
CLR='\033[2J\033[H'  # Clear screen and move cursor home

# ── Pre-flight ──────────────────────────────────────────────────────────────
if [ ! -f "$PEM_PATH" ]; then
    echo -e "${RED}[✗]${NC} PEM file not found: $PEM_PATH"
    exit 1
fi

mkdir -p "$LOCAL_OUTPUT"

# Test SSH
if ! ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -i "$PEM_PATH" "$SSH_USER@$INSTANCE_IP" "echo OK" 2>/dev/null; then
    echo -e "${RED}[✗]${NC} Cannot SSH to $INSTANCE_IP"
    exit 1
fi

# ── Helper: run remote command ──────────────────────────────────────────────
remote() {
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -i "$PEM_PATH" "$SSH_USER@$INSTANCE_IP" "$1" 2>/dev/null
}

# ── Helper: download file ──────────────────────────────────────────────────
download() {
    scp -o StrictHostKeyChecking=no -o ConnectTimeout=5 -i "$PEM_PATH" \
        "$SSH_USER@$INSTANCE_IP:$1" "$2" 2>/dev/null || true
}

# ── Main monitoring loop ────────────────────────────────────────────────────
PREV_STEP=0
PREV_TIME=$(date +%s)

while true; do
    echo -e "$CLR"
    echo -e "${BOLD}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║              ILGAN Training Monitor                            ║${NC}"
    echo -e "${BOLD}║  Instance: $INSTANCE_IP${NC}"
    echo -e "${BOLD}║  Refresh:  every ${REFRESH}s  |  $(date '+%Y-%m-%d %H:%M:%S')${NC}"
    echo -e "${BOLD}╚══════════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    # ── 1. GPU Status ─────────────────────────────────────────────────────
    echo -e "${CYAN}${BOLD}┌─ GPU Status ─────────────────────────────────────────────────┐${NC}"
    GPU_INFO=$(remote "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null" || echo "N/A")
    if [ "$GPU_INFO" != "N/A" ]; then
        echo "$GPU_INFO" | while IFS=',' read -r idx name util mem_used mem_total temp; do
            # Trim whitespace
            idx=$(echo "$idx" | xargs)
            name=$(echo "$name" | xargs)
            util=$(echo "$util" | xargs)
            mem_used=$(echo "$mem_used" | xargs)
            mem_total=$(echo "$mem_total" | xargs)
            temp=$(echo "$temp" | xargs)

            # Color-code utilisation
            util_num=${util%\%}
            if [ "$util_num" -gt 90 ]; then
                util_color="${GREEN}${util}${NC}"
            elif [ "$util_num" -gt 50 ]; then
                util_color="${YELLOW}${util}${NC}"
            else
                util_color="${RED}${util}${NC}"
            fi

            echo -e "  │  GPU $idx: ${name:0:30} | Util: $util_color | Mem: ${mem_used}/${mem_total} | Temp: ${temp}°C"
        done
    else
        echo -e "  │  ${RED}No GPU data (nvidia-smi not available)${NC}"
    fi
    echo -e "${CYAN}└──────────────────────────────────────────────────────────────────┘${NC}"
    echo ""

    # ── 2. Training Progress ──────────────────────────────────────────────
    echo -e "${BLUE}${BOLD}┌─ Training Progress ──────────────────────────────────────────┐${NC}"

    # Get the last 10 log lines
    LOG_LINES=$(remote "tail -n 10 ${REMOTE_DIR}/logs/training.log 2>/dev/null" || echo "No log file yet")
    echo -e "  │  ${YELLOW}Last 10 log lines:${NC}"
    echo "$LOG_LINES" | while IFS= read -r line; do
        if [ -n "$line" ]; then
            # Highlight loss values
            colored=$(echo "$line" | sed -E \
                -e 's/([0-9]+\.[0-9]{4,6})/\x1b[33m\1\x1b[0m/g' \
                -e 's/(Epoch [0-9]+)/\x1b[36m\1\x1b[0m/g' \
                -e 's/(ERROR|WARNING|CRITICAL)/\x1b[31m\1\x1b[0m/g')
            echo -e "  │  $colored"
        fi
    done
    echo -e "${BLUE}└──────────────────────────────────────────────────────────────────┘${NC}"
    echo ""

    # ── 3. Training Speed ─────────────────────────────────────────────────
    echo -e "${GREEN}${BOLD}┌─ Training Speed ─────────────────────────────────────────────┐${NC}"

    # Get current step from log
    CURRENT_STEP=$(remote "grep -oP 'Step \K[0-9]+' ${REMOTE_DIR}/logs/training.log 2>/dev/null | tail -1" || echo "0")
    CURRENT_EPOCH=$(remote "grep -oP 'Epoch \K[0-9]+' ${REMOTE_DIR}/logs/training.log 2>/dev/null | tail -1" || echo "0")
    TOTAL_EPOCHS=$(remote "grep -oP 'Epochs: \K[0-9]+' ${REMOTE_DIR}/logs/training.log 2>/dev/null | head -1" || echo "2000")

    NOW=$(date +%s)
    if [ "$CURRENT_STEP" -gt "$PREV_STEP" ] 2>/dev/null; then
        STEP_DIFF=$((CURRENT_STEP - PREV_STEP))
        TIME_DIFF=$((NOW - PREV_TIME))
        if [ "$TIME_DIFF" -gt 0 ]; then
            SPEED=$(echo "scale=1; $STEP_DIFF / $TIME_DIFF" | bc)
            echo -e "  │  Steps/sec:    ${BOLD}${SPEED}${NC}"
            echo -e "  │  Current step: ${BOLD}${CURRENT_STEP}${NC}"
            echo -e "  │  Current epoch: ${BOLD}${CURRENT_EPOCH}${NC} / ${TOTAL_EPOCHS}"
            if [ "$TOTAL_EPOCHS" -gt 0 ] 2>/dev/null; then
                PCT=$(echo "scale=1; $CURRENT_EPOCH * 100 / $TOTAL_EPOCHS" | bc)
                echo -e "  │  Progress:     ${BOLD}${PCT}%${NC}"
            fi
        fi
    else
        echo -e "  │  ${YELLOW}Waiting for training to start...${NC}"
    fi
    PREV_STEP=$CURRENT_STEP
    PREV_TIME=$NOW

    # Check if training is still running
    TMUX_RUNNING=$(remote "tmux has-session -t ilgan 2>/dev/null && echo 'yes' || echo 'no'" || echo "no")
    if [ "$TMUX_RUNNING" = "no" ]; then
        echo -e "  │  ${RED}⚠  Training session is NOT running!${NC}"
        echo -e "  │  ${YELLOW}Check logs for errors.${NC}"
    else
        echo -e "  │  ${GREEN}✓  Training session active${NC}"
    fi
    echo -e "${GREEN}└──────────────────────────────────────────────────────────────────┘${NC}"
    echo ""

    # ── 4. Checkpoints ───────────────────────────────────────────────────
    echo -e "${YELLOW}${BOLD}┌─ Checkpoints ───────────────────────────────────────────────┐${NC}"
    CKPT_INFO=$(remote "ls -lh ${REMOTE_DIR}/checkpoints/*.pt 2>/dev/null | tail -5" || echo "No checkpoints yet")
    if [ "$CKPT_INFO" != "No checkpoints yet" ]; then
        echo "$CKPT_INFO" | while IFS= read -r line; do
            echo -e "  │  $line"
        done
    else
        echo -e "  │  ${YELLOW}No checkpoints saved yet${NC}"
    fi
    echo -e "${YELLOW}└──────────────────────────────────────────────────────────────────┘${NC}"
    echo ""

    # ── 5. Latest Sample Images ──────────────────────────────────────────
    echo -e "${CYAN}${BOLD}┌─ Latest Sample Images ───────────────────────────────────────┐${NC}"

    # Find the latest sample files
    LATEST_GEN_GRID=$(remote "ls -t ${REMOTE_DIR}/samples/gen_grid_*.png 2>/dev/null | head -1" || echo "")
    LATEST_GEN_BOXES=$(remote "ls -t ${REMOTE_DIR}/samples/gen_boxes_*.png 2>/dev/null | head -1" || echo "")
    LATEST_REAL_BOXES=$(remote "ls -t ${REMOTE_DIR}/samples/real_boxes_*.png 2>/dev/null | head -1" || echo "")
    LATEST_LOSS_PLOT=$(remote "ls -t ${REMOTE_DIR}/samples/loss_curves_*.png 2>/dev/null | head -1" || echo "")

    if [ -n "$LATEST_GEN_GRID" ]; then
        echo -e "  │  Downloading latest samples..."
        download "$LATEST_GEN_GRID" "${LOCAL_OUTPUT}/latest_gen_grid.png" && \
            echo -e "  │  ${GREEN}✓${NC} Generated grid:  ${LOCAL_OUTPUT}/latest_gen_grid.png"
        download "$LATEST_GEN_BOXES" "${LOCAL_OUTPUT}/latest_gen_boxes.png" && \
            echo -e "  │  ${GREEN}✓${NC} Generated boxes: ${LOCAL_OUTPUT}/latest_gen_boxes.png"
        download "$LATEST_REAL_BOXES" "${LOCAL_OUTPUT}/latest_real_boxes.png" && \
            echo -e "  │  ${GREEN}✓${NC} Real boxes:      ${LOCAL_OUTPUT}/latest_real_boxes.png"

        # Show file sizes and timestamps
        echo -e "  │"
        for f in "$LOCAL_OUTPUT"/latest_*.png; do
            if [ -f "$f" ]; then
                size=$(du -h "$f" | cut -f1)
                mtime=$(stat -f "%Sm" "$f" 2>/dev/null || stat -c "%y" "$f" 2>/dev/null || echo "")
                echo -e "  │  $(basename $f): ${size} (${mtime})"
            fi
        done
    else
        echo -e "  │  ${YELLOW}No sample images generated yet${NC}"
        echo -e "  │  Samples are saved every --sample-interval steps"
    fi

    # Download loss curves if available
    if [ -n "$LATEST_LOSS_PLOT" ]; then
        download "$LATEST_LOSS_PLOT" "${LOCAL_OUTPUT}/latest_loss_curves.png"
        echo -e "  │  ${GREEN}✓${NC} Loss curves:     ${LOCAL_OUTPUT}/latest_loss_curves.png"
    fi
    echo -e "${CYAN}└──────────────────────────────────────────────────────────────────┘${NC}"
    echo ""

    # ── 6. Disk Usage ────────────────────────────────────────────────────
    echo -e "${BLUE}${BOLD}┌─ Disk Usage ─────────────────────────────────────────────────┐${NC}"
    DISK_INFO=$(remote "df -h ${REMOTE_DIR} 2>/dev/null | tail -1" || echo "")
    if [ -n "$DISK_INFO" ]; then
        echo -e "  │  $DISK_INFO"
    fi
    CKPT_SIZE=$(remote "du -sh ${REMOTE_DIR}/checkpoints/ 2>/dev/null" || echo "0")
    LOG_SIZE=$(remote "du -sh ${REMOTE_DIR}/logs/ 2>/dev/null" || echo "0")
    echo -e "  │  Checkpoints: $CKPT_SIZE"
    echo -e "  │  Logs:        $LOG_SIZE"
    echo -e "${BLUE}└──────────────────────────────────────────────────────────────────┘${NC}"
    echo ""

    # ── 7. Quick Actions ─────────────────────────────────────────────────
    echo -e "${BOLD}Actions:${NC}"
    echo -e "  ${GREEN}[r]${NC} Refresh now    ${YELLOW}[l]${NC} View full log    ${CYAN}[s]${NC} SSH into instance"
    echo -e "  ${RED}[q]${NC} Quit monitor   ${BLUE}[d]${NC} Download all samples"
    echo ""
    echo -e "  ${BOLD}Waiting ${REFRESH}s...${NC} (press a key for action)"

    # ── Wait with keypress detection ─────────────────────────────────────
    if read -t "$REFRESH" -n 1 key; then
        case "$key" in
            r|R)
                continue
                ;;
            l|L)
                echo ""
                echo -e "${YELLOW}=== Full Log (last 50 lines) ===${NC}"
                remote "tail -n 50 ${REMOTE_DIR}/logs/training.log 2>/dev/null" || echo "No log"
                echo ""
                echo -e "${YELLOW}Press any key to return to monitor...${NC}"
                read -n 1
                ;;
            s|S)
                echo ""
                echo -e "${CYAN}Opening SSH session...${NC}"
                echo -e "${YELLOW}(Type 'exit' to return to monitor)${NC}"
                ssh -o StrictHostKeyChecking=no -i "$PEM_PATH" "$SSH_USER@$INSTANCE_IP"
                ;;
            d|D)
                echo ""
                echo -e "${BLUE}Downloading all samples...${NC}"
                mkdir -p "${LOCAL_OUTPUT}/all_samples"
                remote "ls ${REMOTE_DIR}/samples/*.png 2>/dev/null" | while IFS= read -r f; do
                    if [ -n "$f" ]; then
                        download "$f" "${LOCAL_OUTPUT}/all_samples/"
                    fi
                done
                echo -e "${GREEN}✓${NC} Downloaded to ${LOCAL_OUTPUT}/all_samples/"
                echo -e "${YELLOW}Press any key to continue...${NC}"
                read -n 1
                ;;
            q|Q)
                echo ""
                echo -e "${RED}Exiting monitor.${NC}"
                exit 0
                ;;
        esac
    fi
done
