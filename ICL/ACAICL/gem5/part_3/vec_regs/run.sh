#!/usr/bin/env bash
set -euo pipefail

PYTHON_CMD="python /homes/lp721/aca-gem5/simulate.py"
# Run up to this many jobs in parallel (override via env: MAX_PARALLEL=8 ./part_2.sh)
MAX_PARALLEL="${MAX_PARALLEL:-4}"

VEC_REG_SIZES=("50" "60" "70" "80" "90" "100" "128")

# Helper to run a single job in background and log its output
run_job() {
  local idx="$1" name="$2" regs="$3" logfile="$4"
  {
    echo "=== [$idx] NAME=${name}  REGISTERS=${regs} ===" | tee -a "$logfile"
    echo "CMD: $PYTHON_CMD --num-vec-phys-regs ${regs} --name ${name}" | tee -a "$logfile"
    
    $PYTHON_CMD \
      --window-size 128,85,43 \
      --num-vec-phys-regs "${regs}" \
      --name "./out/${name}" \
      2>&1 | tee -a "$logfile"
    echo "Finished [$idx] -> log: $logfile"
  } &
}


# Create output directory
if [ ! -d "./out" ]; then
  mkdir -p "./out"
fi

count=0
for regs in "${VEC_REG_SIZES[@]}"; do
    # compose flags
    NAME="VECREGS_${regs}"

    # log file for this run
    LOGFILE="./out/${count}_${NAME}.log"

    # If log file does not exist, create it
    if [ ! -f "$LOGFILE" ]; then
      echo "Creating log file: $LOGFILE"
      touch "$LOGFILE"
    fi

    # queue the job in background
    run_job "$count" "$NAME" "$regs" "$LOGFILE"

    # increment and throttle concurrency
    count=$((count + 1))
    while [ "$(jobs -pr | wc -l)" -ge "$MAX_PARALLEL" ]; do
      wait -n || true
    done
done