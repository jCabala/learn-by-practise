#!/usr/bin/env bash
set -euo pipefail

PYTHON_CMD="python /homes/lp721/aca-gem5/simulate.py"
# Run up to this many jobs in parallel (override via env: MAX_PARALLEL=8 ./part_2.sh)
MAX_PARALLEL="${MAX_PARALLEL:-4}"

PIPELINE_WIDTH=("5" "10" "15" "20" "25" "30" "35" "40")

# Helper to run a single job in background and log its output
run_job() {
  local idx="$1" name="$2" width="$3" logfile="$4"
  {
    echo "=== [$idx] NAME=${name}  PIPELINE_WIDTH=${width} ===" | tee -a "$logfile"
    echo "CMD: $PYTHON_CMD --pipeline-width ${width} --name ${name}" | tee -a "$logfile"

    $PYTHON_CMD \
      --window-size 128,43,43 \
      --pipeline-width "${width}" \
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
for width in "${PIPELINE_WIDTH[@]}"; do
    # compose flags
    NAME="PIPELINE_WIDTH_${width}"

    # log file for this run
    LOGFILE="./out/${count}_${NAME}.log"

    # If log file does not exist, create it
    if [ ! -f "$LOGFILE" ]; then
      echo "Creating log file: $LOGFILE"
      touch "$LOGFILE"
    fi

    # queue the job in background
    run_job "$count" "$NAME" "$width" "$LOGFILE"

    # increment and throttle concurrency
    count=$((count + 1))
    while [ "$(jobs -pr | wc -l)" -ge "$MAX_PARALLEL" ]; do
      wait -n || true
    done
done