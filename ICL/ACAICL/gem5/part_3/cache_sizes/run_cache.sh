#!/usr/bin/env bash
set -euo pipefail

PYTHON_CMD="python /homes/lp721/aca-gem5/simulate.py"
# Run up to this many jobs in parallel (override via env: MAX_PARALLEL=8 ./part_2.sh)
MAX_PARALLEL="${MAX_PARALLEL:-4}"

CACHE_1_SIZES=("16" "32" "64" "128" "256" "512" "1024" "2048")
CACHE_2_SIZES=("2" "4" "8" "16" "32" "64" "128" "256")
INSTR_CACHE_SIZES=("16" "32" "64" "128" "256" "512" "1024" "2048")

# Helper to run a single job in background and log its output
run_job_cache_1() {
  local idx="$1" name="$2" size="$3"
  {
    # log file for this run
    LOGFILE="./out/c1/${count}_${NAME}.log"

    # If log file does not exist, create it
    if [ ! -f "$LOGFILE" ]; then
      echo "Creating log file: $LOGFILE"
      touch "$LOGFILE"
    fi

    echo "=== [$idx] NAME=${name}  L1_CACHE_SIZE=${size} ===" | tee -a "$LOGFILE"
    echo "CMD: $PYTHON_CMD  --l1-data-size ${size} --name ${name}" | tee -a "$LOGFILE"

    $PYTHON_CMD \
      --window-size 128,85,43 \
      --l1-data-size "${size}" \
      --name "./out/c1/${name}" \
      2>&1 | tee -a "$LOGFILE"
    echo "Finished [$idx] -> log: $LOGFILE"
  } &
}

# Helper to run a single job in background and log its output
run_job_cache_2() {
  local idx="$1" name="$2" size="$3"
  {
    # log file for this run
    LOGFILE="./out/c2/${count}_${NAME}.log"

    # If log file does not exist, create it
    if [ ! -f "$LOGFILE" ]; then
      echo "Creating log file: $LOGFILE"
      touch "$LOGFILE"
    fi

    echo "=== [$idx] NAME=${name}  L2_CACHE_SIZE=${size} ===" | tee -a "$LOGFILE"
    echo "CMD: $PYTHON_CMD  --l2-size ${size} --name ${name}" | tee -a "$LOGFILE"

    $PYTHON_CMD \
      --window-size 128,85,43 \
      --l2-size "${size}" \
      --name "./out/c2/${name}" \
      2>&1 | tee -a "$LOGFILE"
    echo "Finished [$idx] -> log: $LOGFILE"
  } &
}

run_job_instr_cache() {
  local idx="$1" name="$2" size="$3"
  {
    # log file for this run
    LOGFILE="./out/instr/${count}_${NAME}.log"

    # If log file does not exist, create it
    if [ ! -f "$LOGFILE" ]; then
      echo "Creating log file: $LOGFILE"
      touch "$LOGFILE"
    fi

    echo "=== [$idx] NAME=${name}  INSTR_CACHE_SIZE=${size} ===" | tee -a "$LOGFILE"
    echo "CMD: $PYTHON_CMD  --l1-inst-size ${size} --name ${name}" | tee -a "$LOGFILE"

    $PYTHON_CMD \
      --window-size 128,85,43 \
      --l1-inst-size "${size}" \
      --name "./out/instr/${name}" \
      2>&1 | tee -a "$LOGFILE"
    echo "Finished [$idx] -> log: $LOGFILE"
  } &
}


# Create output directory
if [ ! -d "./out" ]; then
  mkdir -p "./out"
fi

count=0
for size in "${CACHE_1_SIZES[@]}"; do
    # compose flags
    NAME="C1_${size}"

    # queue the job in background
    run_job_cache_1 "$count" "$NAME" "$size"

    # increment and throttle concurrency
    count=$((count + 1))
    while [ "$(jobs -pr | wc -l)" -ge "$MAX_PARALLEL" ]; do
      wait -n || true
    done
done

count=0
for size in "${CACHE_2_SIZES[@]}"; do
    # compose flags
    NAME="C2_${size}"
    # queue the job in background
    run_job_cache_2 "$count" "$NAME" "$size"
    # increment and throttle concurrency
    count=$((count + 1))
    while [ "$(jobs -pr | wc -l)" -ge "$MAX_PARALLEL" ]; do
      wait -n || true
    done
done

count=0
for size in "${INSTR_CACHE_SIZES[@]}"; do
    # compose flags
    NAME="INSTR_${size}"
    # queue the job in background
    run_job_instr_cache "$count" "$NAME" "$size"
    # increment and throttle concurrency
    count=$((count + 1))
    while [ "$(jobs -pr | wc -l)" -ge "$MAX_PARALLEL" ]; do
      wait -n || true
    done
done