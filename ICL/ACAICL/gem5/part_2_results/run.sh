#!/usr/bin/env bash
set -euo pipefail

PYTHON_CMD="python /homes/lp721/aca-gem5/simulate.py"
# Run up to this many jobs in parallel (override via env: MAX_PARALLEL=8 ./part_2.sh)
MAX_PARALLEL="${MAX_PARALLEL:-4}"

# ---- EDIT THESE BLOCKS to match the two columns in the image if needed ----
# Left table (one row per line): ROB IQ LSQ
read -r -d '' LEFT_BLOCK <<'LEFT' || true
16 11 5
32 21 11
64 43 21
128 85 43
256 171 85
512 341 171
# add / remove lines as needed
LEFT

# Right table (one row per line): Local Global BTB RAS
read -r -d '' RIGHT_BLOCK <<'RIGHT' || true
64 128 128 16
128 256 128 16
256 512 256 16
512 1024 512 32
1024 2048 1024 32
2048 4096 2048 64
4096 8192 4096 128
8192 16384 8192 256
# add / remove lines as needed
RIGHT
# -------------------------------------------------------------------------

OUTDIR="./runs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"
echo "Logs and command-lines will be written into: $OUTDIR"

# Helper to run a single job in background and log its output
run_job() {
  local idx="$1" name="$2" window="$3" branch="$4" logfile="$5"
  {
    echo "=== [$idx] NAME=${name}  WINDOW=${window}  BRANCH=${branch} ===" | tee -a "$logfile"
    echo "CMD: $PYTHON_CMD --window-size ${window} --branch-pred-size ${branch} --name ${name}" | tee -a "$logfile"
    $PYTHON_CMD \
      --window-size "${window}" \
      --branch-pred-size "${branch}" \
      --name "${name}" \
      2>&1 | tee -a "$logfile"
    echo "Finished [$idx] -> log: $logfile"
  } &
}

# parse LEFT into an array of lines
mapfile -t LEFT_LINES < <(printf "%s\n" "$LEFT_BLOCK" | sed '/^\s*#/d;/^\s*$/d')
mapfile -t RIGHT_LINES < <(printf "%s\n" "$RIGHT_BLOCK" | sed '/^\s*#/d;/^\s*$/d')

count=0
for left in "${LEFT_LINES[@]}"; do
  # split left into parts
  read -r ROB IQ LSQ <<<"$left"

  # sanity check
  if [[ -z "${ROB:-}" || -z "${IQ:-}" || -z "${LSQ:-}" ]]; then
    echo "Skipping malformed LEFT line: '$left'"
    continue
  fi

  for right in "${RIGHT_LINES[@]}"; do
    read -r LOCAL GLOBAL BTB RAS <<<"$right"

    if [[ -z "${LOCAL:-}" || -z "${GLOBAL:-}" || -z "${BTB:-}" || -z "${RAS:-}" ]]; then
      echo "Skipping malformed RIGHT line: '$right'"
      continue
    fi

    # compose flags
    WINDOW="${ROB},${IQ},${LSQ}"
    BRANCH="${LOCAL},${GLOBAL},${BTB},${RAS}"

    # human-readable name (safe chars only)
    NAME="LROB${ROB}_IQ${IQ}_LSQ${LSQ}__BP_L${LOCAL}_G${GLOBAL}_BTB${BTB}_RAS${RAS}"

    # log file for this run
    LOGFILE="${OUTDIR}/${count}_${NAME}.log"

    # queue the job in background
    run_job "$count" "$NAME" "$WINDOW" "$BRANCH" "$LOGFILE"

    # increment and throttle concurrency
    count=$((count + 1))
    while [ "$(jobs -pr | wc -l)" -ge "$MAX_PARALLEL" ]; do
      wait -n || true
    done
  done
done

# wait for any remaining background jobs
wait || true
echo "All done. Total runs: $count"
