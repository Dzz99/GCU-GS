#!/usr/bin/env bash

set -o pipefail

# The directory containing the LLFF scene folders. Override it at launch with:
# BASE_DATA=/path/to/LLFF bash scripts/train_llff.sh
BASE_DATA="${BASE_DATA:-/path/to/LLFF}"

SCENES=("fern" "flower" "fortress" "horns" "leaves" "orchids" "room" "trex")

# Format: output_directory|additional training arguments
EXPERIMENTS=(
    "test|--n_views 3 --interval 10 --geom_densify"
)

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${BASE_DATA}/logs"
mkdir -p "${LOG_DIR}"

TOTAL_LOG="${LOG_DIR}/run_llff_experiments_${TIMESTAMP}.log"


total=0
success=0

for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r model_suffix extra_flags <<< "${exp}"



    for scene in "${SCENES[@]}"; do
        SCENE_PATH="${BASE_DATA}/${scene}"
        OUTPUT_PATH="${SCENE_PATH}/${model_suffix}"
        SCENE_LOG="${LOG_DIR}/${scene}_${model_suffix}_${TIMESTAMP}.log"

        if [ ! -d "${SCENE_PATH}" ]; then
            echo "[SKIP] Input path does not exist: ${SCENE_PATH}" | tee -a "${TOTAL_LOG}"
            continue
        fi

        if [ -d "${OUTPUT_PATH}" ]; then
            echo "[SKIP] Output already exists: ${OUTPUT_PATH}" | tee -a "${TOTAL_LOG}"
            continue
        fi

        (

            # extra_flags is intentionally expanded into separate CLI arguments.
            # shellcheck disable=SC2086
            python train.py \
                -s "${SCENE_PATH}" \
                -m "${OUTPUT_PATH}" \
                --eval \
                -r 8 \
                ${extra_flags}

            ret=$?
            echo "============================================"
            echo "Finished at: $(date)"

            if [ ${ret} -eq 0 ]; then
                echo "[SUCCESS] Training completed"
                python render.py -m "${OUTPUT_PATH}" --iteration 10000
                python metrics.py -m "${OUTPUT_PATH}"
            else
                echo "[FAILED] Training exited with code ${ret}"
            fi

            exit ${ret}
        ) 2>&1 | tee "${SCENE_LOG}"

        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            echo "[SUCCESS] ${scene}/${model_suffix}" | tee -a "${TOTAL_LOG}"
            ((success++))
        else
            echo "[FAILED] ${scene}/${model_suffix}" | tee -a "${TOTAL_LOG}"
        fi

        ((total++))
        echo "----------------------------------------" | tee -a "${TOTAL_LOG}"
    done
done

