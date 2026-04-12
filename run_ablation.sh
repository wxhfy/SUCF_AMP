#!/bin/bash
# Run ablation experiments with 5 seeds

ABLATIONS=("A" "B" "C" "D")
SEEDS=(32 37 42 47 52)

for abl in "${ABLATIONS[@]}"; do
  echo "=========================================="
  echo "Running Ablation $abl"
  echo "=========================================="

  for seed in "${SEEDS[@]}"; do
    echo "--- Ablation ${abl}, Seed ${seed} ---"
    config="configs/ablation_${abl}_seed${seed}.yaml"

    if [ -f "$config" ]; then
      conda run -n sucf_run python train_sucf.py --config "$config" 2>&1 | tee "outputs/logs/ablation_${abl}_seed${seed}.log"

      # Extract test MCC
      result_file=$(ls outputs/logs/final_test_results_*_seed${seed}.json 2>/dev/null | head -1)
      if [ -n "$result_file" ]; then
        mcc=$(grep -o '"mcc": [0-9.]*' "$result_file" | head -1 | grep -o '[0-9.]*$')
        echo "Seed ${seed} MCC: $mcc"
      fi
    else
      echo "Config $config not found!"
    fi

    echo "---"
    sleep 5
  done

  echo ""
done

echo "=========================================="
echo "All ablation experiments completed!"
echo "=========================================="
