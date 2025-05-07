source /mnt/local/shared/michaelw/venvs/miniconda3/etc/profile.d/conda.sh
conda activate openai_evals

python difficulty.py \
    --sample_size 3000 \
    --out_file results/deepmath_rating_baseline.jsonl \
    --input_file /mnt/local/shared/michaelw/inference-time-compute/itc-data-efforts/qwen3_spike/raw_json/deepmath-103k.json \
    --concurrency 30


python difficulty.py \
    --sample_size 3000 \
    --out_file results/nemotron_rating_baseline.jsonl \
    --input_file /mnt/local/shared/michaelw/inference-time-compute/itc-data-efforts/qwen3_spike/raw_json/nemotron_posttrain_science.json \
    --concurrency 20


python difficulty.py \
    --sample_size 3000 \
    --out_file results/numina_rating_baseline.jsonl \
    --input_file /mnt/local/shared/michaelw/inference-time-compute/itc-data-efforts/qwen3_spike/raw_json/numina_math_1p5_filtered.json \
    --concurrency 30