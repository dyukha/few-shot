cd ..
task=$1
task_name=$2
tag=${task}_prompt_demo_fitler_auto
export task
export tag
parallel -j4 --lb \
  'echo "Start $tag {1} {2} {3}"; \
  TAG=$tag \
  TYPE=prompt-demo \
  TASK=$task \
  SEED={1} \
  BS={2} \
  LR={3} \
  MODEL=roberta-large \
  CUDA_VISIBLE_DEVICES=$(expr {%} - 1) \
  K=16 \
  bash run_experiment.sh "--template_path auto_template/$task/16-{1}.sort.txt --template_id 0 --demo_filter --demo_filter_model sbert-roberta-large"' \
  ::: 13 21 42 87 100 \
  ::: 2 4 8 \
  ::: 1e-5 2e-5 5e-5 \

python tools/gather_result.py --condition "{'tag': '$tag', 'task_name': '$task_name', 'few_shot_type': 'prompt-demo'}"
