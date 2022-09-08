cd ..

task=$1
task_name=$2
K=$3
tag=${task}_${K}_few
type=prompt-demo

export task
export tag
export K
export type

parallel -j4 --lb \
  'echo "Start $tag {1} with $(echo $K \* 2 | bc) examples"; \
  TAG=$tag \
  TYPE=$type \
  TASK=$task \
  SEED={1} \
  BS=2 \
  LR=1e-5 \
  MODEL=roberta-large \
  CUDA_VISIBLE_DEVICES=$(expr {%} - 1) \
  K=$K \
  bash run_experiment.sh "--no_train
                          --num_sample 1
                          --gpt3_in_context_head
                          --gpt3_in_context_num $(echo $K \* 2 | bc)
                          --truncate_head
                          --use_full_length"' \
  ::: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 \


# --no_train
  #                          --num_sample 1
  #                          --gpt3_in_context_head
  #                          --gpt3_in_context_num $(echo $K \* 2 | bc)
  #                          --truncate_head
  #                          --use_full_length
#  CUDA_VISIBLE_DEVICES=$(expr {%} - 1) \

python tools/gather_result.py --condition "{'tag': '$tag', 'task_name': '$task_name', 'few_shot_type': '$type'}"
