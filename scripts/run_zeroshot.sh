cd ..
if [ "$#" -ne 3 ]
then
    echo "The command requires 3 parameters: task name, lower-case task name, and GPU" >&2
    exit 64
fi

task=$1
dir=result/${task}_zero
task_lower=$2
gpu=$3
echo "dir: $dir, task: $task, task_lower: $task_lower"

mkdir -p $dir
tag=${task}_zero-shot

array_id=0

#for seed in 13 21 42 87 100
#do
#    TAG=$tag \
#    TYPE=prompt \
#    TASK=$task \
#    BS=2 \
#    LR=1e-5 \
#    SEED=$seed \
#    MODEL=roberta-large \
#    bash run_experiment.sh "--template_path auto_template/$task/16-$seed.sort.txt
#                            --template_id 0
#                            --demo_filter_model sbert-roberta-large
#                            --model_id 0
#                            --array_id $array_id
#                            --save_logit
#                            --save_logit_dir $dir
#                            --num_sample 1
#                            --no_train"
#
#    array_id=$(expr $array_id + 1)
#done

seed=42

TAG=$tag \
TYPE=prompt \
TASK=$task \
BS=2 \
LR=1e-5 \
SEED=$seed \
MODEL=roberta-large \
CUDA_VISIBLE_DEVICES=$gpu \
K=16 \
bash run_experiment.sh "--no_train --model_id 0 --array_id $array_id --save_logit --save_logit_dir $dir"

echo "$task zero-shot"
python tools/ensemble.py --condition "{'tag': '$tag', 'task_name': '$task_lower'}" --n_models 1 --save_logit_dir $dir
