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

tag=${task}_gpt3-in-context

array_id=0

for seed in 13 21 42 87 100
do
    TAG=$tag \
    TYPE=prompt-demo \
    TASK=$task \
    BS=2 \
    LR=1e-5 \
    SEED=$seed \
    MODEL=roberta-large \
    CUDA_VISIBLE_DEVICES=$gpu \
    bash run_experiment.sh "--model_id 0
                            --array_id $array_id
                            --save_logit
                            --save_logit_dir $dir
                            --no_train
                            --num_sample 1
                            --gpt3_in_context_head
                            --gpt3_in_context_num 32
                            --truncate_head
                            --use_full_length"

    array_id=$(expr $array_id + 1)
done

echo "$task few-shot without filtering"
python tools/ensemble.py --condition "{'tag': '$tag', 'task_name': '$task_lower'}" --n_models 1 --save_logit_dir $dir
