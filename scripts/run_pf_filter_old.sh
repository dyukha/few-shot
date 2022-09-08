cd ..
if [ "$#" -ne 3 ]
then
    echo "The command requires 3 parameters: task name, lower-case task name, and GPU" >&2
    exit 64
fi

task=$1
dir=result/${task}_pf
dir_filter=${dir}_filter
task_lower=$2
gpu=$3
echo "dir: $dir, task: $task, task_lower: $task_lower"

tag=${task}_pf
tag_filter=${tag}_filter

mkdir -p $dir_filter

array_id=0
for seed in 13 21 42 87 100
do
    for bs in 2 4 8
    do
        for lr in 1e-5 2e-5 5e-5
        do
            echo "filter $task: run #$array_id"

            TAG=$tag_filter \
            TYPE=prompt-demo \
            TASK=$task \
            BS=$bs \
            LR=$lr \
            SEED=$seed \
            MODEL=roberta-large \
            CUDA_VISIBLE_DEVICES=$gpu \
            bash run_experiment.sh "--demo_filter
                                    --demo_filter_model sbert-roberta-large
                                    --model_id 0
                                    --array_id $array_id
                                    --save_logit
                                    --save_logit_dir $dir_filter"

            array_id=$(expr $array_id + 1)
        done
    done
done

echo "$task prompt-based finetuning with filtering"
python tools/gather_result.py --condition "{'tag': '$tag_filter', 'task_name': '$task_lower'}"
