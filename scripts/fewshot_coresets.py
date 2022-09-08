import os
import sys
import subprocess

os.chdir("..")
_, task, task_name, num_labels, k, merge_labels, mul, gpu = sys.argv
num_labels = int(num_labels)
k = int(k)

tag = f"{task}_{k}_{num_labels}_{merge_labels}_{mul}_fewshot_coresets"

input_type = "prompt-demo"

if merge_labels == "separate_labels":
    demo_filter_merge_labels = ""
elif merge_labels == "merge_labels":
    demo_filter_merge_labels = "--demo_filter_merge_labels"
else:
    print(f"Unknown merge_labels type {merge_labels}", file=sys.stderr)
    sys.exit(-1)


def true_parameters(num_demos, demo_filter_coresets):
    demo_filter_num = num_demos
    demo_filter_coresets_n_similar = 1
    if mul == "mul":
        num_demos = min(num_demos * num_labels, 32)
        if merge_labels == "merge_labels":
            demo_filter_coresets *= num_labels
            demo_filter_num *= num_labels
            demo_filter_coresets_n_similar *= num_labels
    elif mul == "nomul":
        pass
    else:
        print(f"Unknown mul type {mul}", file=sys.stderr)
        sys.exit(-1)
    directory = f"result/{tag}_{num_demos}_{demo_filter_coresets}_{demo_filter_num}_{demo_filter_coresets_n_similar}"
    return num_demos, demo_filter_coresets, demo_filter_num, demo_filter_coresets_n_similar, directory


def main():
    # inputs = [(1, 4), (2, 4), (4, 4)]
    inputs = [1, 2, 4, 8, 16]

    for num_demos in inputs:
        demo_filter_coresets = num_demos
        num_demos, demo_filter_coresets, demo_filter_num, demo_filter_coresets_n_similar, directory = true_parameters(num_demos, demo_filter_coresets)
        os.makedirs(directory, exist_ok=True)

        for array_id, seed in enumerate([13, 21, 42, 87, 100]):
            print(f"Start fewshot_coresets {tag} {seed} {num_demos} {demo_filter_coresets} on {gpu}")
            args = [
                f"--template_path auto_template/{task}/16-{seed}.sort.txt",
                "--template_id 0",
                "--demo_filter",
                "--demo_filter_model sbert-roberta-large",
                f"--demo_filter_num {demo_filter_num}",
                f"--demo_filter_coresets {demo_filter_coresets}",
                "--demo_filter_use_coresets",
                f"--demo_filter_coresets_n_similar {demo_filter_coresets_n_similar}",
                demo_filter_merge_labels,
                "--no_train",
                "--num_sample 1",
                "--gpt3_in_context_head",
                f"--gpt3_in_context_num {num_demos}",
                "--truncate_head",
                "--use_full_length",
                "--save_logit",
                f"--save_logit_dir {directory}",
                "--model_id 0",
                f"--array_id {array_id}",
            ]
            print(args)

            env = {
                "TAG": f"{tag}",
                "TYPE": f"{input_type}",
                "TASK": f"{task}",
                "SEED": f"{seed}",
                "BS": "1000",
                "LR": "1000",
                "MODEL": "roberta-large",
                "CUDA_VISIBLE_DEVICES": f"{gpu}",
                "K": f"{k}"
            }
            env.update(os.environ)
            subprocess.run(["bash", "run_experiment_not_save.sh", " ".join(args)], env=env)

        subprocess.run(["python", "tools/ensemble.py", "--condition",
                        f"{{ 'tag': '{tag}', 'task_name': '{task_name}', 'gpt3_in_context_num': {num_demos}, 'few_shot_type': '{input_type}', 'demo_filter_coresets': {demo_filter_coresets} }}",
                        "--n_models", "1", "--save_logit_dir", directory])

    for num_demos in inputs:
        demo_filter_coresets = num_demos
        num_demos, demo_filter_coresets, demo_filter_num, demo_filter_coresets_n_similar, directory = true_parameters(num_demos, demo_filter_coresets)

        subprocess.run(["python", "tools/ensemble.py", "--condition",
                        f"{{ 'tag': '{tag}', 'task_name': '{task_name}', 'gpt3_in_context_num': {num_demos}, 'few_shot_type': '{input_type}', 'demo_filter_coresets': {demo_filter_coresets} }}",
                        "--n_models", "1", "--save_logit_dir", directory])


if __name__ == "__main__":
    main()
