import os
import sys
import subprocess

os.chdir("..")
_, task, task_name, num_labels, k, inputs, gpu = sys.argv
num_labels = int(num_labels)
k = int(k)
inputs = eval(inputs)

tag = f"{task}_{k}_{num_labels}_fewshot_random"

input_type = "prompt-demo"


def true_parameters(num_demos):
    demo_filter_num = num_demos
    directory = f"result/{tag}_{num_demos}_{demo_filter_num}"
    return num_demos, demo_filter_num, directory


def main():
    for num_demos in inputs:
        num_demos, demo_filter_num, directory = true_parameters(num_demos)
        os.makedirs(directory, exist_ok=True)

        for array_id, seed in enumerate([13, 21, 42, 87, 100]):
            print(f"Start fewshot_similar {tag} {seed} {num_demos} on {gpu}")
            args = [
                f"--template_path auto_template/{task}/16-{seed}.sort.txt",
                "--template_id 0",
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
            print(args, flush=True)

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
                        f"{{ 'tag': '{tag}', 'task_name': '{task_name}', 'few_shot_type': '{input_type}' }}",
                        "--n_models", "1", "--save_logit_dir", directory])

    for num_demos in inputs:
        num_demos, demo_filter_rate, directory = true_parameters(num_demos)

        subprocess.run(["python", "tools/ensemble.py", "--condition",
                        f"{{ 'tag': '{tag}', 'task_name': '{task_name}', 'few_shot_type': '{input_type}' }}",
                        "--n_models", "1", "--save_logit_dir", directory])


if __name__ == "__main__":
    main()
