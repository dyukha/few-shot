import os
import sys
import subprocess

os.chdir("..")
_, task, task_name, num_labels, k, merge_labels, mul, mask_embeddings, inputs, gpu = sys.argv
num_labels = int(num_labels)
k = int(k)
inputs = eval(inputs)

model = "roberta-large"
tag = f"{task}_{k}_{num_labels}_{merge_labels}_{mul}_{mask_embeddings}_finetune_contrastive"

input_type = "prompt-demo"
gpt3_in_context_num = 4

if merge_labels == "separate_labels":
    demo_filter_merge_labels = ""
elif merge_labels == "merge_labels":
    demo_filter_merge_labels = "--demo_filter_merge_labels"
else:
    print(f"Unknown merge_labels type {merge_labels}", file=sys.stderr)
    sys.exit(-1)

if mask_embeddings == "mask":
    use_mask_embeddings = "--use_mask_embeddings"
elif mask_embeddings == "nomask":
    use_mask_embeddings = ""
else:
    print(f"Unknown mask_embeddings type {mask_embeddings}", file=sys.stderr)
    sys.exit(-1)

def true_parameters(num_demos):
    demo_filter_num = num_demos
    if mul == "mul":
        num_demos = min(num_demos * num_labels, 32)
        if merge_labels == "merge_labels":
            demo_filter_num *= num_labels
    elif mul == "nomul":
        pass
    else:
        print(f"Unknown mul type {mul}", file=sys.stderr)
        sys.exit(-1)
    directory = f"result/{tag}_{num_demos}_{demo_filter_num}"
    return num_demos, demo_filter_num, directory


def main():
    for num_demos in inputs:
        num_demos, demo_filter_num, directory = true_parameters(num_demos)
        os.makedirs(directory, exist_ok=True)

        for array_id, seed in enumerate([13, 21, 42, 87, 100]):
            print(f"Start finetune contrastive {tag} {seed} {num_demos} on {gpu}")
            args = [
                "--template_path", f"auto_template/{task}/16-{seed}.sort.txt",
                "--template_id", "0",
                "--demo_filter",
                "--demo_filter_model", "sbert-roberta-large",
                "--demo_filter_num", f"{demo_filter_num}",
                # f"--demo_filter_num {demo_filter_num}",
                demo_filter_merge_labels,
                use_mask_embeddings,
                "--use_contrastive",
                "--contrastive_dim", "100",
                # "--num_sample 1",
                "--gpt3_in_context_head",
                "--gpt3_in_context_num", f"{gpt3_in_context_num}",
                # "--truncate_head",
                # "--use_full_length",
                "--save_logit",
                "--save_logit_dir", f"{directory}",
                "--model_id", "0",
                "--array_id", f"{array_id}",
            ]
            print(args, flush=True)

            env = {
                "TAG": f"{tag}",
                "TYPE": f"{input_type}",
                "TASK": f"{task}",
                "SEED": f"{seed}",
                "BS": "4",
                "LR": "2e-5",
                "MODEL": model,
                "CUDA_VISIBLE_DEVICES": f"{gpu}",
                "K": f"{k}"
            }
            env.update(os.environ)
            subprocess.run(["bash", "run_experiment_not_save.sh", " ".join(args)], env=env)

        subprocess.run(["python", "tools/ensemble.py", "--condition",
                        f"{{ 'tag': '{tag}', 'task_name': '{task_name}', 'gpt3_in_context_num': {gpt3_in_context_num}, 'few_shot_type': '{input_type}' }}",
                        "--n_models", "1", "--save_logit_dir", directory])

    for num_demos in inputs:
        num_demos, demo_filter_num, directory = true_parameters(num_demos)

        subprocess.run(["python", "tools/ensemble.py", "--condition",
                        f"{{ 'tag': '{tag}', 'task_name': '{task_name}', 'gpt3_in_context_num': {gpt3_in_context_num}, 'few_shot_type': '{input_type}' }}",
                        "--n_models", "1", "--save_logit_dir", directory])


if __name__ == "__main__":
    main()
