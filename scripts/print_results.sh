cd ..
python tools/gather_result.py --condition "{'tag': 'SST-2_pf', 'task_name': 'sst-2', 'few_shot_type': 'prompt-demo'}"
python tools/gather_result.py --condition "{'tag': 'SST-2_pf_filter', 'task_name': 'sst-2', 'few_shot_type': 'prompt-demo'}"