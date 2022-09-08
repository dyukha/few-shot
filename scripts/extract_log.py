import ast
import sys

def device(type, index):
    return None

with open("../log_old", 'r') as fin:
    data = [eval(line) for line in fin]

keys = sys.argv[1:]
# print(data)
out = []
for dct in data:
    res = {}
    for key in keys:
        if key in dct:
            res[key] = dct[key]
        elif "eval_acc" in key:
            for k in dct:
                if key in k:
                    res[key] = dct[k]
                    break
    if len(res) == len(keys):
        out.append(res)

out.sort(key=lambda x: tuple(x[k] for k in x))

for i, x in enumerate(out):
    if i % 9 == 0:
        print()
    print(x)

