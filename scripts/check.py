import torch
from transformers import RobertaTokenizer, RobertaModel, RobertaForMaskedLM
tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
model = RobertaForMaskedLM.from_pretrained('roberta-large')
# model.fc = torch.nn.Identity()
print(model)
newmodel = torch.nn.Sequential(*(list(model.children())[:-1]))
print(newmodel)
# print(model[-1])


# import torch
#
# cached_features_file = "../data/k-shot/SST-2/16-100/cached_dev_NoneType_128_sst-2"
#
# a, b = torch.load(cached_features_file)
# # print(a)
# # print(b)
# print(type(a), type(b))
# print(len(a), len(b))
# print(a[0])
# print(b[0])