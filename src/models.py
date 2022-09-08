"""Custom models for few-shot learning specific operations."""
import random

import torch
import torch.nn as nn
import transformers
from transformers.modeling_bert import BertPreTrainedModel, BertForSequenceClassification, BertModel, BertOnlyMLMHead
from transformers.modeling_roberta import RobertaForSequenceClassification, RobertaModel, RobertaLMHead, RobertaClassificationHead
from transformers.modeling_outputs import SequenceClassifierOutput

import logging
logger = logging.getLogger(__name__)

def resize_token_type_embeddings(model, new_num_types: int, random_segment: bool):
    """
    Resize the segment (token type) embeddings for BERT
    """
    if hasattr(model, 'bert'):
        old_token_type_embeddings = model.bert.embeddings.token_type_embeddings
    else:
        raise NotImplementedError
    new_token_type_embeddings = nn.Embedding(new_num_types, old_token_type_embeddings.weight.size(1))
    if not random_segment:
        new_token_type_embeddings.weight.data[:old_token_type_embeddings.weight.size(0)] = old_token_type_embeddings.weight.data

    model.config.type_vocab_size = new_num_types
    if hasattr(model, 'bert'):
        model.bert.embeddings.token_type_embeddings = new_token_type_embeddings
    else:
        raise NotImplementedError


class BertForPromptFinetuning(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        # print("config:", config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None

        # For regression
        self.lb = None
        self.ub = None

        # For label search.
        self.return_full_softmax = None
        self.total_times = []

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        mask_pos=None,
        labels=None,
    ):
        batch_size = input_ids.size(0)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        # Encode everything
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # Get <mask> token representation
        sequence_output, pooled_output = outputs[:2]
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]

        # Logits over vocabulary tokens
        prediction_mask_scores = self.cls(sequence_mask_output)

        # Exit early and only return mask logits.
        if self.return_full_softmax:
            if labels is not None:
                return torch.zeros(1, out=prediction_mask_scores.new()), prediction_mask_scores
            return prediction_mask_scores

        # Return logits for each label
        logits = []
        for label_id in range(len(self.label_word_list)):
            logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
        logits = torch.cat(logits, -1)

        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits) # Log prob of right polarity

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb), (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
                loss = loss_fct(logits.view(-1, 2), labels)
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)
        return ((loss,) + output) if loss is not None else output



class RobertaForPromptFinetuning(BertPreTrainedModel):

    def __init__(self, config, drop_dense=False):
        super().__init__(config)
        # print("config:", config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)
        if not drop_dense:
            self.lm_head = RobertaLMHead(config)
        self.init_weights()
        self.drop_dense = drop_dense

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None

        # For regression
        self.lb = None
        self.ub = None

        # For auto label search.
        self.return_full_softmax = None
        self.total_times = [0.0] * 8
        self.last_time = None

    def measure_time(self, ind):
        # return
        import time
        torch.cuda.synchronize()
        cur_time = time.time()
        if ind != -1:
            self.total_times[ind] += cur_time - self.last_time
        self.last_time = cur_time

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mask_pos=None,
        labels=None,
    ):
        # if random.random() < 0.01:
        # print(self.total_times, flush=True)
        assert mask_pos is not None

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        self.measure_time(-1)
        # Encode everything
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask
        )
        self.measure_time(0)

        # Get <mask> token representation
        sequence_output, pooled_output = outputs[:2]
        self.measure_time(1)
        # sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0), device='cuda'), mask_pos]
        # sequence_mask_output = sequence_output[:, mask_pos]
        # print("outputs.shape", sequence_output.shape)
        # print("mask_pos", mask_pos)
        # print("sequence_mask_output", sequence_mask_output.shape, flush=True)

        self.measure_time(2)

        # Logits over vocabulary tokens
        if self.drop_dense:
            return sequence_mask_output

        prediction_mask_scores = self.lm_head(sequence_mask_output)

        # print("aaa", flush=True)
        # Exit early and only return mask logits.
        if self.return_full_softmax:
            if labels is not None:
                return torch.zeros(1, out=prediction_mask_scores.new()), prediction_mask_scores
            return prediction_mask_scores
        # print("bbb", flush=True)

        self.measure_time(3)

        assert self.label_word_list is not None
        logits = prediction_mask_scores[:, self.label_word_list]
        # print("prediction_mask_scores", prediction_mask_scores.shape, flush=True)
        # print("logits", logits.shape, flush=True)
        # print("logits shape", logits.shape)
        # print("prediction_mask_scores shape", prediction_mask_scores.shape)
        # print(self.label_word_list)


        self.measure_time(4)

        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits) # Log prob of right polarity

        self.measure_time(5)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb), (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
                loss = loss_fct(logits.view(-1, 2), labels)
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        self.measure_time(6)

        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)
        self.measure_time(7)
        # print("ccc", flush=True)
        # print("output shape", len(output), output[0].shape)
        return ((loss,) + output) if loss is not None else output


class KnnModel(nn.Module):
    def __init__(self, config):
        # super().__init__(config)
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels
        self.total_times = []

    def forward(
        self,
        logits=None,
        labels=None,
    ):
        logits = torch.tensor(logits)
        logits = torch.nn.functional.normalize(logits, p=1, dim=-1)

        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits) # Log prob of right polarity

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb), (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
                loss = loss_fct(logits.view(-1, 2), labels)
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)
        return ((loss,) + output) if loss is not None else output


class KnnAndRobetaModel(nn.Module):
    __initialized = False

    def __init__(self, config, lm_weight):
        super().__init__()
        self.knn_model = KnnModel(config).to('cuda')
        self.roberta_model = RobertaForPromptFinetuning.from_pretrained(
            "roberta-large",
            from_tf=False,
            config=config,
            cache_dir=None,
        ).to('cuda')
        self.lm_weight = lm_weight
        self.__initialized = True

    def __setattr__(self, key, value):
        if self.__initialized:
            self.knn_model.__setattr__(key, value)
            self.roberta_model.__setattr__(key, value)
        object.__setattr__(self, key, value)

    def forward(
        self,
        knn_logits=None,
        input_ids=None,
        attention_mask=None,
        mask_pos=None,
        labels=None
    ):
        lm_loss, lm_logits = self.roberta_model(input_ids=input_ids, attention_mask=attention_mask, mask_pos=mask_pos, labels=labels)
        knn_loss, knn_logits = self.knn_model(knn_logits, labels)
        lm_logits = torch.nn.functional.softmax(lm_logits, dim=-1)
        combined_logits = lm_logits * self.lm_weight + knn_logits * (1 - self.lm_weight)
        return knn_loss + lm_loss, combined_logits

class KnnLinearFixModel(nn.Module):
    def __init__(self, config, regularization, loss_name):
        # super().__init__(config)
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels
        self.regularization = regularization
        self.total_times = []
        self.loss_name = loss_name
        if loss_name == "mse":
            self.criterion = torch.nn.MSELoss()
        elif loss_name == "cross_entropy":
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown loss {loss_name}")

    def forward(
        self,
        demo_embeddings,
        demo_labels,
        embeddings,
        labels,
    ):
        # print("demo_embeddings shape", demo_embeddings.shape)
        # print("demo_labels shape", demo_labels.shape)
        # print("embeddings shape", embeddings.shape)
        # print("labels shape", labels.shape)
        assert len(demo_embeddings.shape) == 3
        assert len(demo_labels.shape) == 2
        assert len(embeddings.shape) == 2
        if labels is not None:
            assert len(labels.shape) == 1
            labels = labels.to('cuda')
        demo_embeddings = demo_embeddings.to('cuda')
        demo_labels = demo_labels.to('cuda')
        embeddings = embeddings.to('cuda')
        logits = []
        for i in range(len(embeddings)):
            cur_demo_embeddings = demo_embeddings[i]
            if self.loss_name == "mse":
                cur_labels = torch.nn.functional.one_hot(demo_labels[i], num_classes=self.num_labels).float()
            elif self.loss_name == "cross_entropy":
                cur_labels = demo_labels[i]
            else:
                raise ValueError(f"Unknown loss {self.loss_name}")
            # print("demo_labels[i] shape", demo_labels[i].shape)
            # print("one_hot_labels shape", one_hot_labels.shape)
            regression_model = torch.nn.Linear(cur_demo_embeddings.shape[-1], self.num_labels).to('cuda')
            # if self.loss_name == "cross_entropy":
            #     regression_model = torch.nn.Sequential(regression_model, torch.nn.Softmax())
            regression_model.train()
            optimizer = torch.optim.SGD(regression_model.parameters(), lr=0.0001, weight_decay=self.regularization)
            # optimizer = torch.optim.AdamW(regressionModel.parameters(), lr=0.1, weight_decay=self.regularization)
            with torch.enable_grad():
                for epoch in range(20):
                    optimizer.zero_grad()
                    out = regression_model(cur_demo_embeddings)
                    loss = self.criterion(out, cur_labels)
                    # if epoch % 10 == 0:
                    # print(loss.item(), flush=True)
                    loss.backward()
                    optimizer.step()

            cur_logits = regression_model(embeddings[i].unsqueeze(0))[0]
            logits.append(cur_logits)

        logits = torch.stack(logits)

        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits) # Log prob of right polarity

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb), (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
                loss = loss_fct(logits.view(-1, 2), labels)
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)
        return ((loss,) + output) if loss is not None else output


class ContrastiveModel(nn.Module):
    def __init__(self, num_labels, n_input_features, out_dimension, n_negatives):
        # super().__init__(config)
        super().__init__()
        self.num_labels = num_labels
        self.dense = nn.Linear(n_input_features, out_dimension)
        self.n_negatives = n_negatives
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = 0.1

    def normalize(self, vectors):
        return nn.functional.normalize(vectors, dim=-1)

    def get_loss(self, predictions):
        return self.criterion(predictions, torch.zeros(predictions.shape[0], dtype=torch.long, device='cuda'))

    def forward(
        self,
        inputs,
        positives,
        negatives
    ):
        # print("demo_embeddings shape", demo_embeddings.shape)
        # print("demo_labels shape", demo_labels.shape)
        # print("embeddings shape", embeddings.shape)
        # print("labels shape", labels.shape)
        assert len(inputs.shape) == 2
        assert len(positives.shape) == 2
        assert len(negatives.shape) == 3
        assert negatives.shape[1] == self.n_negatives, f"{negatives.shape}[1] != {self.n_negatives}"

        inputs = inputs.to('cuda')
        positives = positives.to('cuda')
        negatives = negatives.to('cuda')
        candidates_input = torch.cat([positives.unsqueeze(dim=1), negatives], dim=1)
        candidates_output = self.normalize(self.dense(candidates_input))
        query_outputs = self.normalize(self.dense(inputs)).unsqueeze(-1)
        similarities = torch.bmm(candidates_output, query_outputs).squeeze(-1)
        return similarities / self.temperature


# class KnnModel(nn.Module):
#     def __init__(self, config):
#         # super().__init__(config)
#         super().__init__()
#         self.config = config
#         self.num_labels = config.num_labels
#         self.total_times = []
#
#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         mask_pos=None,
#         labels=None,
#     ):
#         logits = torch.tensor(input_ids)
#
#         # Regression task
#         if self.config.num_labels == 1:
#             logsoftmax = nn.LogSoftmax(-1)
#             logits = logsoftmax(logits) # Log prob of right polarity
#
#         loss = None
#         if labels is not None:
#             if self.num_labels == 1:
#                 # Regression task
#                 loss_fct = nn.KLDivLoss(log_target=True)
#                 labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb), (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
#                 loss = loss_fct(logits.view(-1, 2), labels)
#             else:
#                 loss_fct = nn.CrossEntropyLoss()
#                 loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
#
#         output = (logits,)
#         if self.num_labels == 1:
#             # Regression output
#             output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)
#         return ((loss,) + output) if loss is not None else output


# class KnnModel(BertPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels
#         self.total_times = []
#
#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         mask_pos=None,
#         labels=None,
#     ):
#         logits = torch.tensor(input_ids)
#
#         # Regression task
#         if self.config.num_labels == 1:
#             logsoftmax = nn.LogSoftmax(-1)
#             logits = logsoftmax(logits) # Log prob of right polarity
#
#         loss = None
#         if labels is not None:
#             if self.num_labels == 1:
#                 # Regression task
#                 loss_fct = nn.KLDivLoss(log_target=True)
#                 labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb), (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
#                 loss = loss_fct(logits.view(-1, 2), labels)
#             else:
#                 loss_fct = nn.CrossEntropyLoss()
#                 loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
#
#         output = (logits,)
#         if self.num_labels == 1:
#             # Regression output
#             output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)
#         return ((loss,) + output) if loss is not None else output

