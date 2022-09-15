"""Dataset utils for different data settings for GLUE."""

import os
import logging
import random

import torch
import numpy as np
import time
from filelock import FileLock
import json

from torch.utils.data import DataLoader

from src.processors import processors_mapping, num_labels_mapping, output_modes_mapping, compute_metrics_mapping, median_mapping
from transformers.data.processors.utils import InputFeatures
import dataclasses
from dataclasses import dataclass
from typing import List, Optional, Union
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from src.models import RobertaForPromptFinetuning, ContrastiveModel
import hashlib

from src.tokenizer import tokenize_multipart_input

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class KnnLinearFixFeatures():
    """
    Inherit from Transformers' InputFeatuers.
    """

    demo_embeddings: torch.Tensor
    demo_labels: List[int]
    embeddings: torch.Tensor
    label: Optional[Union[int, float]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


@dataclass(frozen=True)
class KnnInputFeatures():
    """
    Inherit from Transformers' InputFeatuers.
    """

    logits: List[float]
    label: Optional[Union[int, float]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


@dataclass(frozen=True)
class ContrastiveInputFeatures():
    """
    Inherit from Transformers' InputFeatures.
    """

    inputs: torch.Tensor
    positives: torch.Tensor
    negatives: torch.Tensor

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


@dataclass(frozen=True)
class OurInputFeatures(InputFeatures):
    """
    Inherit from Transformers' InputFeatuers.
    """

    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None
    mask_pos: Optional[List[int]] = None # Position of the mask token
    label_word_list: Optional[List[int]] = None # Label word mapping (dynamic)

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"

@dataclass(frozen=True)
class KnnAndRobertaInputFeatures(InputFeatures):
    """
    Inherit from Transformers' InputFeatuers.
    """

    knn_logits: List[float] = None
    input_ids: List[int] = None
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None
    mask_pos: Optional[List[int]] = None # Position of the mask token
    label_word_list: Optional[List[int]] = None # Label word mapping (dynamic)

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


def input_example_to_string(example, sep_token):
    if example.text_b is None:
        return example.text_a
    else:
        # Warning: very simple hack here
        return example.text_a + ' ' + sep_token + ' ' + example.text_b

def input_example_to_tuple(example):
    if example.text_b is None:
        if pd.isna(example.text_a) or example.text_a is None:
            return ['']
            logger.warn("Empty input")
        else:
            return [example.text_a]
    else:
        return [example.text_a, example.text_b]


class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class ContrastiveDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, indices):
        self.embeddings = embeddings
        self.indices = indices

    def __getitem__(self, index):
        inputs, positives, negatives = self.indices[index]
        return self.embeddings[inputs], self.embeddings[positives], self.embeddings[negatives]
        # return ContrastiveInputFeatures(inputs=self.embeddings[inputs], positives=self.embeddings[positives], negatives=self.embeddings[negatives])

    def __len__(self):
        return len(self.indices)


def get_cached(cached_filename, overwrite_cache, data_msg, computer):
    logger.info(f"Creating/loading {data_msg}")

    lock_path = cached_filename + ".lock"
    with FileLock(lock_path):
        if os.path.exists(cached_filename) and not overwrite_cache:
            start = time.time()
            res = torch.load(cached_filename)
            logger.info(f"Loading {data_msg} from cached file {cached_filename} [took {time.time() - start:.3f} s]")
        else:
            logger.info(f"Creating {data_msg}")
            start = time.time()
            res = computer()
            torch.save(res, cached_filename)
            logger.info(f"Saving features into cached file {cached_filename} [took {time.time() - start:.3f} s]")
    return res

class FewShotDataset(torch.utils.data.Dataset):
    """Few-shot dataset."""

    def __init__(
            self,
            args,
            tokenizer,
            cache_dir,
            mode,
            use_demo,
            use_knn,
            use_linear_fix,
            mask_embedding_training,
            label_to_word,
            label_word_list,
            use_contrastive,
            seed,
            lm_weight,
    ):
        self.use_knn = use_knn
        self.use_linear_fix = use_linear_fix
        self.args = args
        self.task_name = args.task_name
        self.processor = processors_mapping[args.task_name]
        self.tokenizer = tokenizer
        self.mode = mode
        self.mask_embedding_training = mask_embedding_training
        self.label_to_word = label_to_word
        self.label_word_list = label_word_list
        self.use_contrastive = use_contrastive
        self.seed = seed
        self.lm_weight = lm_weight

        # If not using demonstrations, use use_demo=True
        self.use_demo = use_demo
        if self.use_demo:
            logger.info("Use demonstrations")
        assert mode in ["train", "dev", "test"]

        # Get label list and (for prompt) label word list
        self.label_list = self.processor.get_labels()
        self.label_to_id = {label: i for i, label in enumerate(self.label_list)}
        self.num_labels = len(self.label_list)
        self.is_regression = self.num_labels == 1

        # Multiple sampling: when using demonstrations, we sample different combinations of demonstrations during
        # inference and aggregate the results by averaging the logits. The number of different samples is num_sample.
        if mode == "train" or not self.use_demo:
            # We do not do multiple sampling when not using demonstrations or when it's the training mode
            self.num_sample = 1
        else:
            self.num_sample = args.num_sample

        # If we use multiple templates, we also need to do multiple sampling during inference.
        if args.prompt and args.template_list is not None:
            logger.info("There are %d templates. Multiply num_sample by %d" % (len(args.template_list), len(args.template_list)))
            self.num_sample *= len(args.template_list)

        logger.info("Total num_sample for mode %s: %d" % (mode, self.num_sample))

        # Load cache
        # Cache name distinguishes mode, task name, tokenizer, and length. So if you change anything beyond these elements, make sure to clear your cache.
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            f"cached_{mode}_{tokenizer.__class__.__name__}_{args.max_seq_length}_{args.task_name}",
        )

        def compute_examples():
            support_examples = self.processor.get_train_examples(args.data_dir)

            if mode == "dev":
                query_examples = self.processor.get_dev_examples(args.data_dir)
            elif mode == "test":
                query_examples = self.processor.get_test_examples(args.data_dir)
            elif mode == "train":
                query_examples = support_examples
            else:
                raise ValueError(f"Unexpected mode {mode}")

            return support_examples, query_examples

        self.support_examples, self.query_examples = get_cached(cached_features_file, args.overwrite_cache, "examples", compute_examples)

        # For filtering in using demonstrations, load pre-calculated embeddings
        self.prepare_embeddings(args, mode)
        # Prepare examples (especially for using demonstrations)
        # print("Lengths", len(self.support_examples), len(self.query_examples))
        # print("Demo num", args.demo_filter_num)
        self.generate_ds(args, mode)

    def generate_contrastive_inputs(self, classes, n_samples, n_negatives):
        all_elements = set()
        for c in classes:
            all_elements.update(c)
        c_rest = [(c, all_elements.difference(c)) for c in classes]
        res = []
        for _ in range(n_samples):
            c, rest = random.choice(c_rest)
            query, positive = random.sample(c, k=2)
            negatives = random.sample(rest, k=n_negatives)
            res.append((query, positive, negatives))
        return res

    def get_contrastive_embeddings(self, args, mode):
        cached_filename = os.path.join(args.data_dir, f"cached_contrastive_emb_{mode}_{len(self.support_examples)}_{args.task_name}_{self.seed}_{args.contrastive_dim}_{args.use_mask_embeddings}")

        def compute_embeddings():
            n_negatives = 16
            cached_model_filename = os.path.join(args.data_dir, f"cached_contrastive_model_{len(self.support_examples)}_{args.task_name}_{self.seed}_{args.contrastive_dim}_{args.use_mask_embeddings}")
            model = ContrastiveModel(self.num_labels, self.support_emb.shape[-1], args.contrastive_dim, n_negatives).to('cuda')
            # print(model)

            def compute_state_dict():
                classes = {}
                for i, example in enumerate(self.support_examples):
                    classes.setdefault(example.label, []).append(i)
                indices = self.generate_contrastive_inputs(list(classes.values()), 1000, n_negatives)
                ds = ContrastiveDataset(self.support_emb, indices)
                data_loader = DataLoader(ds, batch_size=16)
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.001)
                # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0001)
                with torch.enable_grad():
                    for epoch in range(10):
                        for batch in data_loader: ##type: List[ContrastiveInputFeatures]
                            # print([x.shape for x in batch])
                            # inputs = torch.stack([x[0] for x in batch]),
                            # positives = torch.stack([x[1] for x in batch]),
                            # negatives = torch.stack([x[2] for x in batch])
                            # inputs = torch.stack([x.inputs for x in batch]),
                            # positives = torch.stack([x.positives for x in batch]),
                            # negatives = torch.stack([x.negatives for x in batch])
                            inputs, positives, negatives = batch
                            optimizer.zero_grad()
                            out = model(inputs, positives, negatives)
                            loss = model.get_loss(out)
                            # if epoch % 10 == 0:
                            # print(loss.item(), flush=True)
                            loss.backward()
                            optimizer.step()
                return model.state_dict()

            # state_dict = get_cached(cached_model_filename, True, "constrastive model", compute_state_dict)
            state_dict = get_cached(cached_model_filename, args.overwrite_cache, "constrastive model", compute_state_dict)
            model.load_state_dict(state_dict)
            return model.dense(self.support_emb), model.dense(self.query_emb)
        # support_emb, query_emb = get_cached(cached_filename, True, "contrastive embeddings", compute_embeddings)
        support_emb, query_emb = get_cached(cached_filename, args.overwrite_cache, "contrastive embeddings", compute_embeddings)
        return support_emb, query_emb

    def prepare_embeddings(self, args, mode):
        if self.use_demo and args.demo_filter:
            split_name = self.get_split_name(mode, args.task_name)

            def read_embeddings(filename) -> torch.Tensor:
                embeddings = np.load(filename)
                embeddings = torch.tensor(embeddings, device='cuda')
                return torch.nn.functional.normalize(embeddings)

            self.support_emb = read_embeddings(os.path.join(args.data_dir, f"train_{args.demo_filter_model}.npy"))
            query_filename = os.path.join(args.data_dir, f'{split_name}_{args.demo_filter_model}.npy')
            self.query_emb = read_embeddings(query_filename)
            logger.info(f"Load embeddings (for demonstration filtering) from {query_filename}")

            assert len(self.support_emb) == len(self.support_examples)
            assert len(self.query_emb) == len(self.query_examples)

        if self.mask_embedding_training is not None:
            self.support_mask_emb = self.get_mask_embeddings(self.support_examples, args.prompt, args.single_sentence_template, "train", args)
            self.query_mask_emb = self.get_mask_embeddings(self.query_examples, args.prompt, args.single_sentence_template, mode, args)
            if args.use_mask_embeddings:
                self.support_emb = self.support_mask_emb
                self.query_emb = self.query_mask_emb

        if self.use_demo and args.demo_filter and self.use_contrastive:
            self.support_emb, self.query_emb = self.get_contrastive_embeddings(args, mode)

            assert len(self.support_emb) == len(self.support_examples)
            assert len(self.query_emb) == len(self.query_examples)

        if self.use_demo and args.demo_filter:
            self.query_support_sim = self.query_emb @ self.support_emb.T
            self.support_support_sim = self.support_emb @ self.support_emb.T


    def generate_ds(self, args, mode):
        # Size is expanded by num_sample
        self.size = len(self.query_examples) * self.num_sample

        support_indices = list(range(len(self.support_examples)))
        self.example_idx = []

        # start_time = time.time()
        # logger.info("Start coresets")
        # logger.info(f"{self.support_emb.shape} {self.query_emb.shape}")
        # self.support_emb = torch.tensor(self.support_emb, device='cuda')
        # self.query_emb = torch.tensor(self.query_emb, device='cuda')

        for sample_idx in range(self.num_sample):
            for query_idx in range(len(self.query_examples)):
                # If training, exclude the current example. Else keep all.
                if mode != "train":
                    context_indices = list(support_indices)
                else:
                    context_indices = [support_idx for support_idx in support_indices if support_idx != query_idx]

                if self.use_demo and args.demo_filter:
                    if args.demo_filter_use_coresets:
                        context_indices = self.get_coreset_indices_per_class(
                            context_indices,
                            query_idx,
                            n_clusters=args.demo_filter_coresets,
                            filter_merge_labels=args.demo_filter_merge_labels,
                            coresets_n_similar=args.demo_filter_coresets_n_similar,
                        )
                    context_indices = self.filter_similar_models(args, context_indices, query_idx,
                                                                 filter_merge_labels=args.demo_filter_merge_labels)
                    context_indices = context_indices * args.demo_repeat
                    assert len(context_indices) > 0

                # We'll subsample context_indices further later.
                self.example_idx.append((query_idx, context_indices, sample_idx))

        # logger.info(f"End coresets. Elapsed: {time.time() - start_time}")
        # If it is not training, we pre-process the data; otherwise, we process the data online.
        self.label_map = {label: i for i, label in enumerate(self.label_list)}
            # print("Shapes", support_mask_emb.shape, query_mask_emb.shape)
        if mode != "train":
            self.features = []
            for sample_idx, (query_idx, context_indices, bootstrap_idx) in enumerate(self.example_idx):
                self.features.append(self.gen_features(sample_idx, query_idx, context_indices, self.label_map, args))
        else:
            self.features = None

    def gen_features(self, sample_idx, query_idx, context_indices, label_map, args):
        # The input (query) example
        example = self.query_examples[query_idx]
        # The demonstrations
        supports = self.select_context([self.support_examples[i] for i in context_indices])

        if self.use_linear_fix:
            demo_labels = [self.get_label(self.support_examples[i]) for i in context_indices]
            return KnnLinearFixFeatures(
                demo_embeddings=self.support_mask_emb[context_indices],
                demo_labels=demo_labels,
                embeddings=self.query_mask_emb[query_idx],
                label=self.get_label(self.query_examples[query_idx]),
            )
        if self.use_knn:
            logits = [0.0] * len(label_map)
            for support in supports:
                logits[self.label_to_id[support.label]] += 1
            return KnnInputFeatures(logits=logits, label=self.get_label(example))

        if self.lm_weight < 1 - 1e-9:
            logits = [0.0] * len(label_map)
            for support in supports:
                logits[self.label_to_id[support.label]] += 1
            # logits = torch.Tensor(logits)
            # logits = torch. logits
            knn_features = KnnInputFeatures(logits=logits, label=self.get_label(example))
            # self.features.append(OurInputFeatures(input_ids=input_ids, label=example_label))
        if self.lm_weight > 1e-9:
            lm_features = self.call_convert(args, example, supports, sample_idx)

        if self.lm_weight < 1e-9:
            return knn_features
        elif self.lm_weight > 1 - 1e-9:
            return lm_features
        else:
            return \
                KnnAndRobertaInputFeatures(
                    knn_logits=knn_features.logits,
                    input_ids=lm_features.input_ids,
                    attention_mask=lm_features.attention_mask,
                    label=lm_features.label,
                    mask_pos=lm_features.mask_pos,
                    label_word_list=lm_features.label_word_list,
                )

    def call_convert(self, args, example, supports, sample_idx):
        if not args.do_zero_shot:
            return self.convert_fn(
                example=example,
                supports=supports,
                use_demo=self.use_demo,
                label_list=self.label_list,
                prompt=args.prompt,
                template=self.get_template(args, sample_idx),
                label_word_list=self.label_word_list,
                verbose=False,
                # verbose=True if sample_idx == 0 else False,
            )
        else:
            return self.convert_fn(
                example=example,
                supports=[],
                use_demo=self.use_demo,
                label_list=self.label_list,
                prompt=args.prompt,
                template=args.single_sentence_template,
                label_word_list=self.label_word_list,
                verbose=False,
            )

    def get_template(self, args, sample_idx):
        if args.template_list is not None:
            return args.template_list[sample_idx % len(args.template_list)]  # Use template in order
        else:
            return args.template

    def get_label(self, example):
        label_map = {label: i for i, label in enumerate(self.label_list)}
        if example.label is None:
            return None
        elif self.is_regression:
            return float(example.label)
        else:
            return label_map[example.label]

    def get_mask_embeddings(self, examples, prompt, template, mode, args):
        model = self.mask_embedding_training.model
        # print(model.__dict__)
        template_hash = hashlib.sha224(template.encode()).hexdigest()
        cached_features_file = os.path.join(args.data_dir, f"cached_mask_emb_{mode}_{len(examples)}_{args.task_name}_{template_hash}_{model.name_or_path}_{model.drop_dense}")

        def compute_mask_embeddings():
            features = [self.convert_fn(
                example=example,
                supports=[],
                use_demo=self.use_demo,
                label_list=self.label_list,
                prompt=prompt,
                template=template,
                label_word_list=self.label_word_list,
                verbose=False,
            ) for example in examples]

            features = [
                OurInputFeatures(
                    input_ids=f.input_ids,
                    attention_mask=f.attention_mask,
                    token_type_ids=f.token_type_ids,
                    label=None,
                    mask_pos=f.mask_pos,
                    label_word_list=f.label_word_list,
                )
                for f in features]

            ds = EmbeddingDataset(features)
            outputs = self.mask_embedding_training.evaluate(eval_dataset=ds)
            return outputs.predictions

        return torch.tensor(get_cached(cached_features_file, args.overwrite_cache, "mask embeddings", compute_mask_embeddings), device='cuda')

    def get_split_name(self, mode, task_name) -> str:
        if mode == 'train':
            return 'train'
        elif mode == 'dev':
            if task_name == 'mnli':
                return 'dev_matched'
            elif task_name == 'mnli-mm':
                return 'dev_mismatched'
            else:
                return 'dev'
        elif mode == 'test':
            if task_name == 'mnli':
                return 'test_matched'
            elif task_name == 'mnli-mm':
                return 'test_mismatched'
            else:
                return 'test'
        else:
            raise NotImplementedError

    def get_coreset_indices(self, candidates, query_idx, n_clusters, coresets_n_similar):
        candidates.sort()
        sim_score = self.query_support_sim[query_idx][candidates]

        top_inds = sim_score.argsort(descending=True).tolist()[:coresets_n_similar]

        n_clusters = min(n_clusters, len(candidates))
        sims = torch.zeros(len(candidates), device='cuda')
        all_sims = []

        if candidates == list(range(len(self.support_emb))):
            data = self.support_emb
            support_support_sims = self.support_support_sim
        else:
            data = self.support_emb[candidates]
            support_support_sims = self.support_support_sim[candidates][:, candidates]

        for ind in range(n_clusters):
            min_ind = top_inds[ind] if ind < coresets_n_similar else sims.argmin().item()
            cur_sims = support_support_sims[min_ind]
            all_sims.append(cur_sims)
            sims = torch.max(sims, cur_sims)

        all_sims = torch.stack(all_sims)
        closest_cluster = all_sims.argmax(dim=0)

        clusters = [[] for _ in range(n_clusters)]
        for i, c in enumerate(closest_cluster.tolist()):
            clusters[c].append(i)
        fixed_centers = torch.stack([data[cluster].mean(dim=0) for cluster in clusters])
        # fixed_centers = fixed_centers[coresets_n_similar:]
        fixed_centers = torch.nn.functional.normalize(fixed_centers)
        fixed_sims = fixed_centers @ data.T

        true_centers = fixed_sims.argmax(dim=1)
        # true_centers[:coresets_n_similar] = top_inds
        return [candidates[x] for x in top_inds + true_centers[coresets_n_similar:].tolist()]

    def get_coreset_indices_per_class(self, candidates_all, query_idx, n_clusters, filter_merge_labels, coresets_n_similar):
        if filter_merge_labels:
            return self.get_coreset_indices(candidates_all, query_idx, n_clusters, coresets_n_similar)

        candidates_per_class = {}
        for c in candidates_all:
            candidates_per_class.setdefault(self.support_examples[c].label, []).append(c)
        res = []
        for candidates in candidates_per_class.values():
            res += self.get_coreset_indices(candidates, query_idx, n_clusters, coresets_n_similar)
        return res

    def filter_similar_models(self, args, candidates, query_idx, filter_merge_labels):
        sim_score = self.query_support_sim[query_idx][candidates]

        sorted_indices = sim_score.argsort(descending=True).cpu()
        sorted_candidates = [candidates[ind] for ind in sorted_indices]

        limit = args.demo_filter_num
        if filter_merge_labels:
            return sorted_candidates[:limit]

        if self.is_regression:
            count_each_label = {'0': 0, '1': 0}
        else:
            count_each_label = {label: 0 for label in self.label_list}

        context_indices = []

        if args.debug_mode:
            print(f"Query {self.query_examples[query_idx].label}: {self.query_examples[query_idx].text_a}")  # debug
        for support_idx, score in zip(sorted_indices, sim_score):
            if self.is_regression:
                label = '0' if float(self.support_examples[support_idx].label) <= median_mapping[args.task_name] else '1'
            else:
                label = self.support_examples[support_idx].label
            if count_each_label[label] < limit:
                count_each_label[label] += 1
                context_indices.append(support_idx)
                if args.debug_mode:
                    print(f"    {score:.4f} {self.support_examples[support_idx].label} | {self.support_examples[support_idx].text_a}")  # debug
        return context_indices

    def select_context(self, context_examples):
        """
        Select demonstrations from provided examples.
        """
        max_demo_per_label = 1
        counts = {k: 0 for k in self.label_list}
        if len(self.label_list) == 1:
            # Regression
            counts = {'0': 0, '1': 0}
        selection = []

        if self.args.gpt3_in_context_head or self.args.gpt3_in_context_tail:
            # For GPT-3's in-context learning, we sample gpt3_in_context_num demonstrations randomly.
            order = np.random.permutation(len(context_examples))
            selection = [context_examples[x] for x in order[:self.args.gpt3_in_context_num]]
        else:
            # Our sampling strategy
            order = np.random.permutation(len(context_examples))

            for i in order:
                label = context_examples[i].label
                if len(self.label_list) == 1:
                    # Regression
                    label = '0' if float(label) <= median_mapping[self.args.task_name] else '1'
                if counts[label] < max_demo_per_label:
                    selection.append(context_examples[i])
                    counts[label] += 1
                if sum(counts.values()) == len(counts) * max_demo_per_label:
                    break

            assert len(selection) > 0

        return selection

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        if self.features is not None:
            return self.features[i]

        query_idx, context_indices, bootstrap_idx = self.example_idx[i]
        return self.gen_features(i, query_idx, context_indices, self.label_map, self.args)

    def get_labels(self):
        return self.label_list


    def convert_fn(
        self,
        example,
        supports,
        use_demo=False,
        label_list=None,
        prompt=False,
        template=None,
        label_word_list=None,
        verbose=False
    ):
        # print("supports", supports)
        # print("template", template)
        """
        Returns a list of processed "InputFeatures".
        """
        max_length = self.args.max_seq_length

        # Prepare labels
        label_map = {label: i for i, label in enumerate(label_list)} # Mapping the label names to label ids
        if len(label_list) == 1:
            # Regression
            label_map = {'0': 0, '1': 1}

        # Get example's label id (for training/inference)
        example_label = self.get_label(example)

        # Prepare other features
        if not use_demo:
            # No using demonstrations
            inputs = tokenize_multipart_input(
                input_text_list=input_example_to_tuple(example),
                max_length=max_length,
                tokenizer=self.tokenizer,
                task_name=self.args.task_name,
                prompt=prompt,
                template=template,
                label_word_list=label_word_list,
                first_sent_limit=self.args.first_sent_limit,
                other_sent_limit=self.args.other_sent_limit,
            )
            features = OurInputFeatures(**inputs, label=example_label)

        else:
            # Using demonstrations

            # Max length
            if self.args.double_demo:
                # When using demonstrations, double the maximum length
                # Note that in this case, args.max_seq_length is the maximum length for a single sentence
                max_length = max_length * 2
            if self.args.gpt3_in_context_head or self.args.gpt3_in_context_tail:
                # When using GPT-3's in-context learning, take the maximum tokenization length of the model (512)
                max_length = 512

            # All input sentences, including the query and the demonstrations, are put into augmented_examples, 
            # and are numbered based on the order (starting from 0). For single sentence tasks, the input (query)
            # is the sentence 0; for sentence-pair tasks, the input (query) is the sentence 0 and 1. Note that for GPT-3's 
            # in-context learning, the input (query) might be at the end instead of the beginning (gpt3_in_context_head)
            augmented_example = []
            query_text = input_example_to_tuple(example) # Input sentence list for query
            support_by_label = [[] for i in range(len(label_map))]

            # TODO rewrite this
            if self.args.gpt3_in_context_head or self.args.gpt3_in_context_tail:
                support_labels = []
                augmented_example = query_text
                for support_example in supports:
                    augmented_example += input_example_to_tuple(support_example)
                    current_label = support_example.label
                    if len(label_list) == 1:
                        current_label = '0' if float(current_label) <= median_mapping[self.args.task_name] else '1' # Regression
                    support_labels.append(label_map[current_label])
            else:
                # Group support examples by label
                for label_name, label_id in label_map.items():
                    for support_example in supports:
                        if self.is_regression:
                            label = '0' if float(support_example.label) <= median_mapping[self.args.task_name] else '1'
                        else:
                            label = support_example.label
                        if label == label_name:
                            support_by_label[label_id] += input_example_to_tuple(support_example)

                augmented_example = query_text
                for label_id in range(len(label_map)):
                    augmented_example += support_by_label[label_id]

            # Tokenization (based on the template)
            inputs = tokenize_multipart_input(
                input_text_list=augmented_example,
                max_length=max_length,
                tokenizer=self.tokenizer,
                task_name=self.args.task_name,
                prompt=prompt,
                template=template,
                label_word_list=label_word_list,
                first_sent_limit=self.args.first_sent_limit,
                other_sent_limit=self.args.other_sent_limit,
                truncate_head=self.args.truncate_head,
                gpt3=self.args.gpt3_in_context_head or self.args.gpt3_in_context_tail,
                support_labels=None if not (self.args.gpt3_in_context_head or self.args.gpt3_in_context_tail) else support_labels
            )
            features = OurInputFeatures(**inputs, label=example_label)

        if verbose:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("features: %s" % features)
            logger.info("text: %s" % self.tokenizer.decode(features.input_ids))

        return features



