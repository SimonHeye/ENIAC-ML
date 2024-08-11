from ast import parse
import os
import argparse

import ipdb
import torch
import wandb
import random
import json
import numpy as np
from transformers import T5Tokenizer, GPT2Tokenizer, GPT2Model, GPT2LMHeadModel, AutoModelForCausalLM, T5ForConditionalGeneration
from torch.utils.data import DataLoader
from zsre_dataset import Dataset, Sentence
from model import BaseT5
from encoding import *
from trainer import Trainer
from pathlib import Path
from model import ProtoT5
# from models.T5withAdapterActiveMeta import T5ForConditionalGenerationWithActiveLearning
# from models.T5withAdapter import T5ForConditionalGenerationWithActiveLearning
from models.T5withAdapter import T5ForConditionalGenerationWithAdapter

from datasets import load_dataset
import pandas as pd
import psutil
import time
# from omegaconf import DictConfig

def init_args():
    parser = argparse.ArgumentParser("Meta_Extra_Generation")
    
    parser.add_argument('--device', default='cuda:0', type=str)    
    parser.add_argument('--t5_pretrain_model_path', default='/root/autodl-tmp/Meta/PretrainModel/t5-base', type=str) #这里
    parser.add_argument('--train_extract_epochs', default=1, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=0., type=float)
    parser.add_argument('--lr', default=3e-5, type=float, help='[3e-4, 1e-4]')
    parser.add_argument('--aux_lr', default=6e-4, type=float) #6e-4
    parser.add_argument('--adam_epsilon', default=1e-8, type=float)
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--gradient_clip_val', default=1.0, type=float)
    parser.add_argument('--warmup_ratio', default=0, type=float)
    parser.add_argument('--use_scheduler', default=1, type=int)
    parser.add_argument('--aux_loss_weight', default=0.1, type=float)
    parser.add_argument('--do_eval', default=1, type=int)
    parser.add_argument('--num_beams', default=1, type=int)
    parser.add_argument('--max_len', default=256, type=int)

    parser.add_argument('--task_data_num', default=4, type=int)
    parser.add_argument('--maml_inner_loop', default=1, type=int)

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--n_unseen', default=15, type=int) #这里
    parser.add_argument('--task_num', default=1, type=int)
    parser.add_argument('--reptile_m', default=1, type=int)
    parser.add_argument('--dataset', default='fewrel', type=str, help='rebel,fewrel,wiki')#这里
    parser.add_argument('--method', default='pipeline_MAML', type=str, help='pretrain,baseline,recall_noactive,pipeline,pipeline_MAML')

    args = parser.parse_args()

    args.traindata_path = "/root/autodl-tmp/Meta/data/splits/zero_rte/fewrel/unseen_15_seed_0/train.jsonl"

    args.devdata_path = "/root/autodl-tmp/Meta/data/splits/zero_rte/fewrel/unseen_15_seed_0/dev.jsonl"

    args.testdata_path = "/root/autodl-tmp/Meta/data/splits/zero_rte/fewrel/unseen_10_seed_1/test.jsonl"

    args.relationfile_path = "/root/autodl-tmp/Meta/relations_count.tsv"

    args.output_path = f'./outputs/{args.dataset}/{time.time()}/unseen_{args.n_unseen}_seed_{args.seed}/'
    args.recall_path = f'./outputs/decoder_withrecall/{args.dataset}/noactive/{time.time()}/unseen_{args.n_unseen}_seed_{args.seed}/'
    args.pipeline_path = f'./outputs/bsz-4_ep-3_pipeline/{args.dataset}/noactive/{time.time()}/unseen_{args.n_unseen}_seed_{args.seed}/'
    args.pipeline_nometa_path = f'./outputs/bsz-1_ep-1_noreptile/{args.dataset}/noactive/{time.time()}/unseen_{args.n_unseen}_seed_{args.seed}/'
    args.pipeline_bsz4_ep1 = f'./outputs/bsz-4_ep-1_pipeline/{args.dataset}/noactive/{time.time()}/unseen_{args.n_unseen}_seed_{args.seed}/'
    args.pipeline_format_bsz4_ep1 = f'./outputs/bsz-4_ep-1_pipeline_format/{args.dataset}/noactive/{time.time()}/unseen_{args.n_unseen}_seed_{args.seed}/'
    args.pipeline_ActiveMeta_bsz1_ep1 = f'./outputs/bsz-1_ep-1_ActiveMeta/{args.dataset}/noactive/{time.time()}/unseen_{args.n_unseen}_seed_{args.seed}/'
    args.pipeline_ActiveMeta_Metric_bsz1_ep1 = f'./outputs/bsz-1_ep-1_ActiveMeta_Metric/{args.dataset}/noactive/{time.time()}/unseen_{args.n_unseen}_seed_{args.seed}/'
    args.pipeline_ActiveMeta_KL_bsz1_ep1 = f'./outputs/bsz-1_ep-1_ActiveMeta_KL/{args.dataset}/noactive/{time.time()}/unseen_{args.n_unseen}_seed_{args.seed}/'
    args.pipeline_ActiveMeta_Metricplus_bsz1_ep1 = f'./outputs/bsz-1_ep-1_ActiveMeta_Metricplus/{args.dataset}/noactive/{time.time()}/unseen_{args.n_unseen}_seed_{args.seed}/'
    return args

def safe_divide(a: float, b: float) -> float:
    if a == 0 or b == 0:
        return 0
    return a / b

def compute_score_weight(path_pred, path_gold):
    pred = Dataset.load(path_pred)
    gold = Dataset.load(path_gold)

    num_pred = 0
    num_gold = 0
    num_correct = 0
    num_trip = 0

    for i in range(len(gold.sents)):
        num_pred += len(pred.sents[i].triplets)
        num_gold += len(gold.sents[i].triplets)
        for p in pred.sents[num_trip].triplets:
            for g in gold.sents[i].triplets:
                if (p.head, p.tail, p.label) == (g.head, g.tail, g.label):
                    num_correct += 1
        num_trip += len(gold.sents[i].triplets)

    precision = safe_divide(num_correct, num_pred)
    recall = safe_divide(num_correct, num_gold)

    info = dict(
        # path_pred=path_pred,
        # path_gold=path_gold,
        precision=precision,
        recall=recall,
        score=safe_divide(2 * precision * recall, precision + recall),
    )
    return info


def pre_train_rebel(config):
    own_encoder = ExtractEncoder()
    t5_tokenizer = T5Tokenizer.from_pretrained(config.t5_pretrain_model_path)
    t5_tokenizer.add_tokens(['<triplet>', '<subj>', '<obj>'])

    t5_model = T5ForConditionalGeneration.from_pretrained(config.t5_pretrain_model_path)
    t5_model.resize_token_embeddings(len(t5_tokenizer))
    # model = ProtoT5(config, t5_tokenizer, t5_model, own_encoder)
    model = BaseT5(config, t5_tokenizer, t5_model, own_encoder)
    # ipdb.set_trace()
    print(f"load_data：{config.traindata_path}、{config.devdata_path}、{config.testdata_path}")
    train_data = load_dataset(config.Todatasets_file1, data_files = config.traindata_path, trust_remote_code=True)
    dev_data = load_dataset(config.Todatasets_file2, data_files = config.devdata_path, trust_remote_code=True)
    test_data = load_dataset(config.Todatasets_file3, data_files = config.testdata_path, trust_remote_code=True)
    print(train_data["train"])
    print(dev_data["validation"])
    print(test_data["test"])

    # ipdb.set_trace()
    Base_Trainer = Trainer(config, model, t5_tokenizer,own_encoder, train_data, dev_data, test_data)
    Base_Trainer.predata_pretrain()
    # for batch in basedata:
    #     print({k: v.shape for k, v in batch.items()})
    Base_Extractor_path = Base_Trainer.pre_train_process(aux_loss_weight=config.aux_loss_weight)
    # path_pred = str(config.output_path+ 'pred.jsonl')
    # Base_Trainer.predict(config.test_path, path_pred, model_path=Base_Extractor_path)
    precision_result, recall_result, F1_result= Base_Trainer.pretrain_predict(model_path=Base_Extractor_path ,device=config.device)
    # F1_result= Base_Trainer.predict(model_path=Base_Extractor_path ,device=config.device)
    print(str(precision_result))
    print(str(recall_result))
    print(str(F1_result))

    return config

def train_Baseline_model(config):
    own_encoder = ExtractEncoder()
    t5_tokenizer = T5Tokenizer.from_pretrained(config.t5_pretrain_model_path)
    t5_tokenizer.add_tokens(['[HEAD]', '[TAIL]', '[REL]', '[SENT]', '[Relation_discribe]'])

    t5_model = T5ForConditionalGeneration.from_pretrained(config.t5_pretrain_model_path)
    t5_model.resize_token_embeddings(len(t5_tokenizer))
    # model = ProtoT5(config, t5_tokenizer, t5_model, own_encoder)
    model = BaseT5(config, t5_tokenizer, t5_model, own_encoder)
    # ipdb.set_trace()
    print(f"load_data：{config.traindata_path}、{config.devdata_path}、{config.testdata_path}")
    train_data = Dataset.load(config.traindata_path)
    dev_data = Dataset.load(config.devdata_path)
    print(type(train_data))
    print(type(dev_data))

    # ipdb.set_trace()
    Base_Trainer = Trainer(config, model, t5_tokenizer,own_encoder, train_data, dev_data)
    Base_Trainer.predata_Baseline()
    # for batch in basedata:
    #     print({k: v.shape for k, v in batch.items()})
    Base_Extractor_path = Base_Trainer.Baseline_train_process(aux_loss_weight=config.aux_loss_weight)
    # path_pred = str(config.output_path+ 'pred.jsonl')
    # Base_Trainer.predict(config.test_path, path_pred, model_path=Base_Extractor_path)
    path_pred = str(config.baseline_path + 'pred.jsonl')
    Base_Trainer.Baseline_predict(config.testdata_path, path_pred, model_path=Base_Extractor_path)
    results = compute_score_weight(path_pred, config.testdata_path)
    print(json.dumps(results, indent=2))
    # F1_result= Base_Trainer.predict(model_path=Base_Extractor_path ,device=config.device)
    # print(str(precision_result))
    # print(str(recall_result)) 
    # print(str(F1_result))
    return config

def train_Recall_model(config):
    own_encoder = ExtractEncoder()
    t5_tokenizer = T5Tokenizer.from_pretrained(config.t5_pretrain_model_path)
    t5_tokenizer.add_tokens(['[HEAD]', '[TAIL]', '[REL]', '[SENT]', '[rel_candidate]', '<triplet>'])

    t5_model = T5ForConditionalGeneration.from_pretrained(config.t5_pretrain_model_path)
    t5_model.resize_token_embeddings(len(t5_tokenizer))
    # model = ProtoT5(config, t5_tokenizer, t5_model, own_encoder)
    model = BaseT5(config, t5_tokenizer, t5_model, own_encoder)
    # ipdb.set_trace()
    print(f"load_data：{config.traindata_path}、{config.devdata_path}、{config.testdata_path}")
    train_data = Dataset.load(config.traindata_path)
    dev_data = Dataset.load(config.devdata_path)
    print(type(train_data))
    print(type(dev_data))

    # ipdb.set_trace()
    Base_Trainer = Trainer(config, model, t5_tokenizer, own_encoder, train_data, dev_data)

    label_set = train_data.get_labels()
    Base_Trainer.predata_Recall(label_set)
    # for batch in basedata:
    #     print({k: v.shape for k, v in batch.items()})
    Base_Extractor_path = Base_Trainer.Recall_train_process(aux_loss_weight=config.aux_loss_weight)
    path_pred = str(config.recall_path + 'pred.jsonl')
    # Base_Trainer.Recall_predict(config.testdata_path, path_pred, model_path=Base_Extractor_path)
    # results = compute_score_weight(path_pred, config.testdata_path)
    # print(json.dumps(results, indent=2))

    return config    

def train_Pipeline_model(config):
    # config.batch_size = 8
    own_encoder = ExtractEncoder()
    t5_tokenizer = T5Tokenizer.from_pretrained(config.t5_pretrain_model_path)
    t5_tokenizer.add_tokens(['[HEAD]', '[TAIL]', '[REL]', '[SENT]', '[rel_candidate]', '<triplet>', '[Relation_discribe]', '<Task1>', '<Task2>'])

    t5_model = T5ForConditionalGeneration.from_pretrained(config.t5_pretrain_model_path)
    t5_model.resize_token_embeddings(len(t5_tokenizer))
    # model = ProtoT5(config, t5_tokenizer, t5_model, own_encoder)
    model = BaseT5(config, t5_tokenizer, t5_model, own_encoder)
    # ipdb.set_trace()
    print(f"load_data：{config.traindata_path}、{config.devdata_path}、{config.testdata_path}")
    train_data = Dataset.load(config.traindata_path)
    dev_data = Dataset.load(config.devdata_path)
    print(type(train_data))
    print(type(dev_data))
    # ipdb.set_trace()
    Base_Trainer = Trainer(config, model, t5_tokenizer, own_encoder, train_data, dev_data)

    label_set = train_data.get_labels()
    Base_Trainer.predata_Recall(label_set)

    Base_Extractor_path = Base_Trainer.Pipeline_train_process(aux_loss_weight=config.aux_loss_weight)
    path_pred = str(config.pipeline_path + 'pred.jsonl')
    # Base_Trainer.Recall_predict(config.testdata_path, path_pred, model_path=Base_Extractor_path)
    # results = compute_score_weight(path_pred, config.testdata_path)
    # print(json.dumps(results, indent=2))
    return config    

def train_Pipeline_ActiveMeta_model(config):
    # ipdb.set_trace()
    own_encoder = ExtractEncoder()
    t5_tokenizer = T5Tokenizer.from_pretrained(config.t5_pretrain_model_path)
    t5_tokenizer.add_tokens(['[HEAD]', '[TAIL]', '[REL]', '[SENT]', '[rel_candidate]', '<triplet>', '[Relation_discribe]', '<Task1>', '<Task2>'])

    t5_model = T5ForConditionalGeneration.from_pretrained(config.t5_pretrain_model_path)
    t5_model.resize_token_embeddings(len(t5_tokenizer))
    # model = ProtoT5(config, t5_tokenizer, t5_model, own_encoder)
    model = ProtoT5(config, t5_tokenizer, t5_model, own_encoder)
    # ipdb.set_trace()
    print(f"load_data：{config.traindata_path}、{config.devdata_path}、{config.testdata_path}")
    train_data = Dataset.load(config.traindata_path)
    dev_data = Dataset.load(config.devdata_path)
    print(type(train_data))
    print(type(dev_data))
    # ipdb.set_trace()
    Base_Trainer = Trainer(config, model, t5_tokenizer, own_encoder, train_data, dev_data)

    label_set = train_data.get_labels()
    Base_Trainer.predata_Recall_ActiveMeta(label_set)

    Base_Extractor_path = Base_Trainer.Pipeline_train_process_ActiveMeta_Metric(aux_loss_weight=config.aux_loss_weight)
    path_pred = str(config.pipeline_ActiveMeta_Metricplus_bsz1_ep1 + 'pred.jsonl')

    return config        

if __name__ == "__main__":
    opt = init_args()
    # print(opt)
    if opt.method=="pretrain":
        pre_train_rebel(opt)
    elif opt.method=="baseline":
        train_Baseline_model(opt)
    elif opt.method=="recall_noactive":
        train_Recall_model(opt)    
    elif opt.method=="pipeline":
        train_Pipeline_model(opt) 
    elif opt.method=="pipeline_MAML":
        train_Pipeline_ActiveMeta_model(opt)