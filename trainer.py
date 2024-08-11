import collections
import copy
from curses import raw
from enum import EnumMeta
from importlib.resources import path
import os
import pdb
from signal import raise_signal
from tkinter import EXCEPTION
from xmlrpc.client import FastMarshaller

import torch
import wandb
import random
from transformers import AdamW, get_linear_schedule_with_warmup, pipeline, Pipeline, set_seed
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from zsre_dataset import Dataset, Sentence
from transformers import T5Tokenizer, GPT2Tokenizer, GPT2Model, GPT2LMHeadModel, AutoModelForCausalLM, T5ForConditionalGeneration
from models.T5withAdapter import T5ForConditionalGenerationWithAdapter
# from models.T5withAdapterActiveMeta import T5ForConditionalGenerationWithActiveLearning
from encoding import ExtractEncoder, ExtractEncoder_plus
# from encoding import select_encoder
from zsre_dataset import RelationSentence
from torch.nn import functional as F
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from torch import Tensor
from typing import Dict, List, Optional, Tuple
from utils import collate_fn_pretrain, collate_fn_Baseline, find_sublist_index, collate_fn_Recall, collate_fn_Recall_help, collate_fn_Recall_plus, collate_fn_Recall_help_plus, collate_fn_Pipeline1, collate_fn_Pipeline2, collate_fn_Pipeline1_help, collate_fn_Pipeline_traindata, collate_fn_Pipeline_devdata, collate_fn_Pipeline_meta_traindata, collate_fn_Pipeline_ActiveMeta_traindata, collate_fn_Pipeline_ActiveMeta_Metric_traindata
from transformers import PreTrainedModel, PreTrainedTokenizerFast
from pydantic import BaseModel
import shutil
import json
import ipdb
from datasets import load_dataset
import evaluate
from model import BaseT5
import re


class Trainer():
    def __init__(self, opt, model, tokenizer, encoder, train_data, dev_data, d_model=768):
        self.opt = opt
        self.model = model
        self.tokenizer = tokenizer
        self.encoder = encoder
        if torch.cuda.is_available():
            self.model.cuda()
        self.train_dataset, self.dev_dataset = train_data, dev_data
        self.train_dataset_copy = None
        self.dev_dataset_copy = None
        self.train_labels = train_data.get_labels()
        self.dev_labels = dev_data.get_labels()
                
        with open("/root/autodl-tmp/Meta/data/REL_discribe.jsonl","r") as f1:
            contents = f1.read()
            json_object = json.loads(contents)
            self.relation_discribe = json_object

    def step(self, batch):
        outputs = self.model(**batch)
        return outputs

    def Meta_step(self, batch, model):
        outputs = model(**batch)
        return outputs
    # def step2(self, batch):
    #     outputs = self.model(input_ids=batch['step2_input_ids'],)
    #     return outputs

    def pipeforward(self, batch, model_state_dict):
        # raw_model = copy.deepcopy(self.model.t5)
        # raw_model = self.model.t5.load_state_dict(updated_model_state_dict)

        # own_encoder = ExtractEncoder()
        # t5_tokenizer = T5Tokenizer.from_pretrained('/user_data/wujy/SimonHeye/META/PretrainModel/t5-base')#这里
        # t5_tokenizer.add_tokens(['[HEAD]', '[TAIL]', '[REL]', '[SENT]', '[rel_candidate]', '<triplet>', '[Relation_discribe]', '<Task1>', '<Task2>'])
        # raw_model = T5ForConditionalGeneration.from_pretrained('/user_data/wujy/SimonHeye/META/PretrainModel/t5-base')
        # raw_model.to('cuda')
        # raw_model.resize_token_embeddings(len(t5_tokenizer))
        print("1:{}".format(torch.cuda.memory_allocated(0)))
        self.model.t5.load_state_dict(model_state_dict)
        print("2:{}".format(torch.cuda.memory_allocated(0)))
        # raw_model = BaseT5(config, t5_tokenizer, raw_model, own_encoder)
        # Meta_optimizer, Meta_scheduler = self.get_optimizer_meta(raw_model)

        train_step = 0
        batch = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}
        #self.tokenizer.decode(queryset_batch1['input_ids'][0].tolist())
        supportset_batch1= {"input_ids": batch["support_task1_input_ids"], "attention_mask": batch["support_task1_attention_mask"], "decoder_attention_mask": batch["support_task1_decoder_attention_mask"], "labels": batch["support_task1_labels"]}
        supportset_batch2= {"input_ids": batch["support_task2_input_ids"], "attention_mask": batch["support_task2_attention_mask"], "decoder_attention_mask": batch["support_task2_decoder_attention_mask"], "labels": batch["support_task2_labels"]}
        
        queryset_batch1= {"input_ids": batch["query_task1_input_ids"], "attention_mask": batch["query_task1_attention_mask"], "decoder_attention_mask": batch["query_task1_decoder_attention_mask"], "labels": batch["query_task1_labels"]}
        queryset_batch2= {"input_ids": batch["query_task2_input_ids"], "attention_mask": batch["query_task2_attention_mask"], "decoder_attention_mask": batch["query_task2_decoder_attention_mask"], "labels": batch["query_task2_labels"]}
        # ipdb.set_trace()
        outer_loss=0
        # inner_model=None
        # for _ in range(self.opt.maml_inner_loop):
        # model = copy.deepcopy(raw_model)
        # ipdb.set_trace()
        self.model.train()
        # print("3:{}".format(torch.cuda.memory_allocated(0)))
        inner_loss=0
        train_step +=1
        outputs1 = self.step(supportset_batch1)
        inner_loss += outputs1.loss
        outputs2 = self.step(supportset_batch2)
        inner_loss += outputs2.loss
        # print("4:{}".format(torch.cuda.memory_allocated(0)))
        inner_loss = inner_loss / self.opt.gradient_accumulation_steps
        inner_loss.backward()
        # print("5:{}".format(torch.cuda.memory_allocated(0)))
        if train_step % self.opt.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.opt.gradient_clip_val)
            # print("6:{}".format(torch.cuda.memory_allocated(0)))
            self.optimizer.step()
            # print("7:{}".format(torch.cuda.memory_allocated(0)))
            self.optimizer.zero_grad()
            # print("8:{}".format(torch.cuda.memory_allocated(0)))
            if self.scheduler:
                self.scheduler.step()            
        # updated_model_state_dict = raw_model.state_dict()
        
        # raw_model.load_state_dict(updated_model_state_dict)
        outputs3 = self.step(queryset_batch1)
        outer_loss += outputs3.loss
        outputs4 = self.step(queryset_batch2)
        outer_loss += outputs4.loss
        self.optimizer.zero_grad()
        # torch.cuda.empty_cache()
        return outer_loss      

    def predata_pretrain(self):
        self.opt.epochs=self.opt.train_extract_epochs
        self.optimizer, self.scheduler = self.get_optimizer()
        self.train_data = self.Train_data2tensor(self.train_dataset)
        self.dev_data = self.Dev_data2tensor(self.dev_dataset)
        self.test_data = self.Test_data2tensor(self.test_dataset)

        # random.seed(self.opt.seed)
        # random.shuffle(self.train_data)
        # # random.shuffle(self.dev_data)

        self.train_dataloader = DataLoader(self.train_data,
                                      batch_size=self.opt.batch_size,
                                      num_workers=4,
                                      drop_last=False,
                                      shuffle=True,
                                      collate_fn=collate_fn_pretrain)
        self.dev_dataloader = DataLoader(self.dev_data,
                                    batch_size=self.opt.batch_size,
                                    num_workers=4,
                                    drop_last=False,
                                    shuffle=True,
                                    collate_fn=collate_fn_pretrain)
        # ipdb.set_trace()
        self.test_dataloader = DataLoader(self.test_data,
                                    batch_size=self.opt.batch_size,
                                    num_workers=4,
                                    drop_last=False,
                                    shuffle=True,
                                    collate_fn=collate_fn_pretrain)
        
    def predata_Baseline(self, pid2name=''):
            
        self.opt.epochs=self.opt.train_extract_epochs
        self.optimizer, self.scheduler = self.get_optimizer()
        self.train_data = self.process_Baseline_data(self.train_dataset,self.train_labels,self.relation_discribe)
        self.dev_data = self.process_Baseline_data(self.dev_dataset,self.dev_labels,self.relation_discribe)

        random.seed(self.opt.seed)
        random.shuffle(self.train_data)
        random.shuffle(self.dev_data)

        # ipdb.set_trace()
        self.train_dataloader = DataLoader(self.train_data,
                                      batch_size=self.opt.batch_size,
                                      num_workers=4,
                                      drop_last=False,
                                      shuffle=True,
                                      collate_fn=collate_fn_Baseline)
        self.dev_dataloader = DataLoader(self.dev_data,
                                    batch_size=self.opt.batch_size,
                                    num_workers=4,
                                    drop_last=False,
                                    shuffle=True,
                                    collate_fn=collate_fn_Baseline)

    def predata_Recall(self, label_set, pid2name=''):
            
        self.opt.epochs=self.opt.train_extract_epochs
        self.optimizer, self.scheduler = self.get_optimizer()
        self.train_data = self.process_Pipeline_data(self.train_dataset,label_set,self.relation_discribe)#这里
        self.dev_data = self.process_Pipeline_data(self.dev_dataset,label_set,self.relation_discribe)

        random.seed(self.opt.seed)
        random.shuffle(self.train_data)
        random.shuffle(self.dev_data)


        # ipdb.set_trace()

        self.train_dataloader = DataLoader(self.train_data,
                                      batch_size=self.opt.batch_size,
                                      num_workers=4,
                                      drop_last=False,
                                      shuffle=True,
                                      collate_fn=collate_fn_Pipeline_traindata)
        # ipdb.set_trace()
        self.dev_dataloader = DataLoader(self.dev_data,
                                    batch_size=self.opt.batch_size,
                                    num_workers=4,
                                    drop_last=False,
                                    shuffle=True,
                                    collate_fn=collate_fn_Pipeline_devdata)

    def predata_Recall_ActiveMeta(self, label_set, pid2name=''):
            
        self.opt.epochs=self.opt.train_extract_epochs
        self.optimizer, self.scheduler = self.get_optimizer()
        self.train_data = self.process_Pipeline_ActiveMeta_Metric_data(self.train_dataset,label_set,self.relation_discribe)#这里
        self.dev_data = self.process_Pipeline_ActiveMeta_Metric_data(self.dev_dataset,label_set,self.relation_discribe)

        random.seed(self.opt.seed)
        random.shuffle(self.train_data)
        random.shuffle(self.dev_data)


        # ipdb.set_trace()

        self.train_dataloader = DataLoader(self.train_data,
                                      batch_size=self.opt.batch_size,
                                      num_workers=4,
                                      drop_last=False,
                                      shuffle=True,
                                      collate_fn=collate_fn_Pipeline_ActiveMeta_Metric_traindata)
        # ipdb.set_trace()
        self.dev_dataloader = DataLoader(self.dev_data,
                                    batch_size=self.opt.batch_size,
                                    num_workers=4,
                                    drop_last=False,
                                    shuffle=True,
                                    collate_fn=collate_fn_Pipeline_devdata)       

    def Train_data2tensor(self, dataset):
        data=[]
        for i in range(len(dataset["train"])):
            fin_data = {}
            temp_data = {}
            temp_data['input_ids'] = []
            temp_data['attention_mask'] = []
            temp_data['labels'] = []
            temp_data['decoder_attention_mask'] = []

            temp_context = self.tokenizer(dataset["train"][i]['context'], return_tensors='pt', add_special_tokens=True)
            temp_data['input_ids'].append(temp_context['input_ids'])
            temp_data['attention_mask'].append(temp_context['attention_mask'])

            temp_triplet = self.tokenizer(dataset["train"][i]['triplets'], return_tensors='pt', add_special_tokens=True)
            temp_data['labels'].append(temp_triplet['input_ids'])
            temp_data['decoder_attention_mask'].append(temp_triplet['attention_mask'])

            data.append(temp_data)
        return data

    def Dev_data2tensor(self, dataset):
        data=[]
        for i in range(len(dataset["validation"])):
            fin_data = {}
            temp_data = {}
            temp_data['input_ids'] = []
            temp_data['attention_mask'] = []
            temp_data['labels'] = []
            temp_data['decoder_attention_mask'] = []

            temp_context = self.tokenizer(dataset["validation"][i]['context'], return_tensors='pt', add_special_tokens=True)
            temp_data['input_ids'].append(temp_context['input_ids'])
            temp_data['attention_mask'].append(temp_context['attention_mask'])

            temp_triplet = self.tokenizer(dataset["validation"][i]['triplets'], return_tensors='pt', add_special_tokens=True)
            temp_data['labels'].append(temp_triplet['input_ids'])
            temp_data['decoder_attention_mask'].append(temp_triplet['attention_mask'])

            data.append(temp_data)
        return data

    def Test_data2tensor(self, dataset):
        data=[]
        for i in range(len(dataset["test"])):
            fin_data = {}
            temp_data = {}
            temp_data['input_ids'] = []
            temp_data['attention_mask'] = []
            temp_data['labels'] = []
            temp_data['decoder_attention_mask'] = []

            temp_context = self.tokenizer(dataset["test"][i]['context'], return_tensors='pt', add_special_tokens=True)
            temp_data['input_ids'].append(temp_context['input_ids'])
            temp_data['attention_mask'].append(temp_context['attention_mask'])

            temp_triplet = self.tokenizer(dataset["test"][i]['triplets'], return_tensors='pt', add_special_tokens=True)
            temp_data['labels'].append(temp_triplet['input_ids'])
            temp_data['decoder_attention_mask'].append(temp_triplet['attention_mask'])

            data.append(temp_data)
        return data

    def process_Baseline_data(self, dataset, labels_set, relation_discribe):
        
        data = []
        for sent in dataset.sents:
            sents = [self.encoder.encode_to_line(trip) for trip in sent.triplets]# sent : Sentence(triplets=[RelationSentence(tokens=['In', '1689'...], head=[11, 12], tail=[16], label='after a work by', head_id='', tail_id='', label_id='P140', error='', raw='', keyword=[''], score=0.0, zerorc_included=True)])
            #sents = ['{"text": "[SENT] : In 1689 , Konstanty was one of the judges who sentenced Kazimierz \\u0141yszczy\\u0144ski to death for atheism .", "summary": "[HEAD] Kazimierz \\u0141yszczy\\u0144ski , [TAIL] atheism , [REL] after a work by ."}\n']
            trips = [trip for trip in sent.triplets]
            #trips = [RelationSentence(tokens=['In', '1689', ',', 'Konstanty', 'was', 'one', 'of', 'the', 'judges', 'who', 'sentenced', 'Kazimierz', 'Łyszczyński', 'to', 'death', 'for', 'atheism', '.'], head=[11, 12], tail=[16], label='after a work by', head_id='', tail_id='', label_id='P140', error='', raw='', keyword=[''], score=0.0, zerorc_included=True)]
            for sent, trip in zip(sents, trips):
                temp_data = {}
                temp_train = {}
                
                context, labels = self.encoder.parse_line(sent)# content:'[SENT] : Merpati flight 106 departed Jakarta ( CGK ) on a domestic flight to Tanjung Pandan ( TJQ ) .'   labels:'[HEAD] Kazimierz Łyszczyński , [TAIL] atheism , [REL] after a work by .'
                # ipdb.set_trace()
                temp_train['input_ids'], temp_train['attention_mask'] = [], []
                temp_train['labels'], temp_train['decoder_attention_mask'] = [], []

                temp1 = self.tokenizer(context, return_tensors='pt', add_special_tokens=True)
                temp_train['input_ids'], temp_train['attention_mask'] = temp1['input_ids'], temp1['attention_mask']

                temp2 = self.tokenizer(labels, return_tensors='pt', add_special_tokens=True)
                temp_train['decoder_input_ids'], temp_train['decoder_attention_mask'] = temp2['input_ids'], temp2['attention_mask']
                
                dis_id = trip.label_id
                discribe = [relation_discribe[dis_id][1]] #这里
                fin_discribe = '[relation_discribe]' + ': '.join(discribe)+'. '

                # ipdb.set_trace()
                temp3 = self.tokenizer(fin_discribe, return_tensors='pt', add_special_tokens=True)
                dis_input_ids, dis_attn = temp3['input_ids'], temp3['attention_mask']
                
                temp_data['sent'] = temp_train
                temp_data['dis_input_ids'] = dis_input_ids
                temp_data['dis_attn'] = dis_attn

                data.append(temp_data)
        return data        

    def process_Recall_traindata(self, dataset, labels_set, relation_discribe):
        # ipdb.set_trace()
        # if len(labels_set)<self.opt.n_unseen:
        #     labels_set = labels_set*4 
            # ipdb.set_trace()       
        data = []
        for sent in dataset.sents:
            sents = [self.encoder.encode_to_line(trip) for trip in sent.triplets]# sent : Sentence(triplets=[RelationSentence(tokens=['In', '1689'...], head=[11, 12], tail=[16], label='after a work by', head_id='', tail_id='', label_id='P140', error='', raw='', keyword=[''], score=0.0, zerorc_included=True)])
            #sents = ['{"text": "[SENT] : In 1689 , Konstanty was one of the judges who sentenced Kazimierz \\u0141yszczy\\u0144ski to death for atheism .", "summary": "[HEAD] Kazimierz \\u0141yszczy\\u0144ski , [TAIL] atheism , [REL] after a work by ."}\n']
            trips = [trip for trip in sent.triplets]
            #trips = [RelationSentence(tokens=['In', '1689', ',', 'Konstanty', 'was', 'one', 'of', 'the', 'judges', 'who', 'sentenced', 'Kazimierz', 'Łyszczyński', 'to', 'death', 'for', 'atheism', '.'], head=[11, 12], tail=[16], label='after a work by', head_id='', tail_id='', label_id='P140', error='', raw='', keyword=[''], score=0.0, zerorc_included=True)]
            for sent, trip in zip(sents, trips):
                temp_data = {}
                temp_train = {}
                temp_prompt = []

                context, labels = self.encoder.parse_line(sent)# content:'[SENT] : Merpati flight 106 departed Jakarta ( CGK ) on a domestic flight to Tanjung Pandan ( TJQ ) .'   labels:'[HEAD] Kazimierz Łyszczyński , [TAIL] atheism , [REL] after a work by .'
                # ipdb.set_trace()
                temp_train['input_ids'], temp_train['attention_mask'] = [], []
                temp_train['labels'], temp_train['decoder_attention_mask'] = [], []

                temp1 = self.tokenizer(context, return_tensors='pt', add_special_tokens=True)
                temp_train['input_ids'], temp_train['attention_mask'] = temp1['input_ids'], temp1['attention_mask']

                # temp2 = self.tokenizer(labels, return_tensors='pt', add_special_tokens=True)
                # temp_train['decoder_input_ids'], temp_train['decoder_attention_mask'] = temp2['input_ids'], temp2['attention_mask']
                
                dis_id = trip.label_id
                dis_name = [relation_discribe[dis_id][0]]
                discribe = [relation_discribe[dis_id][2]] #这里
                # fin_discribe = dis_name + ': '.join(discribe)+'. '+'<triplet>: '
                fin_discribe = f'{dis_name[0]}:{discribe[0]}. <triplet>:'

                # ipdb.set_trace()
                # temp3 = self.tokenizer(fin_discribe, return_tensors='pt', add_special_tokens=True)
                # dis_input_ids, dis_attn = temp3['input_ids'], temp3['attention_mask']
                
                fin_decode_inputs = fin_discribe + labels
                temp2 = self.tokenizer(fin_decode_inputs, return_tensors='pt', add_special_tokens=True)
                dis_input_ids, dis_attn = temp2['input_ids'], temp2['attention_mask']

                temp_data['sent'] = temp_train
                temp_data['dis_input_ids'] = dis_input_ids
                temp_data['dis_attn'] = dis_attn
                # ipdb.set_trace()
                # self.tokenizer.decode(temp_data['sent']['input_ids'][0].tolist())
                # if len(labels_set)<=self.opt.n_unseen-1:
                #     ipdb.set_trace()
                candidate_rel = random.sample(labels_set,self.opt.n_unseen-1) # ['participant in', 'position held', 'constellation', 'member of']
                # rel_idx = [rel_idx_dict[rel] for rel in candidate_rel]
                rand_int = random.randint(0, self.opt.n_unseen - 1) # 4
                # ipdb.set_trace()
                temp_prompt.extend(candidate_rel[:rand_int]) #['participant in', 'position held', 'constellation', 'member of']
                temp_prompt.append(dis_name[0]) #['participant in', 'position held', 'constellation', 'member of', 'after a work by']
                temp_prompt.extend(candidate_rel[rand_int:])
                prompt =  '[rel_candidate]: '+', '.join(temp_prompt)+ '. '
                temp3 = self.tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
                prompt_ids, prompt_attn = temp3['input_ids'], temp3['attention_mask']
                temp_data['prompt_ids'], temp_data['prompt_attn'] = prompt_ids, prompt_attn
                data.append(temp_data)
        return data        

    def process_Recall_devdata(self, dataset, labels_set, relation_discribe):
        # ipdb.set_trace()
        # if len(labels_set)<self.opt.n_unseen:
        #     labels_set = labels_set*4 
            # ipdb.set_trace()       
        data = []
        for sent in dataset.sents:
            sents = [self.encoder.encode_to_line(trip) for trip in sent.triplets]# sent : Sentence(triplets=[RelationSentence(tokens=['In', '1689'...], head=[11, 12], tail=[16], label='after a work by', head_id='', tail_id='', label_id='P140', error='', raw='', keyword=[''], score=0.0, zerorc_included=True)])
            #sents = ['{"text": "[SENT] : In 1689 , Konstanty was one of the judges who sentenced Kazimierz \\u0141yszczy\\u0144ski to death for atheism .", "summary": "[HEAD] Kazimierz \\u0141yszczy\\u0144ski , [TAIL] atheism , [REL] after a work by ."}\n']
            trips = [trip for trip in sent.triplets]
            #trips = [RelationSentence(tokens=['In', '1689', ',', 'Konstanty', 'was', 'one', 'of', 'the', 'judges', 'who', 'sentenced', 'Kazimierz', 'Łyszczyński', 'to', 'death', 'for', 'atheism', '.'], head=[11, 12], tail=[16], label='after a work by', head_id='', tail_id='', label_id='P140', error='', raw='', keyword=[''], score=0.0, zerorc_included=True)]
            for sent, trip in zip(sents, trips):
                temp_data = {}
                temp_train = {}
                temp_prompt = []

                context, labels = self.encoder.parse_line(sent)# content:'[SENT] : Merpati flight 106 departed Jakarta ( CGK ) on a domestic flight to Tanjung Pandan ( TJQ ) .'   labels:'[HEAD] Kazimierz Łyszczyński , [TAIL] atheism , [REL] after a work by .'
                # ipdb.set_trace()
                temp_train['input_ids'], temp_train['attention_mask'] = [], []
                temp_train['labels'], temp_train['decoder_attention_mask'] = [], []

                temp1 = self.tokenizer(context, return_tensors='pt', add_special_tokens=True)
                temp_train['input_ids'], temp_train['attention_mask'] = temp1['input_ids'], temp1['attention_mask']

                # temp2 = self.tokenizer(labels, return_tensors='pt', add_special_tokens=True)
                # temp_train['decoder_input_ids'], temp_train['decoder_attention_mask'] = temp2['input_ids'], temp2['attention_mask']
                
                dis_id = trip.label_id
                dis_name = [relation_discribe[dis_id][0]]
                discribe = [relation_discribe[dis_id][2]] #这里
                # fin_discribe = dis_name + ': '.join(discribe)+'. '+'<triplet>: '
                fin_discribe = f'{dis_name[0]}:{discribe[0]}. <triplet>:'

                # ipdb.set_trace()
                # temp3 = self.tokenizer(fin_discribe, return_tensors='pt', add_special_tokens=True)
                # dis_input_ids, dis_attn = temp3['input_ids'], temp3['attention_mask']
                
                fin_decode_inputs = fin_discribe + labels
                temp2 = self.tokenizer(fin_decode_inputs, return_tensors='pt', add_special_tokens=True)
                dis_input_ids, dis_attn = temp2['input_ids'], temp2['attention_mask']

                temp_data['sent'] = temp_train
                temp_data['dis_input_ids'] = dis_input_ids
                temp_data['dis_attn'] = dis_attn
                # ipdb.set_trace()
                # self.tokenizer.decode(temp_data['sent']['input_ids'][0].tolist())
                # if len(labels_set)<=self.opt.n_unseen-1:
                #     ipdb.set_trace()
                candidate_rel = random.sample(labels_set,self.opt.n_unseen) # ['participant in', 'position held', 'constellation', 'member of']
                # ipdb.set_trace()
                # rel_idx = [rel_idx_dict[rel] for rel in candidate_rel]
                # rand_int = random.randint(0, self.opt.n_unseen - 1) # 4
                # ipdb.set_trace()
                # temp_prompt.extend(candidate_rel[:rand_int]) #['participant in', 'position held', 'constellation', 'member of']
                # temp_prompt.append(dis_name[0]) #['participant in', 'position held', 'constellation', 'member of', 'after a work by']
                # temp_prompt.extend(candidate_rel[rand_int:])
                prompt =  '[rel_candidate]: '+', '.join(candidate_rel)+ '. '
                temp3 = self.tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
                prompt_ids, prompt_attn = temp3['input_ids'], temp3['attention_mask']
                temp_data['prompt_ids'], temp_data['prompt_attn'] = prompt_ids, prompt_attn
                data.append(temp_data)
        return data        
    
    def process_Pipeline1_traindata(self, dataset, labels_set, relation_discribe):
        # ipdb.set_trace()
        # if len(labels_set)<self.opt.n_unseen:
        #     labels_set = labels_set*4 
            # ipdb.set_trace()       
        data = []
        for sent in dataset.sents:
            sents = [self.encoder.encode_to_line(trip) for trip in sent.triplets]# sent : Sentence(triplets=[RelationSentence(tokens=['In', '1689'...], head=[11, 12], tail=[16], label='after a work by', head_id='', tail_id='', label_id='P140', error='', raw='', keyword=[''], score=0.0, zerorc_included=True)])
            #sents = ['{"text": "[SENT] : In 1689 , Konstanty was one of the judges who sentenced Kazimierz \\u0141yszczy\\u0144ski to death for atheism .", "summary": "[HEAD] Kazimierz \\u0141yszczy\\u0144ski , [TAIL] atheism , [REL] after a work by ."}\n']
            trips = [trip for trip in sent.triplets]
            #trips = [RelationSentence(tokens=['In', '1689', ',', 'Konstanty', 'was', 'one', 'of', 'the', 'judges', 'who', 'sentenced', 'Kazimierz', 'Łyszczyński', 'to', 'death', 'for', 'atheism', '.'], head=[11, 12], tail=[16], label='after a work by', head_id='', tail_id='', label_id='P140', error='', raw='', keyword=[''], score=0.0, zerorc_included=True)]
            for sent, trip in zip(sents, trips):
                temp_data = {}
                temp_train = {}
                temp_prompt = []

                context, labels = self.encoder.parse_line(sent)# content:'[SENT] : Merpati flight 106 departed Jakarta ( CGK ) on a domestic flight to Tanjung Pandan ( TJQ ) .'   labels:'[HEAD] Kazimierz Łyszczyński , [TAIL] atheism , [REL] after a work by .'
                # ipdb.set_trace()
                temp_train['input_ids'], temp_train['attention_mask'] = [], []
                temp_train['labels'], temp_train['decoder_attention_mask'] = [], []

                temp1 = self.tokenizer(context, return_tensors='pt', add_special_tokens=True)
                temp_train['input_ids'], temp_train['attention_mask'] = temp1['input_ids'], temp1['attention_mask']

                # temp2 = self.tokenizer(labels, return_tensors='pt', add_special_tokens=True)
                # temp_train['decoder_input_ids'], temp_train['decoder_attention_mask'] = temp2['input_ids'], temp2['attention_mask']
                
                dis_id = trip.label_id
                dis_name = [relation_discribe[dis_id][0]]
                discribe = [relation_discribe[dis_id][2]] #这里
                # fin_discribe = dis_name + ': '.join(discribe)+'. '+'<triplet>: '
                fin_discribe = f'{dis_name[0]}:{discribe[0]}.'

                # ipdb.set_trace()
                # temp3 = self.tokenizer(fin_discribe, return_tensors='pt', add_special_tokens=True)
                # dis_input_ids, dis_attn = temp3['input_ids'], temp3['attention_mask']

                temp2 = self.tokenizer(fin_discribe, return_tensors='pt', add_special_tokens=True)
                dis_input_ids, dis_attn = temp2['input_ids'], temp2['attention_mask']

                temp_data['sent'] = temp_train
                temp_data['dis_input_ids'] = dis_input_ids
                temp_data['dis_attn'] = dis_attn
                # ipdb.set_trace()
                # self.tokenizer.decode(temp_data['sent']['input_ids'][0].tolist())
                # if len(labels_set)<=self.opt.n_unseen-1:
                #     ipdb.set_trace()
                candidate_rel = random.sample(labels_set,self.opt.n_unseen-1) # ['participant in', 'position held', 'constellation', 'member of']
                # rel_idx = [rel_idx_dict[rel] for rel in candidate_rel]
                rand_int = random.randint(0, self.opt.n_unseen - 1) # 4
                # ipdb.set_trace()
                temp_prompt.extend(candidate_rel[:rand_int]) #['participant in', 'position held', 'constellation', 'member of']
                temp_prompt.append(dis_name[0]) #['participant in', 'position held', 'constellation', 'member of', 'after a work by']
                temp_prompt.extend(candidate_rel[rand_int:])
                prompt =  '[rel_candidate]: '+', '.join(temp_prompt)+ '. '
                temp3 = self.tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
                prompt_ids, prompt_attn = temp3['input_ids'], temp3['attention_mask']
                temp_data['prompt_ids'], temp_data['prompt_attn'] = prompt_ids, prompt_attn
                
                temp_label_name = f'[REL]: {trip.label}'
                temp4 = self.tokenizer(temp_label_name, return_tensors='pt', add_special_tokens=True)
                label_name, label_name_attn = temp4['input_ids'], temp4['attention_mask']
                temp_data['label_name'], temp_data['label_name_attn'] = label_name, label_name_attn
                data.append(temp_data)
        return data        

    def process_Pipeline2_traindata(self, dataset, labels_set, relation_discribe):
        # ipdb.set_trace()
        # if len(labels_set)<self.opt.n_unseen:
        #     labels_set = labels_set*4 
            # ipdb.set_trace()       
        data = []
        for sent in dataset.sents:
            sents = [self.encoder.encode_to_line(trip) for trip in sent.triplets]# sent : Sentence(triplets=[RelationSentence(tokens=['In', '1689'...], head=[11, 12], tail=[16], label='after a work by', head_id='', tail_id='', label_id='P140', error='', raw='', keyword=[''], score=0.0, zerorc_included=True)])
            #sents = ['{"text": "[SENT] : In 1689 , Konstanty was one of the judges who sentenced Kazimierz \\u0141yszczy\\u0144ski to death for atheism .", "summary": "[HEAD] Kazimierz \\u0141yszczy\\u0144ski , [TAIL] atheism , [REL] after a work by ."}\n']
            trips = [trip for trip in sent.triplets]
            #trips = [RelationSentence(tokens=['In', '1689', ',', 'Konstanty', 'was', 'one', 'of', 'the', 'judges', 'who', 'sentenced', 'Kazimierz', 'Łyszczyński', 'to', 'death', 'for', 'atheism', '.'], head=[11, 12], tail=[16], label='after a work by', head_id='', tail_id='', label_id='P140', error='', raw='', keyword=[''], score=0.0, zerorc_included=True)]
            for sent, trip in zip(sents, trips):
                temp_data = {}
                temp_train = {}
                temp_prompt = []

                context, labels = self.encoder.parse_line(sent)# content:'[SENT] : Merpati flight 106 departed Jakarta ( CGK ) on a domestic flight to Tanjung Pandan ( TJQ ) .'   labels:'[HEAD] Kazimierz Łyszczyński , [TAIL] atheism , [REL] after a work by .'
                # ipdb.set_trace()
                temp_train['input_ids'], temp_train['attention_mask'] = [], []
                temp_train['labels'], temp_train['decoder_attention_mask'] = [], []

                temp1 = self.tokenizer(context, return_tensors='pt', add_special_tokens=True)
                temp_train['input_ids'], temp_train['attention_mask'] = temp1['input_ids'], temp1['attention_mask']

                temp2 = self.tokenizer(labels, return_tensors='pt', add_special_tokens=True)
                temp_train['decoder_input_ids'], temp_train['decoder_attention_mask'] = temp2['input_ids'], temp2['attention_mask']
                
                dis_id = trip.label_id
                dis_name = [relation_discribe[dis_id][0]]
                discribe = [relation_discribe[dis_id][2]] #这里
                # fin_discribe = dis_name + ': '.join(discribe)+'. '+'<triplet>: '
                fin_discribe = f'{dis_name[0]}:{discribe[0]}.'

                # ipdb.set_trace()
                # temp3 = self.tokenizer(fin_discribe, return_tensors='pt', add_special_tokens=True)
                # dis_input_ids, dis_attn = temp3['input_ids'], temp3['attention_mask']

                temp3 = self.tokenizer(fin_discribe, return_tensors='pt', add_special_tokens=False)
                dis_input_ids, dis_attn = temp3['input_ids'], temp3['attention_mask']

                temp_data['sent'] = temp_train
                temp_data['fin_discribe'] = dis_input_ids
                temp_data['fin_discribe_attn'] = dis_attn
                # ipdb.set_trace()
                # self.tokenizer.decode(temp_data['sent']['input_ids'][0].tolist())
                # if len(labels_set)<=self.opt.n_unseen-1:
                #     ipdb.set_trace()
                data.append(temp_data)
        return data   

    def process_Pipeline_data(self, dataset, labels_set, relation_discribe):
        # ipdb.set_trace()  
        rel_dict = {rel:[] for rel in labels_set}
        data = []
        for sent in dataset.sents:
            sents = [self.encoder.encode_to_line(trip) for trip in sent.triplets]# sent : Sentence(triplets=[RelationSentence(tokens=['In', '1689'...], head=[11, 12], tail=[16], label='after a work by', head_id='', tail_id='', label_id='P140', error='', raw='', keyword=[''], score=0.0, zerorc_included=True)])
            #sents = ['{"text": "[SENT] : In 1689 , Konstanty was one of the judges who sentenced Kazimierz \\u0141yszczy\\u0144ski to death for atheism .", "summary": "[HEAD] Kazimierz \\u0141yszczy\\u0144ski , [TAIL] atheism , [REL] after a work by ."}\n']
            trips = [trip for trip in sent.triplets]
            #trips = [RelationSentence(tokens=['In', '1689', ',', 'Konstanty', 'was', 'one', 'of', 'the', 'judges', 'who', 'sentenced', 'Kazimierz', 'Łyszczyński', 'to', 'death', 'for', 'atheism', '.'], head=[11, 12], tail=[16], label='after a work by', head_id='', tail_id='', label_id='P140', error='', raw='', keyword=[''], score=0.0, zerorc_included=True)]
            for sent, trip in zip(sents, trips):
                temp_data = {}
                temp_train = {}
                temp_prompt = []

                context, labels = self.encoder.parse_line(sent)# content:'[SENT] : Merpati flight 106 departed Jakarta ( CGK ) on a domestic flight to Tanjung Pandan ( TJQ ) .'   labels:'[HEAD] Kazimierz Łyszczyński , [TAIL] atheism , [REL] after a work by .'
                # ipdb.set_trace()
                temp_train['input_ids'], temp_train['attention_mask'] = [], []
                temp_train['labels'], temp_train['decoder_attention_mask'] = [], []
                # context1 = context + ' Choose [REL] from [rel_candidate]:'
                # context2 = context + ' Extract <triplet> from [sent]:'
                dis_id = trip.label_id
                dis_name = [relation_discribe[dis_id][0]]
                discribe = [relation_discribe[dis_id][2]] #这里
                candidate_rel = random.sample(labels_set,self.opt.n_unseen-1) # ['participant in', 'position held', 'constellation', 'member of']
                rand_int = random.randint(0, self.opt.n_unseen - 1) # 4
                temp_prompt.extend(candidate_rel[:rand_int]) #['participant in', 'position held', 'constellation', 'member of']
                temp_prompt.append(dis_name[0]) #['participant in', 'position held', 'constellation', 'member of', 'after a work by']
                temp_prompt.extend(candidate_rel[rand_int:])
                prompt =  '<Task1>, Choose [REL] in [sent] from [rel_candidate]. [rel_candidate]: '+', '.join(temp_prompt)+ '. '+ context
                temp1 = self.tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
                prompt_ids, prompt_attn = temp1['input_ids'], temp1['attention_mask']
                temp_data['task1_input_ids'], temp_data['task1_input_attn'] = prompt_ids, prompt_attn
                
                temp_label_name = f'[REL]: {trip.label}'
                temp2 = self.tokenizer(temp_label_name, return_tensors='pt', add_special_tokens=True)
                label_name, label_name_attn = temp2['input_ids'], temp2['attention_mask']
                temp_data['task1_decoder_ids'], temp_data['task1_decoder_attn'] = label_name, label_name_attn


                fin_discribe = f'<Task2>, Extract <triplet> in [sent] by [Relation_discribe]. [Relation_discribe]: {dis_name[0]}, {discribe[0]}. {context}'
                temp3 = self.tokenizer(fin_discribe, return_tensors='pt', add_special_tokens=True)
                temp_data['task2_input_ids'], temp_data['task2_input_attn'] = temp3['input_ids'], temp3['attention_mask']


                token_label = '<triplet>: '+labels
                temp4 = self.tokenizer(token_label, return_tensors='pt', add_special_tokens=True)          
                temp_data['task2_decoder_ids'], temp_data['task2_decoder_attn'] = temp4['input_ids'], temp4['attention_mask']                      

                data.append(temp_data)
        return data        

    def process_Pipeline_data_format_relprompt(self, dataset, labels_set, relation_discribe):
        # ipdb.set_trace()  
        rel_dict = {rel:[] for rel in labels_set}
        data = []
        for sent in dataset.sents:
            sents = [self.encoder.encode_to_line(trip) for trip in sent.triplets]# sent : Sentence(triplets=[RelationSentence(tokens=['In', '1689'...], head=[11, 12], tail=[16], label='after a work by', head_id='', tail_id='', label_id='P140', error='', raw='', keyword=[''], score=0.0, zerorc_included=True)])
            #sents = ['{"text": "[SENT] : In 1689 , Konstanty was one of the judges who sentenced Kazimierz \\u0141yszczy\\u0144ski to death for atheism .", "summary": "[HEAD] Kazimierz \\u0141yszczy\\u0144ski , [TAIL] atheism , [REL] after a work by ."}\n']
            trips = [trip for trip in sent.triplets]
            #trips = [RelationSentence(tokens=['In', '1689', ',', 'Konstanty', 'was', 'one', 'of', 'the', 'judges', 'who', 'sentenced', 'Kazimierz', 'Łyszczyński', 'to', 'death', 'for', 'atheism', '.'], head=[11, 12], tail=[16], label='after a work by', head_id='', tail_id='', label_id='P140', error='', raw='', keyword=[''], score=0.0, zerorc_included=True)]
            for sent, trip in zip(sents, trips):
                temp_data = {}
                temp_train = {}
                temp_prompt = []

                context, labels = self.encoder.parse_line(sent)# content:'[SENT] : Merpati flight 106 departed Jakarta ( CGK ) on a domestic flight to Tanjung Pandan ( TJQ ) .'   labels:'[HEAD] Kazimierz Łyszczyński , [TAIL] atheism , [REL] after a work by .'
                # ipdb.set_trace()
                temp_train['input_ids'], temp_train['attention_mask'] = [], []
                temp_train['labels'], temp_train['decoder_attention_mask'] = [], []
                # context1 = context + ' Choose [REL] from [rel_candidate]:'
                # context2 = context + ' Extract <triplet> from [sent]:'
                dis_id = trip.label_id
                dis_name = [relation_discribe[dis_id][0]]
                discribe = [relation_discribe[dis_id][2]] #这里
                candidate_rel = random.sample(labels_set,self.opt.n_unseen-1) # ['participant in', 'position held', 'constellation', 'member of']
                rand_int = random.randint(0, self.opt.n_unseen - 1) # 4
                temp_prompt.extend(candidate_rel[:rand_int]) #['participant in', 'position held', 'constellation', 'member of']
                temp_prompt.append(dis_name[0]) #['participant in', 'position held', 'constellation', 'member of', 'after a work by']
                temp_prompt.extend(candidate_rel[rand_int:])
                prompt =  '<Task1>, Choose [REL] in [sent] from [rel_candidate]. [rel_candidate]: '+', '.join(temp_prompt)+ '. '+ context
                temp1 = self.tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
                prompt_ids, prompt_attn = temp1['input_ids'], temp1['attention_mask']
                temp_data['task1_input_ids'], temp_data['task1_input_attn'] = prompt_ids, prompt_attn
                
                temp_label_name = f'[REL]: {trip.label}'
                temp2 = self.tokenizer(temp_label_name, return_tensors='pt', add_special_tokens=True)
                label_name, label_name_attn = temp2['input_ids'], temp2['attention_mask']
                temp_data['task1_decoder_ids'], temp_data['task1_decoder_attn'] = label_name, label_name_attn


                fin_discribe = f'<Task2>, Extract <triplet> in [sent] by [Relation_discribe]. [Relation_discribe]: {dis_name[0]}, {discribe[0]}. {context}'
                temp3 = self.tokenizer(fin_discribe, return_tensors='pt', add_special_tokens=True)
                temp_data['task2_input_ids'], temp_data['task2_input_attn'] = temp3['input_ids'], temp3['attention_mask']

                head,tail,rel = self.encoder.decode_to_relationprompt(labels)
                labels = f"Head Entity : {head} , Tail Entity : {tail} , Relation : {rel} ."

                token_label = labels
                temp4 = self.tokenizer(token_label, return_tensors='pt', add_special_tokens=True)          
                temp_data['task2_decoder_ids'], temp_data['task2_decoder_attn'] = temp4['input_ids'], temp4['attention_mask']                      

                data.append(temp_data)
        return data    

    def process_Pipeline_ActiveMeta_data(self, dataset, labels_set, relation_discribe):
        # ipdb.set_trace()  
        rel_dict = {rel:[] for rel in labels_set}
        data = []
        
        HEAD_addtoken = self.tokenizer('[HEAD]', return_tensors='pt', add_special_tokens=False).input_ids
        TAIL_addtoken = self.tokenizer('[TAIL]', return_tensors='pt', add_special_tokens=False).input_ids
        
        for sent in dataset.sents:
            sents = [self.encoder.encode_to_line(trip) for trip in sent.triplets]# sent : Sentence(triplets=[RelationSentence(tokens=['In', '1689'...], head=[11, 12], tail=[16], label='after a work by', head_id='', tail_id='', label_id='P140', error='', raw='', keyword=[''], score=0.0, zerorc_included=True)])
            #sents = ['{"text": "[SENT] : In 1689 , Konstanty was one of the judges who sentenced Kazimierz \\u0141yszczy\\u0144ski to death for atheism .", "summary": "[HEAD] Kazimierz \\u0141yszczy\\u0144ski , [TAIL] atheism , [REL] after a work by ."}\n']
            trips = [trip for trip in sent.triplets]
            #trips = [RelationSentence(tokens=['In', '1689', ',', 'Konstanty', 'was', 'one', 'of', 'the', 'judges', 'who', 'sentenced', 'Kazimierz', 'Łyszczyński', 'to', 'death', 'for', 'atheism', '.'], head=[11, 12], tail=[16], label='after a work by', head_id='', tail_id='', label_id='P140', error='', raw='', keyword=[''], score=0.0, zerorc_included=True)]
            for sent, trip in zip(sents, trips):
                temp_data = {}
                temp_train = {}
                temp_prompt = []

                context, labels = self.encoder.parse_line(sent)# content:'[SENT] : Merpati flight 106 departed Jakarta ( CGK ) on a domestic flight to Tanjung Pandan ( TJQ ) .'   labels:'[HEAD] Kazimierz Łyszczyński , [TAIL] atheism , [REL] after a work by .'
                # ipdb.set_trace()
                temp_train['input_ids'], temp_train['attention_mask'] = [], []
                temp_train['labels'], temp_train['decoder_attention_mask'] = [], []
                # context1 = context + ' Choose [REL] from [rel_candidate]:'
                # context2 = context + ' Extract <triplet> from [sent]:'
                dis_id = trip.label_id
                dis_name = [relation_discribe[dis_id][0]]
                discribe = [relation_discribe[dis_id][1]] #这里
                candidate_rel = random.sample(labels_set,self.opt.n_unseen-1) # ['participant in', 'position held', 'constellation', 'member of']
                rand_int = random.randint(0, self.opt.n_unseen - 1) # 4
                temp_prompt.extend(candidate_rel[:rand_int]) #['participant in', 'position held', 'constellation', 'member of']
                temp_prompt.append(dis_name[0]) #['participant in', 'position held', 'constellation', 'member of', 'after a work by']
                temp_prompt.extend(candidate_rel[rand_int:])
                prompt =  '<Task1>, Choose [REL] in [sent] from [rel_candidate]. [rel_candidate]: '+', '.join(temp_prompt)+ '. '+ context
                temp1 = self.tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
                prompt_ids, prompt_attn = temp1['input_ids'], temp1['attention_mask']
                temp_data['task1_input_ids'], temp_data['task1_input_attn'] = prompt_ids, prompt_attn
                
                temp_label_name = f'[REL]: {trip.label}'
                temp2 = self.tokenizer(temp_label_name, return_tensors='pt', add_special_tokens=True)
                label_name, label_name_attn = temp2['input_ids'], temp2['attention_mask']
                temp_data['task1_decoder_ids'], temp_data['task1_decoder_attn'] = label_name, label_name_attn


                fin_discribe = f'<Task2>, Extract <triplet> in [sent] by [Relation_discribe]. [Relation_discribe]: {dis_name[0]}, {discribe[0]}. {context}'
                temp3 = self.tokenizer(fin_discribe, return_tensors='pt', add_special_tokens=True)
                temp_data['task2_input_ids'], temp_data['task2_input_attn'] = temp3['input_ids'], temp3['attention_mask']


                token_label = '<triplet>: '+labels
                temp4 = self.tokenizer(token_label, return_tensors='pt', add_special_tokens=True)          
                temp_data['task2_decoder_ids'], temp_data['task2_decoder_attn'] = temp4['input_ids'], temp4['attention_mask']                   
                
                HEAD_addtoken_pos_encode = find_sublist_index(temp3.input_ids[0], HEAD_addtoken[0])
                TAIL_addtoken_pos_encode = find_sublist_index(temp3.input_ids[0], TAIL_addtoken[0])
                HEAD_addtoken_pos_decode = find_sublist_index(temp4.input_ids[0], HEAD_addtoken[0])
                TAIL_addtoken_pos_decode = find_sublist_index(temp4.input_ids[0], TAIL_addtoken[0])
                
                # ipdb.set_trace()
                if trip.head !=[] and trip.tail !=[]:
                    HEADstart_pos_encode = find_sublist_index(temp3.input_ids[0].tolist(), self.tokenizer(trip.tokens[trip.head[0]], add_special_tokens=False).input_ids)
                    TAILstart_pos_encode = find_sublist_index(temp3.input_ids[0].tolist(), self.tokenizer(trip.tokens[trip.tail[0]], add_special_tokens=False).input_ids)    
                    HEADstart_pos_decode = find_sublist_index(temp4.input_ids[0].tolist(), self.tokenizer(trip.tokens[trip.head[0]], add_special_tokens=False).input_ids)
                    TAILstart_pos_decode = find_sublist_index(temp4.input_ids[0].tolist(), self.tokenizer(trip.tokens[trip.tail[0]], add_special_tokens=False).input_ids)
                else:
                    HEADstart_pos_encode = 0
                    TAILstart_pos_encode = 0
                    HEADstart_pos_decode = 0
                    TAILstart_pos_decode = 0                    
                HEAD_length = len(trip.head)
                TAIL_length = len(trip.tail)
                
                temp_data['pos_encode'] = {"HEAD_addtoken_pos_encode":HEAD_addtoken_pos_encode, "TAIL_addtoken_pos_encode":TAIL_addtoken_pos_encode, "HEADstart_pos_encode":HEADstart_pos_encode, "TAILstart_pos_encode":TAILstart_pos_encode, "HEAD_length":HEAD_length, "TAIL_length":TAIL_length}
                temp_data['pos_decode'] = {"HEAD_addtoken_pos_decode":HEAD_addtoken_pos_decode, "TAIL_addtoken_pos_decode":TAIL_addtoken_pos_decode, "HEADstart_pos_decode":HEADstart_pos_decode, "TAILstart_pos_decode":TAILstart_pos_decode, "HEAD_length":HEAD_length, "TAIL_length":TAIL_length}
                # ipdb.set_trace()
                data.append(temp_data)
        return data

    def process_Pipeline_ActiveMeta_Metric_data(self, dataset, labels_set, relation_discribe):
        # ipdb.set_trace()  
        rel_dict = {rel:[] for rel in labels_set}
        data = []
        
        HEAD_addtoken = self.tokenizer('[HEAD]', return_tensors='pt', add_special_tokens=False).input_ids
        TAIL_addtoken = self.tokenizer('[TAIL]', return_tensors='pt', add_special_tokens=False).input_ids
        REL_addtoken = self.tokenizer('[REL]', return_tensors='pt', add_special_tokens=False).input_ids
        
        for sent in dataset.sents:
            sents = [self.encoder.encode_to_line(trip) for trip in sent.triplets]# sent : Sentence(triplets=[RelationSentence(tokens=['In', '1689'...], head=[11, 12], tail=[16], label='after a work by', head_id='', tail_id='', label_id='P140', error='', raw='', keyword=[''], score=0.0, zerorc_included=True)])
            #sents = ['{"text": "[SENT] : In 1689 , Konstanty was one of the judges who sentenced Kazimierz \\u0141yszczy\\u0144ski to death for atheism .", "summary": "[HEAD] Kazimierz \\u0141yszczy\\u0144ski , [TAIL] atheism , [REL] after a work by ."}\n']
            trips = [trip for trip in sent.triplets]
            #trips = [RelationSentence(tokens=['In', '1689', ',', 'Konstanty', 'was', 'one', 'of', 'the', 'judges', 'who', 'sentenced', 'Kazimierz', 'Łyszczyński', 'to', 'death', 'for', 'atheism', '.'], head=[11, 12], tail=[16], label='after a work by', head_id='', tail_id='', label_id='P140', error='', raw='', keyword=[''], score=0.0, zerorc_included=True)]
            for sent, trip in zip(sents, trips):
                temp_data = {}
                temp_train = {}
                temp_prompt = []

                context, labels = self.encoder.parse_line(sent)# content:'[SENT] : Merpati flight 106 departed Jakarta ( CGK ) on a domestic flight to Tanjung Pandan ( TJQ ) .'   labels:'[HEAD] Kazimierz Łyszczyński , [TAIL] atheism , [REL] after a work by .'
                # ipdb.set_trace()
                temp_train['input_ids'], temp_train['attention_mask'] = [], []
                temp_train['labels'], temp_train['decoder_attention_mask'] = [], []
                # context1 = context + ' Choose [REL] from [rel_candidate]:'
                # context2 = context + ' Extract <triplet> from [sent]:'
                dis_id = trip.label_id
                dis_name = [relation_discribe[dis_id][0]]
                discribe = [relation_discribe[dis_id][1]] #这里
                candidate_rel = random.sample(labels_set,self.opt.n_unseen-1) # ['participant in', 'position held', 'constellation', 'member of']
                rand_int = random.randint(0, self.opt.n_unseen - 1) # 4
                temp_prompt.extend(candidate_rel[:rand_int]) #['participant in', 'position held', 'constellation', 'member of']
                temp_prompt.append(dis_name[0]) #['participant in', 'position held', 'constellation', 'member of', 'after a work by']
                temp_prompt.extend(candidate_rel[rand_int:])
                prompt =  '<Task1>, Choose [REL] in [sent] from [rel_candidate]. [rel_candidate]: '+', '.join(temp_prompt)+ '. '+ context
                temp1 = self.tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
                prompt_ids, prompt_attn = temp1['input_ids'], temp1['attention_mask']
                temp_data['task1_input_ids'], temp_data['task1_input_attn'] = prompt_ids, prompt_attn
                
                temp_label_name = f'[REL]: {trip.label}'
                temp2 = self.tokenizer(temp_label_name, return_tensors='pt', add_special_tokens=True)
                label_name, label_name_attn = temp2['input_ids'], temp2['attention_mask']
                temp_data['task1_decoder_ids'], temp_data['task1_decoder_attn'] = label_name, label_name_attn


                fin_discribe = f'<Task2>, Extract <triplet> in [sent] by [Relation_discribe]. [Relation_discribe]: {dis_name[0]}, {discribe[0]}. {context}'
                temp3 = self.tokenizer(fin_discribe, return_tensors='pt', add_special_tokens=True)
                temp_data['task2_input_ids'], temp_data['task2_input_attn'] = temp3['input_ids'], temp3['attention_mask']


                token_label = '<triplet>: '+labels
                temp4 = self.tokenizer(token_label, return_tensors='pt', add_special_tokens=True)          
                temp_data['task2_decoder_ids'], temp_data['task2_decoder_attn'] = temp4['input_ids'], temp4['attention_mask']                   
                
                HEAD_addtoken_pos_encode = find_sublist_index(temp3.input_ids[0], HEAD_addtoken[0])
                TAIL_addtoken_pos_encode = find_sublist_index(temp3.input_ids[0], TAIL_addtoken[0])
                
                HEAD_addtoken_pos_decode = find_sublist_index(temp4.input_ids[0], HEAD_addtoken[0])
                TAIL_addtoken_pos_decode = find_sublist_index(temp4.input_ids[0], TAIL_addtoken[0])
                REL_addtoken_pos_decode = find_sublist_index(temp2.input_ids[0], REL_addtoken[0])
                
                # ipdb.set_trace()
                head_pos, tail_pos =[], []
                HEAD_length = len(trip.head)
                TAIL_length = len(trip.tail)
                
                try:
                    HEADstart_pos_encode = find_sublist_index(temp3.input_ids[0].tolist(), self.tokenizer(trip.tokens[trip.head[0]], add_special_tokens=False).input_ids)
                    TAILstart_pos_encode = find_sublist_index(temp3.input_ids[0].tolist(), self.tokenizer(trip.tokens[trip.tail[0]], add_special_tokens=False).input_ids)    
                    head_pos.append([HEADstart_pos_encode, HEADstart_pos_encode + HEAD_length])
                    tail_pos.append([TAILstart_pos_encode, TAILstart_pos_encode + TAIL_length])
                except:
                    break
                
                temp_data['head_pos']=head_pos
                temp_data['tail_pos']=tail_pos
                
                temp_data['HEAD_addtoken_pos_encode']=HEAD_addtoken_pos_encode
                temp_data['TAIL_addtoken_pos_encode']=TAIL_addtoken_pos_encode
                
                temp_data['HEAD_addtoken_pos_decode']=HEAD_addtoken_pos_decode
                temp_data['TAIL_addtoken_pos_decode']=TAIL_addtoken_pos_decode
                # ipdb.set_trace()
                temp_data['REL_addtoken_pos_decode'] = torch.where(temp2['input_ids'] == 32103, )[1].tolist()
                
                rel_pos = []
                # ipdb.set_trace()
                rel_idx = self.tokenizer(trip.label, return_tensors='pt', add_special_tokens=False)['input_ids']
                rel_len = rel_idx.shape[1]  #rel长度
                try:
                    start_list = torch.where(temp1['input_ids'] == rel_idx[0,0],)[1].tolist()[0]
                except:
                    break
                rel_pos.append([start_list, start_list + rel_len])
                temp_data['rel_pos']=rel_pos

                # ipdb.set_trace()
                data.append(temp_data)
        return data

    def get_optimizer(self):
        no_decay = ['bias', 'LayerNorm.weight']
        other_lr_name = ['adapter']
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if
                           not any(nd in n for nd in no_decay) and not any(nd in n for nd in other_lr_name)],
                "weight_decay": self.opt.weight_decay,
                "lr": self.opt.lr
            },
            {
                "params": [p for n, p in self.model.named_parameters() if
                           any(nd in n for nd in no_decay) and not any(nd in n for nd in other_lr_name)],
                "weight_decay": 0.0,
                "lr": self.opt.lr
            },
            {
                "params": [p for n, p in self.model.named_parameters() if
                           not any(nd in n for nd in no_decay) and any(nd in n for nd in other_lr_name)],
                "weight_decay": self.opt.weight_decay,
                "lr": self.opt.aux_lr
            },
            {
                "params": [p for n, p in self.model.named_parameters() if
                           any(nd in n for nd in no_decay) and any(nd in n for nd in other_lr_name)],
                "weight_decay": 0.0,
                "lr": self.opt.aux_lr
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, eps=self.opt.adam_epsilon)
        t_total = (len(self.train_dataset.sents) // self.opt.batch_size // self.opt.gradient_accumulation_steps * float(self.opt.epochs))
        warmup_steps = int(self.opt.warmup_ratio * t_total)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total) if self.opt.use_scheduler else None
        return optimizer, scheduler

    def get_optimizer_meta(self, model):
        no_decay = ['bias', 'LayerNorm.weight']
        other_lr_name = ['adapter']
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if
                           not any(nd in n for nd in no_decay) and not any(nd in n for nd in other_lr_name)],
                "weight_decay": self.opt.weight_decay,
                "lr": self.opt.lr
            },
            {
                "params": [p for n, p in model.named_parameters() if
                           any(nd in n for nd in no_decay) and not any(nd in n for nd in other_lr_name)],
                "weight_decay": 0.0,
                "lr": self.opt.lr
            },
            {
                "params": [p for n, p in model.named_parameters() if
                           not any(nd in n for nd in no_decay) and any(nd in n for nd in other_lr_name)],
                "weight_decay": self.opt.weight_decay,
                "lr": self.opt.aux_lr
            },
            {
                "params": [p for n, p in model.named_parameters() if
                           any(nd in n for nd in no_decay) and any(nd in n for nd in other_lr_name)],
                "weight_decay": 0.0,
                "lr": self.opt.aux_lr
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, eps=self.opt.adam_epsilon)
        t_total = (len(self.train_dataset.sents) // self.opt.batch_size // self.opt.gradient_accumulation_steps * float(self.opt.epochs))
        warmup_steps = int(self.opt.warmup_ratio * t_total)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total) if self.opt.use_scheduler else None
        return optimizer, scheduler

    def pre_train_process(self, aux_loss_weight=0.):
        train_step, best_step, best_epoch, best_dev_score = 0, 0, 0, 1000

        data_loader = self.train_dataloader

        for epoch in range(self.opt.epochs):
            self.model.train()
            bar = tqdm(data_loader)
            len_data = len(data_loader)//4-1

            for i, batch in enumerate(bar):
                loss = 0
                train_step += 1
                # ipdb.set_trace()
                batch = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}
                # batch['aux_loss_weight']=aux_loss_weight
                outputs = self.step(batch)
                loss += outputs.loss

                loss = loss / self.opt.gradient_accumulation_steps
                loss.backward()
                if train_step % self.opt.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.opt.gradient_clip_val)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.scheduler:
                        self.scheduler.step()
                
                if self.opt.do_eval and (i+1) % len_data==0: #  and (i+1) % len_data==0
                    score = self.evaluate_extract()
                    checkpoint_path = self.opt.output_path + f'checkpoint_{self.opt.dataset}_{epoch}_{train_step}_{score}/'
                    print(f"Save model {checkpoint_path}!")
                    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)  # 如果没有要创建文件夹
                    self.model.save_pretrained(checkpoint_path)
                    self.tokenizer.save_pretrained(checkpoint_path)
                    if score < best_dev_score:
                        best_dev_score = score
                        best_epoch = epoch
                        best_step = train_step

        best_path = self.opt.output_path + f'checkpoint_{self.opt.dataset}_{best_epoch}_{best_step}_{best_dev_score}/'
        if Path(self.opt.output_path + 'best_extractor/').exists():
            shutil.rmtree(self.opt.output_path + 'best_extractor/')
        shutil.move(best_path, self.opt.output_path + 'best_extractor/')
        delete_checkpoints(self.opt.output_path)
        return self.opt.output_path + 'best_extractor/'
    
    def Baseline_train_process(self, aux_loss_weight=0.):
        train_step, best_step, best_epoch, best_dev_score = 0, 0, 0, 1000

        data_loader = self.train_dataloader

        for epoch in range(self.opt.epochs):
            self.model.train()
            bar = tqdm(data_loader)
            len_data = len(data_loader)//4-1

            for i, batch in enumerate(bar):
                loss = 0
                train_step += 1
                # ipdb.set_trace()
                batch = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}
                # batch['aux_loss_weight']=aux_loss_weight
                outputs = self.step(batch)
                loss += outputs.loss
                
                loss = loss / self.opt.gradient_accumulation_steps
                loss.backward()
                if train_step % self.opt.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.opt.gradient_clip_val)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.scheduler:
                        self.scheduler.step()
                
                if self.opt.do_eval and (i+1) % len_data==0: #  and (i+1) % len_data==0
                    score = self.evaluate_extract()
                    checkpoint_path = self.opt.baseline_path + 'extractor/' + f'checkpoint_{self.opt.dataset}_{epoch}_{train_step}_{score}/'
                    print(f"Save model {checkpoint_path}!")
                    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)  # 如果没有要创建文件夹
                    self.model.save_pretrained(checkpoint_path)
                    self.tokenizer.save_pretrained(checkpoint_path)
                    if score < best_dev_score:
                        best_dev_score = score
                        best_epoch = epoch
                        best_step = train_step

        best_path = self.opt.baseline_path + 'extractor/'+ f'checkpoint_{self.opt.dataset}_{best_epoch}_{best_step}_{best_dev_score}/'
        if Path(self.opt.baseline_path + 'best_extractor/').exists():
            shutil.rmtree(self.opt.baseline_path + 'best_extractor/')
        shutil.move(best_path, self.opt.baseline_path + 'best_extractor/')
        delete_checkpoints(self.opt.baseline_path + 'extractor/')
        return self.opt.baseline_path + 'best_extractor/'

    def Recall_train_process(self, aux_loss_weight=0.):
        train_step, best_step, best_epoch, best_dev_score = 0, 0, 0, 1000

        data_loader = self.train_dataloader

        for epoch in range(self.opt.epochs):
            self.model.train()
            bar = tqdm(data_loader)
            len_data = len(data_loader)//4-1

            for i, batch in enumerate(bar):
                loss = 0
                train_step += 1
                ipdb.set_trace()
                batch = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}
                # batch['aux_loss_weight']=aux_loss_weight
                outputs = self.step(batch)

                loss += outputs.loss
                
                loss = loss / self.opt.gradient_accumulation_steps
                loss.backward()
                if train_step % self.opt.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.opt.gradient_clip_val)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.scheduler:
                        self.scheduler.step()
                
                if self.opt.do_eval and (i+1) % len_data==0: #  and (i+1) % len_data==0
                    score = self.evaluate_extract()
                    checkpoint_path = self.opt.recall_path + 'extractor/' + f'checkpoint_{self.opt.dataset}_{epoch}_{train_step}_{score}/'
                    print(f"Save model {checkpoint_path}!")
                    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)  # 如果没有要创建文件夹
                    self.model.save_pretrained(checkpoint_path)
                    self.tokenizer.save_pretrained(checkpoint_path)
                    if score < best_dev_score:
                        best_dev_score = score
                        best_epoch = epoch
                        best_step = train_step

        best_path = self.opt.recall_path + 'extractor/'+ f'checkpoint_{self.opt.dataset}_{best_epoch}_{best_step}_{best_dev_score}/'
        if Path(self.opt.recall_path + 'best_extractor/').exists():
            shutil.rmtree(self.opt.recall_path + 'best_extractor/')
        shutil.move(best_path, self.opt.recall_path + 'best_extractor/')
        delete_checkpoints(self.opt.recall_path + 'extractor/')
        return self.opt.recall_path + 'best_extractor/'

    def Pipeline_train_process(self, aux_loss_weight=0.):
        train_step, best_step, best_epoch, best_dev_score = 0, 0, 0, 1000

        data_loader = self.train_dataloader

        for epoch in range(self.opt.epochs):
            self.model.train()
            bar = tqdm(data_loader)
            len_data = len(data_loader)//4-1

            for i, batch in enumerate(bar):
                loss = 0
                train_step += 1
                # self.tokenizer.decode(batch['labels'][0].tolist())
                # ipdb.set_trace()
                batch = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}
                # batch['aux_loss_weight']=aux_loss_weight
                query_dict = {'help_input_ids':'input_ids', 'help_attention_mask':'attention_mask', 'help_decoder_attention_mask':'decoder_attention_mask', 'help_labels':'labels'}                
                batch1 = {query_dict[k]: v for k, v in batch.items() if k in query_dict}
                batch2 = {k: v for k, v in batch.items() if k not in query_dict}
                # ipdb.set_trace()
                outputs = self.step(batch2)
                loss += outputs.loss

                outputs = self.step(batch1)
                loss += outputs.loss
                loss = loss / self.opt.gradient_accumulation_steps
                loss.backward()
                
                if train_step % self.opt.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.opt.gradient_clip_val)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.scheduler:
                        self.scheduler.step()
                
                if self.opt.do_eval and (i+1) % len_data==0: #  and (i+1) % len_data==0
                    score = self.evaluate_extract()
                    checkpoint_path = self.opt.pipeline_bsz4_ep1 + 'extractor/' + f'checkpoint_{self.opt.dataset}_{epoch}_{train_step}_{score}/'
                    print(f"Save model {checkpoint_path}!")
                    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)  # 如果没有要创建文件夹
                    self.model.save_pretrained(checkpoint_path)
                    self.tokenizer.save_pretrained(checkpoint_path)
                    if score < best_dev_score:
                        best_dev_score = score
                        best_epoch = epoch
                        best_step = train_step

        best_path = self.opt.pipeline_bsz4_ep1 + 'extractor/'+ f'checkpoint_{self.opt.dataset}_{best_epoch}_{best_step}_{best_dev_score}/'
        if Path(self.opt.pipeline_bsz4_ep1 + 'best_extractor/').exists():
            shutil.rmtree(self.opt.pipeline_bsz4_ep1 + 'best_extractor/')
        shutil.move(best_path, self.opt.pipeline_bsz4_ep1 + 'best_extractor/')
        delete_checkpoints(self.opt.pipeline_bsz4_ep1 + 'extractor/')
        return self.opt.pipeline_bsz4_ep1 + 'best_extractor/'

    def Pipeline_train_process_ActiveMeta_Metric(self, aux_loss_weight=0.):
        train_step, best_step, best_epoch, best_dev_score = 0, 0, 0, 1000

        data_loader = self.train_dataloader

        for epoch in range(self.opt.epochs):
            self.model.train()
            bar = tqdm(data_loader)
            len_data = len(data_loader)//4-1

            for i, batch in enumerate(bar):
                loss = 0
                train_step += 1
                # self.tokenizer.decode(batch['labels'][0].tolist())
                # ipdb.set_trace()
                batch = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}
                
                # batch['aux_loss_weight']=aux_loss_weight
                query_dict = {'help_input_ids':'input_ids', 'help_attention_mask':'attention_mask', 'help_decoder_attention_mask':'decoder_attention_mask', 'help_labels':'labels'}                
                batch1 = {query_dict[k]: v for k, v in batch.items() if k in query_dict }
                batch1["head_pos"]=batch["head_pos"]
                batch1["tail_pos"]=batch["tail_pos"]
                batch1["HEAD_addtoken_pos_encode"]=batch["HEAD_addtoken_pos_encode"]
                batch1["TAIL_addtoken_pos_encode"]=batch["TAIL_addtoken_pos_encode"]
                batch1["HEAD_addtoken_pos_decode"]=batch["HEAD_addtoken_pos_decode"]
                batch1["TAIL_addtoken_pos_decode"]=batch["TAIL_addtoken_pos_decode"]
                
                batch2 = {k: v for k, v in batch.items() if k not in query_dict}
                batch2["rel_pos"]=batch["rel_pos"]
                batch2["REL_addtoken_pos_decode"]=batch["REL_addtoken_pos_decode"]
                # ipdb.set_trace()
                
                outputs = self.model(**batch2, is_pipeline1=True)
                loss += outputs.loss

                # ipdb.set_trace()
                outputs = self.model(**batch1, is_pipeline1=False)
                loss += outputs.loss
                loss = loss / self.opt.gradient_accumulation_steps
                loss.backward()
                
                if train_step % self.opt.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.opt.gradient_clip_val)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.scheduler:
                        self.scheduler.step()
                
                if self.opt.do_eval and (i+1) % len_data==0: #  and (i+1) % len_data==0
                    score = self.evaluate_extract()
                    checkpoint_path = self.opt.pipeline_ActiveMeta_Metricplus_bsz1_ep1 + 'extractor/' + f'checkpoint_{self.opt.dataset}_{epoch}_{train_step}_{score}/'
                    print(f"Save model {checkpoint_path}!")
                    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)  # 如果没有要创建文件夹
                    self.model.save_pretrained(checkpoint_path)
                    self.tokenizer.save_pretrained(checkpoint_path)
                    if score < best_dev_score:
                        best_dev_score = score
                        best_epoch = epoch
                        best_step = train_step

        best_path = self.opt.pipeline_ActiveMeta_Metricplus_bsz1_ep1 + 'extractor/'+ f'checkpoint_{self.opt.dataset}_{best_epoch}_{best_step}_{best_dev_score}/'
        if Path(self.opt.pipeline_ActiveMeta_Metricplus_bsz1_ep1 + 'best_extractor/').exists():
            shutil.rmtree(self.opt.pipeline_ActiveMeta_Metricplus_bsz1_ep1 + 'best_extractor/')
        shutil.move(best_path, self.opt.pipeline_ActiveMeta_Metricplus_bsz1_ep1 + 'best_extractor/')
        delete_checkpoints(self.opt.pipeline_ActiveMeta_Metricplus_bsz1_ep1 + 'extractor/')
        return self.opt.pipeline_ActiveMeta_Metricplus_bsz1_ep1 + 'best_extractor/'

    def Pipeline_train_meta_process(self, config, aux_loss_weight=0.):
        # torch.autograd.set_detect_anomaly(True)
        train_step, best_step, best_epoch, best_dev_score = 0, 0, 0, 1000

        data_loader = self.train_dataloader

        for epoch in range(self.opt.epochs):
            self.model.train()
            bar = tqdm(data_loader)
            len_data = len(data_loader)//4-1
            # model_state_dict = self.model.t5.state_dict()
            model_state_dict = copy.deepcopy(self.model.t5.state_dict())
            meta_loss = []
            for i, batch in enumerate(bar):
                # print(f'{i}:{torch.cuda.memory_allocated(0)}')
                # loss = 0
                # train_step += 1
                # copy_model = copy.deepcopy(self.model.t5)
                meta_loss.append(self.pipeforward(batch, model_state_dict))
            # loss = loss / self.opt.gradient_accumulation_steps
            self.model.t5.load_state_dict(model_state_dict)
            self.optimizer.zero_grad()
            meta_loss = torch.stack(meta_loss).mean()
            final_meta_loss = meta_loss.detach()
            # meta_loss = torch.mean(torch.stack(meta_loss))
            # ipdb.set_trace()
            final_meta_loss.requires_grad_(True)
            final_meta_loss.backward()
            # meta_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.opt.gradient_clip_val)
            self.optimizer.step()
            if self.scheduler:
               self.scheduler.step()
            # if train_step % self.opt.gradient_accumulation_steps == 0:
            # if self.scheduler:
            #     self.scheduler.step()
            # if self.opt.do_eval and (i+1) % len_data==0: #  and (i+1) % len_data==0
            if self.opt.do_eval: #  and (i+1) % len_data==0            
                score = self.evaluate_extract()
                checkpoint_path = self.opt.pipeline_meta_path + 'extractor/' + f'checkpoint_{self.opt.dataset}_{epoch}_{train_step}_{score}/'
                print(f"Save model {checkpoint_path}!")
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)  # 如果没有要创建文件夹
                self.model.save_pretrained(checkpoint_path)
                self.tokenizer.save_pretrained(checkpoint_path)
                if score < best_dev_score:
                    best_dev_score = score
                    best_epoch = epoch
                    best_step = train_step

        best_path = self.opt.pipeline_meta_path + 'extractor/'+ f'checkpoint_{self.opt.dataset}_{best_epoch}_{best_step}_{best_dev_score}/'
        if Path(self.opt.pipeline_meta_path + 'best_extractor/').exists():
            shutil.rmtree(self.opt.pipeline_meta_path + 'best_extractor/')
        shutil.move(best_path, self.opt.pipeline_meta_path + 'best_extractor/')
        delete_checkpoints(self.opt.pipeline_meta_path + 'extractor/')
        return self.opt.pipeline_meta_path + 'best_extractor/'

    def evaluate_extract(self):
        with torch.no_grad():
            self.model.eval()
            bar = tqdm(self.dev_dataloader)
            length = len(self.dev_dataloader)
            sentences, predicts, targets, score = [], [], [], 0
            for i, batch in enumerate(bar):
                batch = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}
                # ipdb.set_trace()
                # outputs = self.model.predict(**batch, return_dict=True, output_hidden_states=True)#
                # outputs = self.model.predict(batch['input_ids'], batch['attention_mask'])
                outputs = self.model(**batch)#这里
                score += outputs.loss
                score = score / self.opt.gradient_accumulation_steps
            score = score/length
        return score
    
    def pretrain_predict(self, model_path, device=torch.device("cuda:0")):
        data = self.test_dataloader
        # accuracy_metric = evaluate.load("/user_data/wujy/SimonHeye/META/metrics/accuracy/accuracy.py")
        F1_metric = evaluate.load("/user_data/wujy/SimonHeye/META/metrics/f1/f1.py")
        precision_metric = evaluate.load("/user_data/wujy/SimonHeye/META/metrics/precision/precision.py")
        recall_metric = evaluate.load("/user_data/wujy/SimonHeye/META/metrics/recall/recall.py")
        # rouge_metic = evaluate.load("/user_data/wujy/SimonHeye/META/metrics/rouge/rouge.py")
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        model = model.to(device)
        model.eval()
        bar = tqdm(data)
        for i, batch in enumerate(bar):
            batch = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}
            with torch.no_grad():
                outputs = model(**batch)
            # pdb.set_trace()
            logits = outputs.logits
            # accuracy_predictions = torch.argmax(logits, dim=-1).squeeze(0).tolist()
            # rouge_predictions = self.tokenizer.batch_decode(torch.argmax(logits, dim=-1))
            precision_predictions = torch.argmax(logits, dim=-1).squeeze(0).tolist()
            recall_predictions = torch.argmax(logits, dim=-1).squeeze(0).tolist()
            F1_predictions = torch.argmax(logits, dim=-1).squeeze(0).tolist()
            # accuracy_metric.add_batch(predictions = accuracy_predictions, references = batch['labels'].squeeze(0).tolist())
            # rouge_metic.add_batch(predictions = rouge_predictions, references = self.tokenizer.batch_decode(batch['labels']))
            precision_metric.add_batch(predictions=precision_predictions, references=batch['labels'].squeeze(0).tolist())
            recall_metric.add_batch(predictions=recall_predictions, references=batch['labels'].squeeze(0).tolist())
            F1_metric.add_batch(predictions=F1_predictions, references=batch['labels'].squeeze(0).tolist())
        return precision_metric.compute(average="micro"), recall_metric.compute(average="micro"), F1_metric.compute(average="micro")
        # return F1_metric.compute(average="micro")
        # return accuracy_metric.compute() , rouge_metic.compute(use_stemmer=True)
        
    def Baseline_predict(self, path_in, path_out, model_path, pid2name='', use_label_constraint=True, max_target_length=128, device=torch.device("cuda")):
        data = Dataset.load(path_in)
        pred_labels = data.get_labels()
        texts = [trip.text for sent in data.sents for trip in sent.triplets]
        test_data = self.process_Baseline_data(data, pred_labels, self.relation_discribe)
        self.test_dataloader = DataLoader(test_data,
                                           batch_size=self.opt.batch_size,
                                           num_workers=2,
                                           drop_last=False,
                                           shuffle=False,
                                           collate_fn=collate_fn_Baseline)
        gen = TextGenerator(
            model=T5ForConditionalGeneration.from_pretrained(model_path),
            tokenizer=T5Tokenizer.from_pretrained(model_path),
            max_length=max_target_length,
        )
        gen.model = gen.model.to(device)
        constraint = LabelConstraint(labels=pred_labels, tokenizer=self.tokenizer)
        sents, sents_no = [], 0
        for batch in tqdm(self.test_dataloader):
            del batch['labels']
            batch = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}
            outputs = gen.run(
                batch, num_beams=1, save_scores=use_label_constraint, do_sample=False, num_return=1,
            )

            for j, raw in enumerate(outputs):
                triplet = self.encoder.safe_decode(texts[sents_no], y=raw)
                sents_no+=1
                if use_label_constraint:
                    assert gen.scores is not None
                    triplet = constraint.run(triplet, gen.scores[j])
                sents.append(Sentence(triplets=[triplet]))

        Dataset(sents=sents).save(path_out)

    def Recall_predict(self, path_in, path_out, model_path, pid2name='', use_label_constraint=True, max_target_length=128, device=torch.device("cuda")):
        path = path_out
        data = Dataset.load(path_in)
        pred_labels = data.get_labels()
        texts = [trip.text for sent in data.sents for trip in sent.triplets]#所有句子构成的列表
        # ipdb.set_trace()
        test_data = self.process_Recall_data(data, pred_labels, self.relation_discribe)
        self.test_dataloader = DataLoader(test_data,
                                        #    batch_size=self.opt.batch_size,
                                           batch_size=1,
                                           num_workers=2,
                                           drop_last=False,
                                           shuffle=False,
                                           collate_fn=collate_fn_Recall)
        gen = TextGenerator(
            model=T5ForConditionalGeneration.from_pretrained(model_path),
            tokenizer=T5Tokenizer.from_pretrained(model_path),
            max_length=max_target_length,
        )
        gen.model = gen.model.to(device)
        constraint = LabelConstraint_format(labels=pred_labels, tokenizer=self.tokenizer,)
        sents, sents_no = [], 0

        for batch in tqdm(self.test_dataloader):
            del batch['labels']
            batch = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}
            outputs = gen.run(
                batch, num_beams=1, save_scores=use_label_constraint, do_sample=False, num_return=1,
            )

            for j, raw in enumerate(outputs):
                # ipdb.set_trace()
                triple = raw.split(" <triplet>: ")[1]
                triplet = self.encoder.safe_decode(texts[sents_no], y=triple)
                sents_no+=1
                if use_label_constraint:
                    assert gen.scores is not None
                    triplet = constraint.run(triplet, gen.scores[j])
                temp_sent = Sentence(triplets=[triplet])
                Trainer.Recall_predict_help(data = temp_sent, path_out = path, gen_model = gen, sents = sents)
                # sents.append(Sentence(triplets=[triplet]))
        Dataset(sents=sents).save(path_out)

    def Recall_predict_help(self, data, path_out, gen_model, sents, pid2name='', use_label_constraint=True, max_target_length=128, device=torch.device("cuda")):
        pred_labels = data.get_labels()
        texts = [trip.text for sent in data.sents for trip in sent.triplets]
        test_data = self.process_Recall_traindata(data, pred_labels, self.relation_discribe)
        self.test_dataloader = DataLoader(test_data,
                                           batch_size=1,
                                        #    batch_size=self.opt.batch_size,
                                           num_workers=2,
                                           drop_last=False,
                                           shuffle=False,
                                           collate_fn=collate_fn_Recall_help)
        gen_model.model = gen_model.model.to(device)
        constraint = LabelConstraint(labels=pred_labels, tokenizer=self.tokenizer)
        sents, sents_no = [], 0
        for batch in tqdm(self.test_dataloader):
            del batch['labels']
            batch = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}
            outputs = gen_model.run(
                batch, num_beams=1, save_scores=use_label_constraint, do_sample=False, num_return=1,
            )

            for j, raw in enumerate(outputs):
                triplet = self.encoder.safe_decode(texts[sents_no], y=raw)
                sents_no+=1
                if use_label_constraint:
                    assert gen_model.scores is not None
                    triplet = constraint.run(triplet, gen_model.scores[j])
                sents.append(Sentence(triplets=[triplet]))

        # Dataset(sents=sents).save(path_out)


def delete_checkpoints(
    folder: str = ".", pattern="**/checkpoint*", delete: bool = True
):
    for p in Path(folder).glob(pattern):
        print(p)
        if delete:
            if p.is_dir():
                shutil.rmtree(p)
            elif p.is_file():
                os.remove(p)
            else:
                raise ValueError("Unknown Type")

class DynamicModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True

class TextGenerator(DynamicModel):
    model: T5ForConditionalGeneration
    tokenizer: T5Tokenizer
    scores: Optional[List[Tensor]] = None
    max_length: int

    def tokenize(self, texts: List[str], **kwargs):
        # ipdb.set_trace()
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            **kwargs,
        ).to(self.model.device)

    def run(
        self,
        # texts: List[str],
        batch,#这里
        do_sample=True,
        top_k=50,
        temperature=1.0,
        num_return: int = 4,
        prompt: Optional[str] = None,
        prompt_ids: Optional[List[int]] = None,
        multi_prompt_ids: Optional[List[List[int]]] = None,
        decoder_input_ids: Optional[Tensor] = None,
        decoder_attention_mask: Optional[Tensor] = None,
        save_scores: bool = False,
        **kwargs,
    ) -> List[str]:
        # https://huggingface.co/transformers/v4.7.0/main_classes/model.html#generation
        tok = self.tokenizer
        eos = tok.eos_token_id#tok.bos_token_Id

        # if prompt is not None:
        #     prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        # if prompt_ids is not None:
        #     prompt_ids = [eos] + prompt_ids
        #     decoder_input_ids = torch.tensor([prompt_ids])
        # if multi_prompt_ids is not None:
        #     assert len(texts) == len(multi_prompt_ids)
        #     multi_prompt_ids = [[eos] + lst for lst in multi_prompt_ids]
        #     decoder_input_ids = torch.tensor(multi_prompt_ids)
        # if decoder_input_ids is not None:
        #     kwargs.update(decoder_input_ids=decoder_input_ids.to(self.model.device))
        #     kwargs.update(decoder_attention_mask=decoder_attention_mask.to(self.model.device))

        # outputs = self.model.generate(
        #     **self.tokenize(texts),
        #     do_sample=do_sample,
        #     top_k=top_k,
        #     temperature=temperature,
        #     num_return_sequences=num_return,
        #     return_dict_in_generate=True,
        #     output_scores=save_scores,
        #     max_length=self.max_length,
        #     **kwargs,
        # )

        if 'decoder_input_ids' in batch:
            batch['decoder_input_ids'] = batch['decoder_input_ids'].to(self.model.device)
        
        if 'desc_pos' in batch:
            outputs = self.model.generate(
                **batch,
                do_sample=do_sample,
                top_k=top_k,
                temperature=temperature,
                num_return_sequences=num_return,
                return_dict_in_generate=True,
                output_scores=save_scores,
                max_length=self.max_length,
            )
        else:
            # ipdb.set_trace()
            # print(1)
            if 'decoder_inputs_embeds' not in batch:
                outputs = self.model.generate(
                    **batch,
                    do_sample=do_sample,
                    top_k=top_k,
                    temperature=temperature,
                    num_return_sequences=num_return,
                    return_dict_in_generate=True,
                    output_scores=save_scores,
                    max_length=self.max_length,
                    # decoder_attention_mask = decoder_attention_mask,
                    # decoder_input_ids = decoder_input_ids,
                    **kwargs,
                )
            else:
                ipdb.set_trace()
                outputs = self.model.generate(
                    **batch,
                    do_sample=do_sample,
                    top_k=top_k,
                    temperature=temperature,
                    num_return_sequences=num_return,
                    return_dict_in_generate=True,
                    output_scores=save_scores,
                    max_length=self.max_length,
                    # decoder_inputs_embeds=decoder_input_ids,
                    # decoder_attention_mask = decoder_attention_mask,
                    # decoder_input_ids = decoder_input_ids,
                    **kwargs,
                )

        self.scores = None
        if save_scores:
            self.scores = [_ for _ in torch.stack(outputs.scores, 1).cpu()]
        # ipdb.set_trace()
        return self.decode(outputs.sequences)

    def decode(self, outputs) -> List[str]:
        tok = self.tokenizer
        texts = tok.batch_decode(
            outputs, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

        # Manually remove <bos><eos><pad> in case we have custom special tokens
        special_tokens = [tok.eos_token, tok.pad_token]
        for i, t in enumerate(texts):
            for token in special_tokens:
                t = t.replace(token, "")
                texts[i] = t
        return texts

class TextGenerator_ActiveMeta(DynamicModel):
    model: T5ForConditionalGenerationWithAdapter
    tokenizer: T5Tokenizer
    scores: Optional[List[Tensor]] = None
    max_length: int

    def tokenize(self, texts: List[str], **kwargs):
        # ipdb.set_trace()
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            **kwargs,
        ).to(self.model.device)

    def run(
        self,
        # texts: List[str],
        batch,#这里
        do_sample=True,
        top_k=50,
        temperature=1.0,
        num_return: int = 4,
        prompt: Optional[str] = None,
        prompt_ids: Optional[List[int]] = None,
        multi_prompt_ids: Optional[List[List[int]]] = None,
        decoder_input_ids: Optional[Tensor] = None,
        decoder_attention_mask: Optional[Tensor] = None,
        save_scores: bool = False,
        **kwargs,
    ) -> List[str]:
        # https://huggingface.co/transformers/v4.7.0/main_classes/model.html#generation
        tok = self.tokenizer
        eos = tok.eos_token_id#tok.bos_token_Id

        # if prompt is not None:
        #     prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        # if prompt_ids is not None:
        #     prompt_ids = [eos] + prompt_ids
        #     decoder_input_ids = torch.tensor([prompt_ids])
        # if multi_prompt_ids is not None:
        #     assert len(texts) == len(multi_prompt_ids)
        #     multi_prompt_ids = [[eos] + lst for lst in multi_prompt_ids]
        #     decoder_input_ids = torch.tensor(multi_prompt_ids)
        # if decoder_input_ids is not None:
        #     kwargs.update(decoder_input_ids=decoder_input_ids.to(self.model.device))
        #     kwargs.update(decoder_attention_mask=decoder_attention_mask.to(self.model.device))

        # outputs = self.model.generate(
        #     **self.tokenize(texts),
        #     do_sample=do_sample,
        #     top_k=top_k,
        #     temperature=temperature,
        #     num_return_sequences=num_return,
        #     return_dict_in_generate=True,
        #     output_scores=save_scores,
        #     max_length=self.max_length,
        #     **kwargs,
        # )

        if 'decoder_input_ids' in batch:
            batch['decoder_input_ids'] = batch['decoder_input_ids'].to(self.model.device)
        
        if 'desc_pos' in batch:
            outputs = self.model.generate(
                **batch,
                do_sample=do_sample,
                top_k=top_k,
                temperature=temperature,
                num_return_sequences=num_return,
                return_dict_in_generate=True,
                output_scores=save_scores,
                max_length=self.max_length,
            )
        else:
            # ipdb.set_trace()
            # print(1)
            if 'decoder_inputs_embeds' not in batch:
                outputs = self.model.generate(
                    **batch,
                    do_sample=do_sample,
                    top_k=top_k,
                    temperature=temperature,
                    num_return_sequences=num_return,
                    return_dict_in_generate=True,
                    output_scores=save_scores,
                    max_length=self.max_length,
                    # decoder_attention_mask = decoder_attention_mask,
                    # decoder_input_ids = decoder_input_ids,
                    **kwargs,
                )
            else:
                ipdb.set_trace()
                outputs = self.model.generate(
                    **batch,
                    do_sample=do_sample,
                    top_k=top_k,
                    temperature=temperature,
                    num_return_sequences=num_return,
                    return_dict_in_generate=True,
                    output_scores=save_scores,
                    max_length=self.max_length,
                    # decoder_inputs_embeds=decoder_input_ids,
                    # decoder_attention_mask = decoder_attention_mask,
                    # decoder_input_ids = decoder_input_ids,
                    **kwargs,
                )

        self.scores = None
        if save_scores:
            self.scores = [_ for _ in torch.stack(outputs.scores, 1).cpu()]
        # ipdb.set_trace()
        return self.decode(outputs.sequences)

    def decode(self, outputs) -> List[str]:
        tok = self.tokenizer
        texts = tok.batch_decode(
            outputs, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

        # Manually remove <bos><eos><pad> in case we have custom special tokens
        special_tokens = [tok.eos_token, tok.pad_token]
        for i, t in enumerate(texts):
            for token in special_tokens:
                t = t.replace(token, "")
                texts[i] = t
        return texts

class TextGeneratormulti(DynamicModel):
    
    # model: T5ForConditionalGeneration
    tokenizer: T5Tokenizer

    model: PreTrainedModel
    # tokenizer: PreTrainedTokenizerFast

    scores: Optional[List[Tensor]] = None
    max_length: int

    def tokenize(self, texts: List[str], **kwargs):
        # ipdb.set_trace()
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            **kwargs,
        ).to(self.model.device)

    def run(
        self,
        texts: List[str],
        # batch,#这里
        do_sample=True,
        top_k=50,
        temperature=1.0,
        num_return: int = 4,
        prompt: Optional[str] = None,
        # prompt = '<triplet>:',
        # prompt = ' ',
        prompt_ids: Optional[List[int]] = None,
        # prefix: Optional[List[int]] = None,
        multi_prompt_ids: Optional[List[List[int]]] = None,
        decoder_input_ids: Optional[Tensor] = None,
        # decoder_attention_mask: Optional[Tensor] = None,
        # decoder_inputs_embeds: Optional[Tensor] = None,
        save_scores: bool = False,
        **kwargs,
    ) -> List[str]:
        # https://huggingface.co/transformers/v4.7.0/main_classes/model.html#generation
        tok = self.tokenizer

        eos = tok.eos_token_id#tok.bos_token_Id
        pad = tok.pad_token_id
        bos = tok.bos_token_id
        
        # temp=tok(texts, return_tensors='pt', add_special_tokens=True)
        # input_ids, input_attn=temp['input_ids'], temp['attention_mask']
        # ipdb.set_trace()

        if prompt is not None:
            # prompt_ids = f'{prompt}, {prefix}'
            # prompt_ids = self.tokenizer(prompt_ids, add_special_tokens=False).input_ids
            prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
            # prompt_embed = self.tokenizer(prompt, return_tensors='pt', add_special_tokens=False)['input_ids']
        if prompt_ids is not None:
            prompt_ids = [eos, pad] + prompt_ids
            # prompt_ids = [eos, bos] + prompt_ids
            decoder_input_ids = torch.tensor([prompt_ids])
        if multi_prompt_ids is not None:
            assert len(texts) == len(multi_prompt_ids)
            multi_prompt_ids = [[eos] + lst for lst in multi_prompt_ids]
            # multi_prompt_ids = [[eos, bos] + lst for lst in multi_prompt_ids]
            decoder_input_ids = torch.tensor(multi_prompt_ids)
        if decoder_input_ids is not None:
            kwargs.update(decoder_input_ids=decoder_input_ids.to(self.model.device))
        #     kwargs.update(decoder_attention_mask=decoder_attention_mask.to(self.model.device))

        # if prompt_embed is not None:
        #     kwargs.update(decoder_inputs_embeds=prompt_embed.to(self.model.device))
      

        outputs = self.model.generate(
            **self.tokenize(texts),
            # **batch,
            do_sample=do_sample,
            top_k=top_k,
            temperature=temperature,
            num_return_sequences=num_return,
            return_dict_in_generate=True,
            output_scores=save_scores,
            max_length=self.max_length,
            # decoder_inputs_embeds=decoder_inputs_embeds,
            **kwargs,
        )

        self.scores = None
        if save_scores:
            self.scores = [_ for _ in torch.stack(outputs.scores, 1).cpu()]
        # ipdb.set_trace()
        return self.decode(outputs.sequences)
        #self.tokenizer.decode(outputs.sequences[0].tolist())

    def decode(self, outputs) -> List[str]:
        tok = self.tokenizer
        texts = tok.batch_decode(
            outputs, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

        # Manually remove <bos><eos><pad> in case we have custom special tokens
        special_tokens = [tok.eos_token, tok.unk_token, tok.pad_token]
        for i, t in enumerate(texts):
            for token in special_tokens:
                t = t.replace(token, "")
                texts[i] = t
        return texts

class TextGenerator_weight(DynamicModel):
    model: T5ForConditionalGenerationWithAdapter
    tokenizer: T5Tokenizer
    scores: Optional[List[Tensor]] = None
    max_length: int

    def tokenize(self, texts: List[str], **kwargs):
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            **kwargs,
        ).to(self.model.device)

    def run(
        self,
        batch,
        do_sample=True,
        top_k=50,
        temperature=1.0,
        num_return: int = 4,
        save_scores: bool = False,
        **kwargs,
    ) -> List[str]:
        # https://huggingface.co/transformers/v4.7.0/main_classes/model.html#generation
        tok = self.tokenizer
        eos = tok.eos_token_id#tok.bos_token_Id
        if 'decoder_input_ids' in batch:
            batch['decoder_input_ids'] = batch['decoder_input_ids'].to(self.model.device)
        if 'desc_pos' in batch:
            outputs = self.model.generate(
                **batch,
                do_sample=do_sample,
                top_k=top_k,
                temperature=temperature,
                num_return_sequences=num_return,
                return_dict_in_generate=True,
                output_scores=save_scores,
                max_length=self.max_length,
            )
        else:
            outputs = self.model.generate(
                **batch,
                do_sample=do_sample,
                top_k=top_k,
                temperature=temperature,
                num_return_sequences=num_return,
                return_dict_in_generate=True,
                output_scores=save_scores,
                max_length=self.max_length,
                **kwargs,
            )

        self.scores = None
        if save_scores:
            self.scores = [_ for _ in torch.stack(outputs.scores, 1).cpu()]
        return self.decode(outputs.sequences)

    def decode(self, outputs) -> List[str]:
        tok = self.tokenizer
        texts = tok.batch_decode(
            outputs, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

        # Manually remove <bos><eos><pad> in case we have custom special tokens
        special_tokens = [tok.eos_token, tok.pad_token]
        for i, t in enumerate(texts):
            for token in special_tokens:
                t = t.replace(token, "")
                texts[i] = t
        return texts
    
class LabelConstraint:
    def __init__(
        self,
        labels: List[str],
        tokenizer,
        prefix: str = " [REL] ",
    ):
        self.prefix: List[int] = tokenizer(prefix, add_special_tokens=False).input_ids
        self.label_map: Dict[int, str] = {
            tokenizer(" " + x, add_special_tokens=False).input_ids[0]: x for x in labels
        }
        self.tokenizer = tokenizer

    def run(self, triplet, scores):

        triplet = triplet.copy(deep=True)
        assert scores.ndim == 2
        token_ids = scores.argmax(dim=-1).int().tolist()
        i = find_sublist_index(token_ids, self.prefix)
        if i == -1:
            return triplet

        position = i + len(self.prefix)
        best = ""
        best_score = -1e9
        for j, label in self.label_map.items():
            score = scores[position, j].item()
            if score > best_score:
                best = label
                best_score = score

        if triplet.label in self.label_map.values():
            assert best == triplet.label

        assert len(best) > 0
        triplet.label = best
        triplet.score = best_score
        return triplet

class LabelConstraint_format:
    def __init__(
        self,
        labels: List[str],
        tokenizer,
        prefix: str = " [REL] : ",
        # raw_data, 
    ):  
        # self.raw_data = raw_data
        # self.raw_pos_data = raw_data.split(":")[0].strip()
        # self.prefix: List[int] = tokenizer(self.raw_pos_data, add_special_tokens=False).input_ids
        self.prefix: List[int] = tokenizer(prefix, add_special_tokens=False).input_ids
        self.label_map: Dict[int, str] = {
            tokenizer(" " + x, add_special_tokens=False).input_ids[0]: x for x in labels
        }
        self.tokenizer = tokenizer

    def run(self, triplet, scores, rel_dis):
        # ipdb.set_trace()

        triplet_dup = triplet.copy(deep=True)
        assert scores.ndim == 2
        token_ids = scores.argmax(dim=-1).int().tolist()        
        i = find_sublist_index(token_ids, self.prefix)
        if i == -1:
            return triplet_dup

        position = i + len(self.prefix)
        # position = 0
        best = ""
        best_score = -1e9
        for j, label in self.label_map.items():
            score = scores[position, j].item()
            if score > best_score:
                best = label
                best_score = score

        if triplet_dup.label in self.label_map.values():
            assert best == triplet_dup.label

        assert len(best) > 0
        
        triplet_dup.label = best
        triplet_dup.score = best_score
        triplet_dup.label_id = self.find_key_by_word(triplet_dup.label, rel_dis)
        # ipdb.set_trace()
        return triplet_dup

    def find_key_by_word(self, word, dictionary):
        for key, values in dictionary.items():
            # ipdb.set_trace()
            if word == values[0]:
                return key
        return "P999"

class LabelConstraint_multi:
    def __init__(
        self,
        labels: List[str],
        tokenizer,
        prefix: str = " [REL] : ",
        # raw_data, 
    ):  
        # self.raw_data = raw_data
        # self.raw_pos_data = raw_data.split(":")[0].strip()
        # self.prefix: List[int] = tokenizer(self.raw_pos_data, add_special_tokens=False).input_ids
        self.prefix: List[int] = tokenizer(prefix, add_special_tokens=False).input_ids
        self.label_map: Dict[int, str] = {
            tokenizer(" " + x, add_special_tokens=False).input_ids[0]: x for x in labels
        }
        self.tokenizer = tokenizer

    def run(self, triplet, scores, rel_dis):
        # ipdb.set_trace()

        score_relname={}

        triplet_dup = triplet.copy(deep=True)
        assert scores.ndim == 2
        token_ids = scores.argmax(dim=-1).int().tolist()        
        i = find_sublist_index(token_ids, self.prefix)
        if i == -1:
            return triplet_dup

        position = i + len(self.prefix)
        # position = 0
        best = ""
        best_score = -1e9
        
        for j, label in self.label_map.items():
            score = scores[position, j].item()
            score_relname[score] = label
            # if score > best_score:
            #     best = label
            #     best_score = score
        top_key = sorted(score_relname.keys())
        max_two_keys = top_key[-2:]    

        # if triplet_dup.label in self.label_map.values():
        #     assert best == triplet_dup.label

        # assert len(best) > 0
        
        ipdb.set_trace()
        triplet_dup.label = best
        triplet_dup.score = best_score
        triplet_dup.label_id = self.find_key_by_word(triplet_dup.label, rel_dis)
        # ipdb.set_trace()
        return triplet_dup

    def find_key_by_word(self, word, dictionary):
        for key, values in dictionary.items():
            # ipdb.set_trace()
            if word == values[0]:
                return key
        return "P999"

class TripletSearchDecoder(DynamicModel):
    gen: TextGeneratormulti
    constraint: LabelConstraint
    encoder: ExtractEncoder_plus
    top_k: int = 4

    # prompt: str = ''
    # tokenizer: T5Tokenizer
    relation_discribe: json
    # prefix: str = ''
    


    def generate(self, text: str, **kwargs) -> Tuple[str, Tensor]:

        # ipdb.set_trace()
        outputs = self.gen.run(
            [text],
            # self.batch,
            do_sample=False,
            num_return=1,
            num_beams=1,
            save_scores=True,
            **kwargs,
        )

        assert len(outputs) == 1
        assert self.gen.scores is not None
        scores = torch.log_softmax(self.gen.scores[0], dim=-1)
        assert scores.ndim == 2
        return outputs[0], scores

    def find_prefix_end(self, token_ids: List[str], prefix: str) -> int:
        prefix_ids = self.gen.tokenizer(prefix, add_special_tokens=False).input_ids
        i = find_sublist_index(token_ids, prefix_ids)

        # if i != -1:
        #     position = i + len(prefix_ids)
        # else:
        #     position = 0
        # return position
        position = i + len(prefix_ids)
        return position

    def branch(
        self, text: str, prefix: str, prompt: Optional[str] = None, **kwargs,#prompt: str = ' ',  #这里
    ) -> List[Tuple[str, float]]:

        # self.tokenizer.decode(token_ids)
        # ipdb.set_trace()
        # _, scores = self.generate(text, prompt=prompt, prefix=prefix, **kwargs)
        _, scores = self.generate(text, prompt=prompt, **kwargs)
        token_ids = scores.argmax(dim=-1).int().tolist()
        i = self.find_prefix_end(token_ids, prefix)

        # raw_text_ids = self.tokenizer(_, add_special_tokens=False).input_ids
        # target_num = self.find_prefix_end(raw_text_ids, prefix)
        # pretext=raw_text_ids[:target_num]
        
        # ipdb.set_trace()
        pairs = []
        for j in torch.argsort(scores[i])[-self.top_k :]:
            #self.tokenizer.decode(torch.argsort(scores[i])[-self.top_k :])
            # p = (prompt or "") + self.gen.decode([token_ids[:i] + [j]])[0]
            # if prompt != None:
            #     p = self.gen.decode([pretext + token_ids[:i] + [j]])[0]
            # else:
            #     p = self.gen.decode([token_ids[:i] + [j]])[0]
            p = (prompt or "") + self.gen.decode([token_ids[:i] + [j]])[0]
            pairs.append((p, scores[i, j].item()))

        return pairs

    def run(self, text: str, triplet) -> List[RelationSentence]:
        # ipdb.set_trace()
        dis_id = triplet[0].label_id
        dis_name = [self.relation_discribe[dis_id][0]]
        discribe = [self.relation_discribe[dis_id][2]]
        x = f'<Task2>, Extract <triplet> in [sent] by [Relation_discribe]. [Relation_discribe]: {dis_name[0]}, {discribe[0]}. {self.encoder.encode_x(text)}'
        # x = self.prompt + self.encoder.encode_x(text)
        outputs = []
        # ipdb.set_trace()
        # self.model.eval()
        for prompt_a, score_a in self.branch(x, prefix="[HEAD]"): #Head Entity : / <triplet> : [HEAD]
            for prompt_b, score_b in self.branch(
                x, prefix=" [TAIL]", prompt=prompt_a    # Tail Entity :
            ):
                # ipdb.set_trace()
                output, scores = self.generate(x, prompt=prompt_b)
                # output, scores = self.generate(x, prefix="[REL]", prompt=prompt_b)

                token_ids = token_ids = scores.argmax(dim=-1).int().tolist()
                i = self.find_prefix_end(token_ids, prefix="[REL]")    #Relation :

                # output = output.split('[REL]').strip(',')
                # i = 0

                score_c = max(scores[i].tolist())
                s = self.encoder.safe_decode_raw(x=x, y=output)
                s = self.constraint.run(s, scores)
                # score_c = s.score  # From LabelConstraint
                s.score = (score_a + score_b + score_c) / 3
                outputs.append(s)

        return outputs

class test:
    def __init__(self, opt, model, tokenizer, encoder, d_model=768):
        self.opt = opt
        self.model = model
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.raw_relation_label:list
        if torch.cuda.is_available():
            self.model.cuda()
                
        with open("/root/autodl-tmp/Meta/data/REL_discribe.jsonl","r") as f1:
            contents = f1.read()
            json_object = json.loads(contents)
            self.relation_discribe = json_object

    def predict_withoutconstrain(self, path_in, model_path, use_label_constraint=True, max_target_length=128, device=torch.device("cuda")):
        data = Dataset.load(path_in)
        num_pred = 0

        pred_labels = data.get_labels()
        texts = [trip.text for sent in data.sents for trip in sent.triplets]
        test_data = self.process_Baseline_data(data, pred_labels, self.relation_discribe)
        # ipdb.set_trace()
        self.test_dataloader = DataLoader(test_data,
                                           batch_size=1,
                                        #    batch_size=self.opt.batch_size,
                                           drop_last=False,
                                           shuffle=False,
                                           collate_fn=collate_fn_Baseline)
        gen = TextGenerator(
            model=T5ForConditionalGeneration.from_pretrained(model_path),
            tokenizer=T5Tokenizer.from_pretrained(model_path),
            max_length=max_target_length,
        )
        gen.model = gen.model.to(device)
        # constraint = LabelConstraint(labels=pred_labels, tokenizer=self.tokenizer)
        sents, sents_no = [], 0
        for batch in tqdm(self.test_dataloader):
            del batch['labels']
            batch = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}
            outputs = gen.run(
                batch, num_beams=1, save_scores=use_label_constraint, do_sample=False, num_return=1,
            )

            for j, raw in enumerate(outputs):
                triplet = self.encoder.safe_decode(texts[sents_no], y=raw)
                # ipdb.set_trace()
                print(triplet.raw)

                num_trip = 0
                # for i in range(len(data.sents)):
                    # num_pred += len(data.sents[i].triplets)
                for p in data.sents[num_pred].triplets:
                    print(' [HEAD] ',end="")
                    for l in range(len(p.head)):
                        print(p.tokens[p.head[0]+l]+' ',end="")
                    print(', ',end="")
                    print('[TAIL] ',end="")
                    for l in range(len(p.tail)):
                        print(p.tokens[p.tail[0]+l]+' ',end="")
                    print(', ',end="")
                    print('[REL] '+p.label+' .')
                num_pred+=1
                # sents_no+=1
                    # num_trip += len(data.sents[i].triplets)
                # if use_label_constraint:
                #     assert gen.scores is not None
                #     triplet = constraint.run(triplet, gen.scores[j])
                # sents.append(Sentence(triplets=[triplet]))
        print(num_pred)
        # Dataset(sents=sents).save(path_out)

    def predict_withconstrain(self, path_in, model_path, pid2name='', use_label_constraint=True, max_target_length=128, device=torch.device("cuda")):
        data = Dataset.load(path_in)
        pred_labels = data.get_labels()
        texts = [trip.text for sent in data.sents for trip in sent.triplets]
        test_data = self.process_Baseline_data(data, pred_labels, self.relation_discribe)
        self.test_dataloader = DataLoader(test_data,
                                           batch_size=self.opt.batch_size,
                                           num_workers=2,
                                           drop_last=False,
                                           shuffle=False,
                                           collate_fn=collate_fn_Baseline)
        gen = TextGenerator(
            model=T5ForConditionalGeneration.from_pretrained(model_path),
            tokenizer=T5Tokenizer.from_pretrained(model_path),
            max_length=max_target_length,
        )
        gen.model = gen.model.to(device)
        constraint = LabelConstraint(labels=pred_labels, tokenizer=self.tokenizer)
        sents, sents_no = [], 0

        # ipdb.set_trace()

        for batch in tqdm(self.test_dataloader):
            del batch['labels']
            batch = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}
            outputs = gen.run(
                batch, num_beams=1, save_scores=use_label_constraint, do_sample=False, num_return=1,
            )

            for j, raw in enumerate(outputs):
                triplet = self.encoder.safe_decode(texts[sents_no], y=raw)
                sents_no+=1
                if use_label_constraint:
                    assert gen.scores is not None
                    triplet = constraint.run(triplet, gen.scores[j])
                sents.append(Sentence(triplets=[triplet]))
        # Dataset(sents=sents).save(path_out)
    


    def Recall_predict(self, path_in, path_out, model_path, pid2name='', use_label_constraint=True, max_target_length=128, device=torch.device("cuda")):

        data = Dataset.load(path_in)
        pred_labels = data.get_labels()
        triple_list = []
        triple_list = [trip for sent in data.sents for trip in sent.triplets]    
        texts = [trip.text for sent in data.sents for trip in sent.triplets]#所有句子构成的列表
        # ipdb.set_trace()
        test_data = self.process_Recall_data(data, pred_labels, self.relation_discribe)
        self.test_dataloader = DataLoader(test_data,
                                        #    batch_size=self.opt.batch_size,
                                           batch_size=1,
                                           num_workers=2,
                                           drop_last=False,
                                           shuffle=False,
                                           collate_fn=collate_fn_Recall_plus)
        gen = TextGenerator(
            model=T5ForConditionalGeneration.from_pretrained(model_path),
            tokenizer=T5Tokenizer.from_pretrained(model_path),
            max_length=max_target_length,
        )
        gen.model = gen.model.to(device)
        constraint = LabelConstraint_format(labels=pred_labels, tokenizer=self.tokenizer,)
        sents_no = 0
        sents = []
        output_data = []
        for batch in tqdm(self.test_dataloader):
            del batch['labels']#
            # ipdb.set_trace()
            batch = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}
            outputs = gen.run(
                batch, num_beams=1, save_scores=use_label_constraint, do_sample=False, num_return=1,
            )
            
            for j, raw in enumerate(outputs):
                # ipdb.set_trace()
                try:
                    rel = raw.strip().split(":")[0].strip()
                except Exception as e:
                    rel = ' '
                # triple = raw.split(" <triplet>: ")[1]
                # triple = self.encoder.encode_output_rel(self.encoder.decode_from_line(triple_list[sents_no]), rel)
                triple = self.encoder.encode_output_rel(triple_list[sents_no], rel)
                label_id = triple_list[sents_no].label_id
                triplet = self.encoder.safe_decode(texts[sents_no], y=triple, label_id=label_id)
                # ipdb.set_trace()
                sents_no+=1
                if use_label_constraint:
                    assert gen.scores is not None
                    triplet = constraint.run(triplet, gen.scores[j], self.relation_discribe)
                output_data.append(Sentence(triplets=[triplet]))
        trans_data = Dataset(sents = output_data)
        # ipdb.set_trace()
        test.Recall_predict_withconstrain(self, data = trans_data, gen_model = gen, pred_labels = pred_labels, texts = texts, sents = sents)
                # sents.append(Sentence(triplets=[triplet]))
        Dataset(sents=sents).save(path_out)

    def Recall_predict_withoutconstrain(self, path_in, data, gen_model, pred_labels, texts, sents, pid2name='', use_label_constraint=True, max_target_length=128, device=torch.device("cuda")):
        
        path_in = Dataset.load(path_in)
        # ipdb.set_trace()
        # texts = [trip.text for sent in data.sents for trip in sent.triplets]
        test_data = self.process_Recall_data_decodeinput(data, pred_labels, self.relation_discribe)
        self.test_dataloader = DataLoader(test_data,
                                           batch_size=1,
                                        #    batch_size=self.opt.batch_size,
                                           num_workers=2,
                                           drop_last=False,
                                           shuffle=False,
                                           collate_fn=collate_fn_Recall_help_plus)
        gen_model.model = gen_model.model.to(device)
        # ipdb.set_trace()
        constraint = LabelConstraint(labels=pred_labels, tokenizer=self.tokenizer)
        sents, sents_no = [], 0
        num=0
        for batch in self.test_dataloader:
            # del batch['labels']
            # self.tokenizer.decode(batch['decoder_input_ids'][0].tolist()) 
            batch = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}
            outputs = gen_model.run(
                batch, num_beams=1, save_scores=use_label_constraint, do_sample=False, num_return=1,
            )

            for j, raw in enumerate(outputs):
                raw_trip = raw.split(" <triplet>")[1].split(": ")[1].strip()
                # ipdb.set_trace()
                triplet = self.encoder.safe_decode_raw(texts[sents_no], y=raw_trip)
                
                print(triplet.raw)
                # constraint.run(triplet, gen_model.scores[j])
                num_trip = 0
                # for i in range(len(data.sents)):
                    # num_pred += len(data.sents[i].triplets)
                for pl in path_in.sents[num].triplets:
                    # ipdb.set_trace()
                    print(' [HEAD] ',end="")
                    for l in range(len(pl.head)):
                        print(pl.tokens[pl.head[0]+l]+' ',end="")
                    print(', ',end="")
                    print('[TAIL] ',end="")
                    for l in range(len(pl.tail)):
                        print(pl.tokens[pl.tail[0]+l]+' ',end="")
                    print(', ',end="")
                    print('[REL] '+pl.label+' .')
            num+=1


    def Recall_predict_withconstrain(self, data, gen_model, pred_labels, texts, sents, pid2name='', use_label_constraint=True, max_target_length=128, device=torch.device("cuda")):
        
        # texts = [trip.text for sent in data.sents for trip in sent.triplets]
        test_data = self.process_Recall_data_decodeinput(data, pred_labels, self.relation_discribe)
        self.test_dataloader = DataLoader(test_data,
                                           batch_size=1,
                                        #    batch_size=self.opt.batch_size,
                                           num_workers=2,
                                           drop_last=False,
                                           shuffle=False,
                                           collate_fn=collate_fn_Recall_help_plus)
        gen_model.model = gen_model.model.to(device)
        # ipdb.set_trace()
        constraint = LabelConstraint(labels=pred_labels, tokenizer=self.tokenizer)
        sents_no = 0
        for batch in self.test_dataloader:
            # ipdb.set_trace()
            # del batch['labels']
            # self.tokenizer.decode(batch['decoder_input_ids'][0].tolist()) 
            batch = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}
            outputs = gen_model.run(
                batch, num_beams=1,  save_scores=use_label_constraint, do_sample=False, num_return=1,
            )# decoder_input_ids = batch['decoder_input_ids'],
            # outputs = gen_model.run(
            #     batch, num_beams=1, save_scores=use_label_constraint, do_sample=False, num_return=1,
            # ) decoder_attention_mask = batch['decoder_attention_mask'], decoder_input_ids = batch['decoder_input_ids'],

            for j, raw in enumerate(outputs):
                # ipdb.set_trace()
                try:
                    raw_trip = raw.split(" <triplet>")[1].split(": ")[1].strip()
                except Exception as e:
                    raw_trip =  "[HEAD] [TAIL] [REL]."
                triplet = self.encoder.safe_decode_raw(texts[sents_no], y=raw_trip)
                sents_no+=1

                # constraint.run(triplet, gen_model.scores[j])
                if use_label_constraint:
                    assert gen_model.scores is not None
                    triplet = constraint.run(triplet, gen_model.scores[j])
                sents.append(Sentence(triplets=[triplet]))

        # Dataset(sents=sents).save(path_out)

    def Pipeline_predict_withconstrain_ActiveMeta(self, data, gen_model, pred_labels, texts, sents, pid2name='', use_label_constraint=True, max_target_length=128, device=torch.device("cuda")):
        
        # ipdb.set_trace()
        # texts = [trip.text for sent in data.sents for trip in sent.triplets]
        test_data = self.process_Pipeline_data(data, pred_labels, self.relation_discribe)
        self.test_dataloader = DataLoader(test_data,
                                           batch_size=1,
                                        #    batch_size=self.opt.batch_size,
                                           num_workers=2,
                                           drop_last=False,
                                           shuffle=False,
                                           collate_fn=collate_fn_Pipeline_devdata)
        gen_model.model = gen_model.model.to(device)
        # ipdb.set_trace()
        constraint = LabelConstraint(labels=pred_labels, tokenizer=self.tokenizer)
        sents_no = 0
        for batch in tqdm(self.test_dataloader):
            # ipdb.set_trace()
            del batch['labels']
            # self.tokenizer.decode(batch['decoder_input_ids'][0].tolist()) 
            batch = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}
            outputs = gen_model.run(
                batch, num_beams=1,  save_scores=use_label_constraint, do_sample=False, num_return=1,
            )# decoder_input_ids = batch['decoder_input_ids'],
            # outputs = gen_model.run(
            #     batch, num_beams=1, save_scores=use_label_constraint, do_sample=False, num_return=1,
            # ) decoder_attention_mask = batch['decoder_attention_mask'], decoder_input_ids = batch['decoder_input_ids'],

            for j, raw in enumerate(outputs):
                # ipdb.set_trace()
                triplet = self.encoder.safe_decode_raw(texts[sents_no], y=raw)
                sents_no+=1

                # constraint.run(triplet, gen_model.scores[j])
                if use_label_constraint:
                    assert gen_model.scores is not None
                    triplet = constraint.run(triplet, gen_model.scores[j])
                sents.append(Sentence(triplets=[triplet]))

    def Pipeline_predict(self, path_in, path_out, model_path, task_type, gold_spilit_path, pid2name='', use_label_constraint=True, max_target_length=128, device=torch.device("cuda")):

        data = Dataset.load(path_in)
        self.raw_relation_label = data.get_labels()
        
        if task_type=="multi":
            data.sents = [s for s in data.sents if len(s.triplets) > 1]  
        else:
            data.sents = [s for s in data.sents if len(s.triplets) == 1]
        data.save(gold_spilit_path)
        pred_labels = data.get_labels()
        # ipdb.set_trace()
        triple_list = []
        if task_type=="singal":
            triple_list = [trip for sent in data.sents for trip in sent.triplets]    
            texts = [trip.text for sent in data.sents for trip in sent.triplets]#所有句子构成的列表
        else:
            triple_list = [sent.triplets[0] for sent in data.sents]    
            texts = [sent.triplets[0].text for sent in data.sents]
        # ipdb.set_trace()
        test_data = self.process_Pipeline_data(data, self.raw_relation_label, self.relation_discribe) #pred_labels
        self.test_dataloader = DataLoader(test_data,
                                        #    batch_size=self.opt.batch_size,
                                           batch_size=1,
                                           num_workers=2,
                                           drop_last=False,
                                           shuffle=False,
                                           collate_fn=collate_fn_Pipeline_traindata)
        gen = TextGenerator(
            model=T5ForConditionalGeneration.from_pretrained(model_path),
            tokenizer=T5Tokenizer.from_pretrained(model_path),
            max_length=max_target_length,
        )
        gen.model = gen.model.to(device)
        constraint = LabelConstraint_format(labels=pred_labels, tokenizer=self.tokenizer,)#这里
        sents_no = 0
        sents = []
        output_data = []
        for batch in tqdm(self.test_dataloader):
            # ipdb.set_trace()
            del batch['labels']#
            # self.tokenizer.decode(batch['labels'][0].tolist())
            batch = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}
            query_dict = {'help_input_ids':'input_ids', 'help_attention_mask':'attention_mask', 'help_decoder_attention_mask':'decoder_attention_mask', 'help_labels':'labels'}                
            batch1 = {query_dict[k]: v for k, v in batch.items() if k in query_dict}
            batch2 = {k: v for k, v in batch.items() if k not in query_dict}
            # ipdb.set_trace()           
            outputs = gen.run(
                batch2, num_beams=1, save_scores=use_label_constraint, do_sample=False, num_return=1,
            )
            
            for j, raw in enumerate(outputs):
                try:
                    rel = raw.strip().split(':')[1].strip()
                except Exception as e:
                    rel = ' '
                # triple = raw.split(" <triplet>: ")[1]
                # triple = self.encoder.encode_output_rel(self.encoder.decode_from_line(triple_list[sents_no]), rel)
                triple = self.encoder.encode_output_rel(triple_list[sents_no], rel)
                label_id = triple_list[sents_no].label_id
                triplet = self.encoder.safe_decode(texts[sents_no], y=triple, label_id=label_id)
                sents_no+=1
                if use_label_constraint:
                    assert gen.scores is not None
                    triplet = constraint.run(triplet, gen.scores[j], self.relation_discribe)
                output_data.append(Sentence(triplets=[triplet]))
        trans_data = Dataset(sents = output_data)
        # ipdb.set_trace()
        if task_type=="singal":
            test.Pipeline_predict_withconstrain(self, data = trans_data, gen_model = gen, pred_labels = pred_labels, texts = texts, sents = sents)
            Dataset(sents=sents).save(path_out)#这里
        else:
            test.predict_multi(self, data = trans_data, pred_labels = pred_labels, texts = texts, sents = sents, model_path=model_path, path_out=path_out)


    def Pipeline_predict_withconstrain(self, data, gen_model, pred_labels, texts, sents, pid2name='', use_label_constraint=True, max_target_length=128, device=torch.device("cuda")):
        
        # ipdb.set_trace()
        # texts = [trip.text for sent in data.sents for trip in sent.triplets]
        test_data = self.process_Pipeline_data(data, pred_labels, self.relation_discribe)
        self.test_dataloader = DataLoader(test_data,
                                           batch_size=1,
                                        #    batch_size=self.opt.batch_size,
                                           num_workers=2,
                                           drop_last=False,
                                           shuffle=False,
                                           collate_fn=collate_fn_Pipeline_devdata)
        gen_model.model = gen_model.model.to(device)
        # ipdb.set_trace()
        constraint = LabelConstraint(labels=pred_labels, tokenizer=self.tokenizer)
        sents_no = 0
        for batch in tqdm(self.test_dataloader):
            # ipdb.set_trace()
            del batch['labels']
            # self.tokenizer.decode(batch['decoder_input_ids'][0].tolist()) 
            batch = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}
            outputs = gen_model.run(
                batch, num_beams=1,  save_scores=use_label_constraint, do_sample=False, num_return=1,
            )# decoder_input_ids = batch['decoder_input_ids'],
            # outputs = gen_model.run(
            #     batch, num_beams=1, save_scores=use_label_constraint, do_sample=False, num_return=1,
            # ) decoder_attention_mask = batch['decoder_attention_mask'], decoder_input_ids = batch['decoder_input_ids'],

            for j, raw in enumerate(outputs):
                # ipdb.set_trace()
                triplet = self.encoder.safe_decode_raw(texts[sents_no], y=raw)
                sents_no+=1

                # constraint.run(triplet, gen_model.scores[j])
                if use_label_constraint:
                    assert gen_model.scores is not None
                    triplet = constraint.run(triplet, gen_model.scores[j])
                sents.append(Sentence(triplets=[triplet]))

    def Pipeline_predict_ActiveMeta(self, path_in, path_out, model_path, task_type, gold_spilit_path, pid2name='', use_label_constraint=True, max_target_length=128, device=torch.device("cuda")):

        data = Dataset.load(path_in)
        self.raw_relation_label = data.get_labels()
        
        if task_type=="multi":
            data.sents = [s for s in data.sents if len(s.triplets) > 1]
            data.save(gold_spilit_path)

        pred_labels = data.get_labels()

        triple_list = []
        triple_list = [trip for sent in data.sents for trip in sent.triplets]    
        texts = [trip.text for sent in data.sents for trip in sent.triplets]#所有句子构成的列表
        # ipdb.set_trace()
        test_data = self.process_Pipeline_data(data, self.raw_relation_label, self.relation_discribe) #pred_labels
        self.test_dataloader = DataLoader(test_data,
                                        #    batch_size=self.opt.batch_size,
                                           batch_size=1,
                                           num_workers=2,
                                           drop_last=False,
                                           shuffle=False,
                                           collate_fn=collate_fn_Pipeline_traindata)
        gen = TextGenerator(
            model=T5ForConditionalGeneration.from_pretrained(model_path),
            tokenizer=T5Tokenizer.from_pretrained(model_path),
            max_length=max_target_length,
        )
        gen.model = gen.model.to(device)
        constraint = LabelConstraint_format(labels=pred_labels, tokenizer=self.tokenizer,)#这里
        sents_no = 0
        sents = []
        output_data = []
        for batch in tqdm(self.test_dataloader):
            # ipdb.set_trace()
            del batch['labels']#
            # self.tokenizer.decode(batch['labels'][0].tolist())
            batch = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}
            query_dict = {'help_input_ids':'input_ids', 'help_attention_mask':'attention_mask', 'help_decoder_attention_mask':'decoder_attention_mask', 'help_labels':'labels'}                
            batch1 = {query_dict[k]: v for k, v in batch.items() if k in query_dict}
            batch2 = {k: v for k, v in batch.items() if k not in query_dict}
            # ipdb.set_trace()           
            outputs = gen.run(
                batch2, num_beams=1, save_scores=use_label_constraint, do_sample=False, num_return=1,
            )
            
            for j, raw in enumerate(outputs):
                try:
                    rel = raw.strip().split(':')[1].strip()
                except Exception as e:
                    rel = ' '
                # triple = raw.split(" <triplet>: ")[1]
                # triple = self.encoder.encode_output_rel(self.encoder.decode_from_line(triple_list[sents_no]), rel)
                triple = self.encoder.encode_output_rel(triple_list[sents_no], rel)
                label_id = triple_list[sents_no].label_id
                triplet = self.encoder.safe_decode(texts[sents_no], y=triple, label_id=label_id)
                sents_no+=1
                if use_label_constraint:
                    assert gen.scores is not None
                    triplet = constraint.run(triplet, gen.scores[j], self.relation_discribe)
                output_data.append(Sentence(triplets=[triplet]))
        trans_data = Dataset(sents = output_data)
        # ipdb.set_trace()
        test.Pipeline_predict_withconstrain_ActiveMeta(self, data = trans_data, gen_model = gen, pred_labels = pred_labels, texts = texts, sents = sents)
        # test.predict_multi(self, data = trans_data, pred_labels = pred_labels, texts = texts, sents = sents, model_path=model_path, path_out=path_out)
        Dataset(sents=sents).save(path_out)#这里



    def predict_multi(self, data, pred_labels, texts, sents, model_path, path_out, use_label_constraint=True, max_target_length=128, search_threshold=-0.9906, device=torch.device("cuda")):
        # data.sents = [s for s in data.sents if len(s.triplets) > 1]
        # split_data_path = '/user_data/wujy/SimonHeye/META/outputs/bsz-1_ep-1_noreptile/fewrel/noactive/1712383059.494721/unseen_5_seed_0/split_data.json'
        # data.save(split_data_path)
        stem = Path(path_out).stem
        path_raw = path_out.replace(stem, f"{stem}_raw")

        gen_model = TextGeneratormulti(
            model=T5ForConditionalGeneration.from_pretrained(model_path),
            tokenizer=T5Tokenizer.from_pretrained(model_path),
            max_length=max_target_length,
        )        
        # gen_model.model = gen_model.model.to(device)        
        # prompt = '[PROTO] ' + ', '.join(pred_labels) + '. '
        # prompt=''
        constraint = LabelConstraint(labels=pred_labels, tokenizer=self.tokenizer)
        searcher = TripletSearchDecoder(
            gen=gen_model, encoder=self.encoder, constraint=constraint, relation_discribe=self.relation_discribe, tokenizer=self.tokenizer
        )

        sents = [
            Sentence(tokens=s.tokens, triplets=searcher.run(s.text, s.triplets))
            for s in tqdm(data.sents)
        ]

        Dataset(sents=sents).save(path_raw)
        for s in sents:
            s.triplets = [t for t in s.triplets if t.score > search_threshold]
        Dataset(sents=sents).save(path_out)













    def process_Baseline_data(self, dataset, labels_set, relation_discribe):
        
        data = []
        for sent in dataset.sents:
            sents = [self.encoder.encode_to_line(trip) for trip in sent.triplets]# sent : Sentence(triplets=[RelationSentence(tokens=['In', '1689'...], head=[11, 12], tail=[16], label='after a work by', head_id='', tail_id='', label_id='P140', error='', raw='', keyword=[''], score=0.0, zerorc_included=True)])
            #sents = ['{"text": "[SENT] : In 1689 , Konstanty was one of the judges who sentenced Kazimierz \\u0141yszczy\\u0144ski to death for atheism .", "summary": "[HEAD] Kazimierz \\u0141yszczy\\u0144ski , [TAIL] atheism , [REL] after a work by ."}\n']
            trips = [trip for trip in sent.triplets]
            #trips = [RelationSentence(tokens=['In', '1689', ',', 'Konstanty', 'was', 'one', 'of', 'the', 'judges', 'who', 'sentenced', 'Kazimierz', 'Łyszczyński', 'to', 'death', 'for', 'atheism', '.'], head=[11, 12], tail=[16], label='after a work by', head_id='', tail_id='', label_id='P140', error='', raw='', keyword=[''], score=0.0, zerorc_included=True)]
            for sent, trip in zip(sents, trips):
                temp_data = {}
                temp_train = {}
                
                context, labels = self.encoder.parse_line(sent)# content:'[SENT] : Merpati flight 106 departed Jakarta ( CGK ) on a domestic flight to Tanjung Pandan ( TJQ ) .'   labels:'[HEAD] Kazimierz Łyszczyński , [TAIL] atheism , [REL] after a work by .'
                # ipdb.set_trace()
                temp_train['input_ids'], temp_train['attention_mask'] = [], []
                temp_train['labels'], temp_train['decoder_attention_mask'] = [], []

                temp1 = self.tokenizer(context, return_tensors='pt', add_special_tokens=True)
                temp_train['input_ids'], temp_train['attention_mask'] = temp1['input_ids'], temp1['attention_mask']

                temp2 = self.tokenizer(labels, return_tensors='pt', add_special_tokens=True)
                temp_train['decoder_input_ids'], temp_train['decoder_attention_mask'] = temp2['input_ids'], temp2['attention_mask']
                
                dis_id = trip.label_id
                discribe = [relation_discribe[dis_id][1]]
                fin_discribe = '[relation_discribe]' + ': '.join(discribe)+'. '

                # ipdb.set_trace()
                temp3 = self.tokenizer(fin_discribe, return_tensors='pt', add_special_tokens=True)
                dis_input_ids, dis_attn = temp3['input_ids'], temp3['attention_mask']
                
                temp_data['sent'] = temp_train
                temp_data['dis_input_ids'] = dis_input_ids
                temp_data['dis_attn'] = dis_attn

                data.append(temp_data)
        return data            

    def process_Recall_data(self, dataset, labels_set, relation_discribe):
        
        data = []
        for sent in dataset.sents:
            sents = [self.encoder.encode_to_line(trip) for trip in sent.triplets]# sent : Sentence(triplets=[RelationSentence(tokens=['In', '1689'...], head=[11, 12], tail=[16], label='after a work by', head_id='', tail_id='', label_id='P140', error='', raw='', keyword=[''], score=0.0, zerorc_included=True)])
            #sents = ['{"text": "[SENT] : In 1689 , Konstanty was one of the judges who sentenced Kazimierz \\u0141yszczy\\u0144ski to death for atheism .", "summary": "[HEAD] Kazimierz \\u0141yszczy\\u0144ski , [TAIL] atheism , [REL] after a work by ."}\n']
            trips = [trip for trip in sent.triplets]
            #trips = [RelationSentence(tokens=['In', '1689', ',', 'Konstanty', 'was', 'one', 'of', 'the', 'judges', 'who', 'sentenced', 'Kazimierz', 'Łyszczyński', 'to', 'death', 'for', 'atheism', '.'], head=[11, 12], tail=[16], label='after a work by', head_id='', tail_id='', label_id='P140', error='', raw='', keyword=[''], score=0.0, zerorc_included=True)]
            for sent, trip in zip(sents, trips):
                temp_data = {}
                temp_train = {}
                temp_prompt = []

                context, labels = self.encoder.parse_line(sent)# content:'[SENT] : Merpati flight 106 departed Jakarta ( CGK ) on a domestic flight to Tanjung Pandan ( TJQ ) .'   labels:'[HEAD] Kazimierz Łyszczyński , [TAIL] atheism , [REL] after a work by .'
                # ipdb.set_trace()
                temp_train['input_ids'], temp_train['attention_mask'] = [], []
                temp_train['labels'], temp_train['decoder_attention_mask'] = [], []

                temp1 = self.tokenizer(context, return_tensors='pt', add_special_tokens=True)
                temp_train['input_ids'], temp_train['attention_mask'] = temp1['input_ids'], temp1['attention_mask']

                # temp2 = self.tokenizer(labels, return_tensors='pt', add_special_tokens=True)
                # temp_train['decoder_input_ids'], temp_train['decoder_attention_mask'] = temp2['input_ids'], temp2['attention_mask']
                
                dis_id = trip.label_id
                dis_name = [relation_discribe[dis_id][0]]
                discribe = [relation_discribe[dis_id][2]] #这里
                # fin_discribe = dis_name + ': '.join(discribe)+'. '+'<triplet>: '
                fin_discribe = f'{dis_name[0]}:{discribe[0]}. <triplet>:'

                # ipdb.set_trace()
                # temp3 = self.tokenizer(fin_discribe, return_tensors='pt', add_special_tokens=True)
                # dis_input_ids, dis_attn = temp3['input_ids'], temp3['attention_mask']
                
                fin_decode_inputs = fin_discribe + labels
                temp2 = self.tokenizer(fin_decode_inputs, return_tensors='pt', add_special_tokens=True)
                dis_input_ids, dis_attn = temp2['input_ids'], temp2['attention_mask']

                temp_data['sent'] = temp_train
                temp_data['dis_input_ids'] = dis_input_ids
                temp_data['dis_attn'] = dis_attn
                # ipdb.set_trace()
                # self.tokenizer.decode(temp_data['sent']['input_ids'][0].tolist())
                candidate_rel = random.sample(labels_set,self.opt.n_unseen) # ['participant in', 'position held', 'constellation', 'member of']
                # rel_idx = [rel_idx_dict[rel] for rel in candidate_rel]
                # rand_int = random.randint(0, self.opt.n_unseen) # 4
                # # ipdb.set_trace()
                # temp_prompt.extend(candidate_rel[:rand_int]) #['participant in', 'position held', 'constellation', 'member of']
                # # temp_prompt.append(dis_name[0]) #['participant in', 'position held', 'constellation', 'member of', 'after a work by']
                # temp_prompt.extend(candidate_rel[rand_int:])
                prompt =  '[rel_candidate]: '+', '.join(candidate_rel)+ '. '
                temp3 = self.tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
                prompt_ids, prompt_attn = temp3['input_ids'], temp3['attention_mask']
                temp_data['prompt_ids'], temp_data['prompt_attn'] = prompt_ids, prompt_attn      
                # ipdb.set_trace()          
                data.append(temp_data)
        return data        

    def process_Recall_data_decodeinput(self, dataset, labels_set, relation_discribe):
        
        data = []
        for sent in dataset.sents:
            sents = [self.encoder.encode_to_line(trip) for trip in sent.triplets]# sent : Sentence(triplets=[RelationSentence(tokens=['In', '1689'...], head=[11, 12], tail=[16], label='after a work by', head_id='', tail_id='', label_id='P140', error='', raw='', keyword=[''], score=0.0, zerorc_included=True)])
            #sents = ['{"text": "[SENT] : In 1689 , Konstanty was one of the judges who sentenced Kazimierz \\u0141yszczy\\u0144ski to death for atheism .", "summary": "[HEAD] Kazimierz \\u0141yszczy\\u0144ski , [TAIL] atheism , [REL] after a work by ."}\n']
            trips = [trip for trip in sent.triplets]
            #trips = [RelationSentence(tokens=['In', '1689', ',', 'Konstanty', 'was', 'one', 'of', 'the', 'judges', 'who', 'sentenced', 'Kazimierz', 'Łyszczyński', 'to', 'death', 'for', 'atheism', '.'], head=[11, 12], tail=[16], label='after a work by', head_id='', tail_id='', label_id='P140', error='', raw='', keyword=[''], score=0.0, zerorc_included=True)]
            for sent, trip in zip(sents, trips):
                temp_data = {}
                temp_train = {}
                temp_prompt = []

                context, labels = self.encoder.parse_line(sent)# content:'[SENT] : Merpati flight 106 departed Jakarta ( CGK ) on a domestic flight to Tanjung Pandan ( TJQ ) .'   labels:'[HEAD] Kazimierz Łyszczyński , [TAIL] atheism , [REL] after a work by .'
                # ipdb.set_trace()
                temp_train['input_ids'], temp_train['attention_mask'] = [], []
                temp_train['labels'], temp_train['decoder_attention_mask'] = [], []

                temp1 = self.tokenizer(context, return_tensors='pt', add_special_tokens=True)
                temp_train['input_ids'], temp_train['attention_mask'] = temp1['input_ids'], temp1['attention_mask']

                # temp2 = self.tokenizer(labels, return_tensors='pt', add_special_tokens=True)
                # temp_train['decoder_input_ids'], temp_train['decoder_attention_mask'] = temp2['input_ids'], temp2['attention_mask']
                
                dis_id = trip.label_id
                dis_name = [relation_discribe[dis_id][0]]
                discribe = [relation_discribe[dis_id][2]] #这里
                # fin_discribe = dis_name + ': '.join(discribe)+'. '+'<triplet>: '
                fin_discribe = f'{dis_name[0]}:{discribe[0]}. <triplet>:'

                # ipdb.set_trace()
                # temp3 = self.tokenizer(fin_discribe, return_tensors='pt', add_special_tokens=True)
                # dis_input_ids, dis_attn = temp3['input_ids'], temp3['attention_mask']
                
                fin_decode_inputs = fin_discribe
                # ipdb.set_trace()
                temp2 = self.tokenizer(fin_decode_inputs, return_tensors='pt', add_special_tokens=False)
                # if temp2['input_ids'].tolist()[0][-1] == self.tokenizer.eos_token_id:
                #     temp2['input_ids'] = torch.tensor([temp2['input_ids'].tolist()[0][:-1]])
                #     temp2['attention_mask'] = torch.tensor([temp2['attention_mask'].tolist()[0][:-1]])
                dis_input_ids, dis_attn = temp2['input_ids'], temp2['attention_mask']

                temp_data['sent'] = temp_train
                temp_data['dis_input_ids'] = dis_input_ids
                temp_data['dis_attn'] = dis_attn

                candidate_rel = random.sample(labels_set,self.opt.n_unseen) # ['participant in', 'position held', 'constellation', 'member of']
                # rel_idx = [rel_idx_dict[rel] for rel in candidate_rel]
                # rand_int = random.randint(0, self.opt.n_unseen) # 4
                # # ipdb.set_trace()
                # temp_prompt.extend(candidate_rel[:rand_int]) #['participant in', 'position held', 'constellation', 'member of']
                # # temp_prompt.append(dis_name[0]) #['participant in', 'position held', 'constellation', 'member of', 'after a work by']
                # temp_prompt.extend(candidate_rel[rand_int:])
                prompt =  '[rel_candidate]: '+', '.join(candidate_rel)+ '. '
                temp3 = self.tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
                prompt_ids, prompt_attn = temp3['input_ids'], temp3['attention_mask']
                temp_data['prompt_ids'], temp_data['prompt_attn'] = prompt_ids, prompt_attn                      
                # ipdb.set_trace()
                # self.tokenizer.decode(temp_data['sent']['input_ids'][0].tolist())
                data.append(temp_data)
        return data        

    def process_Pipeline1_traindata(self, dataset, labels_set, relation_discribe):
        # ipdb.set_trace()
        # if len(labels_set)<self.opt.n_unseen:
        #     labels_set = labels_set*4 
            # ipdb.set_trace()       
        data = []
        for sent in dataset.sents:
            sents = [self.encoder.encode_to_line(trip) for trip in sent.triplets]# sent : Sentence(triplets=[RelationSentence(tokens=['In', '1689'...], head=[11, 12], tail=[16], label='after a work by', head_id='', tail_id='', label_id='P140', error='', raw='', keyword=[''], score=0.0, zerorc_included=True)])
            #sents = ['{"text": "[SENT] : In 1689 , Konstanty was one of the judges who sentenced Kazimierz \\u0141yszczy\\u0144ski to death for atheism .", "summary": "[HEAD] Kazimierz \\u0141yszczy\\u0144ski , [TAIL] atheism , [REL] after a work by ."}\n']
            trips = [trip for trip in sent.triplets]
            #trips = [RelationSentence(tokens=['In', '1689', ',', 'Konstanty', 'was', 'one', 'of', 'the', 'judges', 'who', 'sentenced', 'Kazimierz', 'Łyszczyński', 'to', 'death', 'for', 'atheism', '.'], head=[11, 12], tail=[16], label='after a work by', head_id='', tail_id='', label_id='P140', error='', raw='', keyword=[''], score=0.0, zerorc_included=True)]
            for sent, trip in zip(sents, trips):
                temp_data = {}
                temp_train = {}
                temp_prompt = []

                context, labels = self.encoder.parse_line(sent)# content:'[SENT] : Merpati flight 106 departed Jakarta ( CGK ) on a domestic flight to Tanjung Pandan ( TJQ ) .'   labels:'[HEAD] Kazimierz Łyszczyński , [TAIL] atheism , [REL] after a work by .'
                # ipdb.set_trace()
                temp_train['input_ids'], temp_train['attention_mask'] = [], []
                temp_train['labels'], temp_train['decoder_attention_mask'] = [], []

                temp1 = self.tokenizer(context, return_tensors='pt', add_special_tokens=True)
                temp_train['input_ids'], temp_train['attention_mask'] = temp1['input_ids'], temp1['attention_mask']

                # temp2 = self.tokenizer(labels, return_tensors='pt', add_special_tokens=True)
                # temp_train['decoder_input_ids'], temp_train['decoder_attention_mask'] = temp2['input_ids'], temp2['attention_mask']
                
                dis_id = trip.label_id
                dis_name = [relation_discribe[dis_id][0]]
                discribe = [relation_discribe[dis_id][2]] #这里
                # fin_discribe = dis_name + ': '.join(discribe)+'. '+'<triplet>: '
                fin_discribe = f'{dis_name[0]}:{discribe[0]}.'

                # ipdb.set_trace()
                # temp3 = self.tokenizer(fin_discribe, return_tensors='pt', add_special_tokens=True)
                # dis_input_ids, dis_attn = temp3['input_ids'], temp3['attention_mask']

                temp2 = self.tokenizer(fin_discribe, return_tensors='pt', add_special_tokens=True)
                dis_input_ids, dis_attn = temp2['input_ids'], temp2['attention_mask']

                temp_data['sent'] = temp_train
                temp_data['dis_input_ids'] = dis_input_ids
                temp_data['dis_attn'] = dis_attn
                # ipdb.set_trace()
                # self.tokenizer.decode(temp_data['sent']['input_ids'][0].tolist())
                # if len(labels_set)<=self.opt.n_unseen-1:
                #     ipdb.set_trace()
                candidate_rel = random.sample(labels_set,self.opt.n_unseen-1) # ['participant in', 'position held', 'constellation', 'member of']
                # rel_idx = [rel_idx_dict[rel] for rel in candidate_rel]
                rand_int = random.randint(0, self.opt.n_unseen - 1) # 4
                # ipdb.set_trace()
                temp_prompt.extend(candidate_rel[:rand_int]) #['participant in', 'position held', 'constellation', 'member of']
                temp_prompt.append(dis_name[0]) #['participant in', 'position held', 'constellation', 'member of', 'after a work by']
                temp_prompt.extend(candidate_rel[rand_int:])
                prompt =  '[rel_candidate]: '+', '.join(temp_prompt)+ '. '
                temp3 = self.tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
                prompt_ids, prompt_attn = temp3['input_ids'], temp3['attention_mask']
                temp_data['prompt_ids'], temp_data['prompt_attn'] = prompt_ids, prompt_attn
                
                temp_label_name = f'[REL]: {trip.label}'
                temp4 = self.tokenizer(temp_label_name, return_tensors='pt', add_special_tokens=True)
                label_name, label_name_attn = temp4['input_ids'], temp4['attention_mask']
                temp_data['label_name'], temp_data['label_name_attn'] = label_name, label_name_attn

                temp5 = self.tokenizer('[REL]: ', return_tensors='pt', add_special_tokens=False)
                REL_token, REL_token_attn = temp5['input_ids'], temp5['attention_mask']
                temp_data['REL_token'], temp_data['REL_token_attn'] = REL_token, REL_token_attn
                data.append(temp_data)
        return data                

    def process_Pipeline2_traindata(self, dataset, labels_set, relation_discribe):
        # ipdb.set_trace()
        # if len(labels_set)<self.opt.n_unseen:
        #     labels_set = labels_set*4 
            # ipdb.set_trace()       
        data = []
        for sent in dataset.sents:
            sents = [self.encoder.encode_to_line(trip) for trip in sent.triplets]# sent : Sentence(triplets=[RelationSentence(tokens=['In', '1689'...], head=[11, 12], tail=[16], label='after a work by', head_id='', tail_id='', label_id='P140', error='', raw='', keyword=[''], score=0.0, zerorc_included=True)])
            #sents = ['{"text": "[SENT] : In 1689 , Konstanty was one of the judges who sentenced Kazimierz \\u0141yszczy\\u0144ski to death for atheism .", "summary": "[HEAD] Kazimierz \\u0141yszczy\\u0144ski , [TAIL] atheism , [REL] after a work by ."}\n']
            trips = [trip for trip in sent.triplets]
            #trips = [RelationSentence(tokens=['In', '1689', ',', 'Konstanty', 'was', 'one', 'of', 'the', 'judges', 'who', 'sentenced', 'Kazimierz', 'Łyszczyński', 'to', 'death', 'for', 'atheism', '.'], head=[11, 12], tail=[16], label='after a work by', head_id='', tail_id='', label_id='P140', error='', raw='', keyword=[''], score=0.0, zerorc_included=True)]
            for sent, trip in zip(sents, trips):
                temp_data = {}
                temp_train = {}
                temp_prompt = []

                context, labels = self.encoder.parse_line(sent)# content:'[SENT] : Merpati flight 106 departed Jakarta ( CGK ) on a domestic flight to Tanjung Pandan ( TJQ ) .'   labels:'[HEAD] Kazimierz Łyszczyński , [TAIL] atheism , [REL] after a work by .'
                # ipdb.set_trace()
                temp_train['input_ids'], temp_train['attention_mask'] = [], []
                temp_train['labels'], temp_train['decoder_attention_mask'] = [], []

                temp1 = self.tokenizer(context, return_tensors='pt', add_special_tokens=True)
                temp_train['input_ids'], temp_train['attention_mask'] = temp1['input_ids'], temp1['attention_mask']

                temp2 = self.tokenizer(labels, return_tensors='pt', add_special_tokens=True)
                temp_train['decoder_input_ids'], temp_train['decoder_attention_mask'] = temp2['input_ids'], temp2['attention_mask']
                
                dis_id = trip.label_id
                dis_name = [relation_discribe[dis_id][0]]
                discribe = [relation_discribe[dis_id][2]] #这里
                # fin_discribe = dis_name + ': '.join(discribe)+'. '+'<triplet>: '
                fin_discribe = f'{dis_name[0]}:{discribe[0]}.'

                # ipdb.set_trace()
                # temp3 = self.tokenizer(fin_discribe, return_tensors='pt', add_special_tokens=True)
                # dis_input_ids, dis_attn = temp3['input_ids'], temp3['attention_mask']

                temp3 = self.tokenizer(fin_discribe, return_tensors='pt', add_special_tokens=False)
                dis_input_ids, dis_attn = temp3['input_ids'], temp3['attention_mask']

                temp_data['sent'] = temp_train
                temp_data['fin_discribe'] = dis_input_ids
                temp_data['fin_discribe_attn'] = dis_attn
                # ipdb.set_trace()
                # self.tokenizer.decode(temp_data['sent']['input_ids'][0].tolist())
                # if len(labels_set)<=self.opt.n_unseen-1:
                #     ipdb.set_trace()
                data.append(temp_data)
        return data           

    def process_Pipeline_data(self, dataset, labels_set, relation_discribe):
        # ipdb.set_trace()  
        # if len(labels_set)<self.opt.n_unseen:
        #     labels_set = labels_set*4
        # ipdb.set_trace()
        data = []
        
        dataset.sents = [sent.triplets[0] for sent in dataset.sents]#这里
        for sent in dataset.sents:
            sents = [self.encoder.encode_to_line(trip) for trip in [sent]]# sent : Sentence(triplets=[RelationSentence(tokens=['In', '1689'...], head=[11, 12], tail=[16], label='after a work by', head_id='', tail_id='', label_id='P140', error='', raw='', keyword=[''], score=0.0, zerorc_included=True)])
            #sents = ['{"text": "[SENT] : In 1689 , Konstanty was one of the judges who sentenced Kazimierz \\u0141yszczy\\u0144ski to death for atheism .", "summary": "[HEAD] Kazimierz \\u0141yszczy\\u0144ski , [TAIL] atheism , [REL] after a work by ."}\n']
            trips = [trip for trip in [sent]]
            #trips = [RelationSentence(tokens=['In', '1689', ',', 'Konstanty', 'was', 'one', 'of', 'the', 'judges', 'who', 'sentenced', 'Kazimierz', 'Łyszczyński', 'to', 'death', 'for', 'atheism', '.'], head=[11, 12], tail=[16], label='after a work by', head_id='', tail_id='', label_id='P140', error='', raw='', keyword=[''], score=0.0, zerorc_included=True)]
            for sent, trip in zip(sents, trips):
                temp_data = {}
                temp_train = {}
                temp_prompt = []

                context, labels = self.encoder.parse_line(sent)# content:'[SENT] : Merpati flight 106 departed Jakarta ( CGK ) on a domestic flight to Tanjung Pandan ( TJQ ) .'   labels:'[HEAD] Kazimierz Łyszczyński , [TAIL] atheism , [REL] after a work by .'
                # ipdb.set_trace()
                temp_train['input_ids'], temp_train['attention_mask'] = [], []
                temp_train['labels'], temp_train['decoder_attention_mask'] = [], []

                context1 = context + ' Choose [REL] from [rel_candidate]:'
                context2 = context + ' Extract <triplet> from [sent]:'

                dis_id = trip.label_id
                dis_name = [relation_discribe[dis_id][0]]
                discribe = [relation_discribe[dis_id][2]] #这里
                candidate_rel = random.sample(labels_set,self.opt.n_unseen-1) # ['participant in', 'position held', 'constellation', 'member of']
                rand_int = random.randint(0, self.opt.n_unseen - 1) # 4
                temp_prompt.extend(candidate_rel[:rand_int]) #['participant in', 'position held', 'constellation', 'member of']
                temp_prompt.append(dis_name[0]) #['participant in', 'position held', 'constellation', 'member of', 'after a work by']
                temp_prompt.extend(candidate_rel[rand_int:])

                # ipdb.set_trace()
                prompt =  '<Task1>, Choose [REL] in [sent] from [rel_candidate]. [rel_candidate]: '+', '.join(temp_prompt)+ '. '+ context
                temp1 = self.tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
                prompt_ids, prompt_attn = temp1['input_ids'], temp1['attention_mask']
                temp_data['task1_input_ids'], temp_data['task1_input_attn'] = prompt_ids, prompt_attn
                
                temp_label_name = f'[REL]: {trip.label}'
                temp2 = self.tokenizer(temp_label_name, return_tensors='pt', add_special_tokens=True)
                label_name, label_name_attn = temp2['input_ids'], temp2['attention_mask']
                temp_data['task1_decoder_ids'], temp_data['task1_decoder_attn'] = label_name, label_name_attn


                fin_discribe = f'<Task2>, Extract <triplet> in [sent] by [Relation_discribe]. [Relation_discribe]: {dis_name[0]}, {discribe[0]}. {context}'
                temp3 = self.tokenizer(fin_discribe, return_tensors='pt', add_special_tokens=True)
                temp_data['task2_input_ids'], temp_data['task2_input_attn'] = temp3['input_ids'], temp3['attention_mask']

                token_label = '<triplet>: '+labels
                temp4 = self.tokenizer(token_label, return_tensors='pt', add_special_tokens=True)          
                temp_data['task2_decoder_ids'], temp_data['task2_decoder_attn'] = temp4['input_ids'], temp4['attention_mask']                      
                data.append(temp_data)
        return data        