import pdb
import random

import ipdb
import torch
import copy
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import pandas as pd
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from pydantic import BaseModel
from zsre_dataset import Dataset
import numpy as np
import re

def collate_fn(batch):
    ins = dict()
    ins['input_ids'] = pad_sequence([x['input_ids'].squeeze(0) for x in batch], batch_first=True, padding_value=50256) # self.tokenizer.eos_token_id
    ins['attention_mask'] = pad_sequence([x['attention_mask'].squeeze(0) for x in batch], batch_first=True)
    ins['labels'] =ins['input_ids']
    return ins

# temp_data['sent'] = sent
# temp_data['candidate_sents'] = candidate_sents
# temp_data['prompt_idx'] = prompt_idx
# temp_data['prompt_attn'] = prompt_attn

def collate_fn_extract_weight(batch):
    ins = dict()
    # for x in batch
    ins['input_ids'] = pad_sequence([torch.cat([x['prompt_ids'],x['sent']['input_ids']],-1).squeeze(0) for x in batch], batch_first=True)
    ins['attention_mask'] = pad_sequence([torch.cat([x['prompt_attn'],x['sent']['attention_mask']],-1).squeeze(0) for x in batch], batch_first=True)
    ins['decoder_attention_mask'] = pad_sequence([x['sent']['decoder_attention_mask'].squeeze(0) for x in batch], batch_first=True)
    ins['labels'] = pad_sequence([x['sent']['decoder_input_ids'].squeeze(0) for x in batch], batch_first=True)

    ins['support_ids'] = pad_sequence([torch.cat([x['prompt_ids'], sent['input_ids']], -1).squeeze(0) for x in batch for sent in x['candidate_sents']], batch_first=True)
    ins['support_attn'] = pad_sequence([torch.cat([x['prompt_attn'], sent['attention_mask']], -1).squeeze(0) for x in batch for sent in x['candidate_sents']], batch_first=True)
    ins['support_decoder_attention_mask'] = pad_sequence([sent['decoder_attention_mask'].squeeze(0) for x in batch for sent in x['candidate_sents']], batch_first=True)
    ins['support_labels'] = pad_sequence([sent['decoder_input_ids'].squeeze(0) for x in batch for sent in x['candidate_sents']], batch_first=True)
    return ins

def collate_fn_Baseline(batch):
    ins = dict()
    # for x in batch
    ins['input_ids'] = pad_sequence([torch.cat([x['dis_input_ids'],x['sent']['input_ids']],-1).squeeze(0) for x in batch], batch_first=True)
    ins['attention_mask'] = pad_sequence([torch.cat([x['dis_attn'],x['sent']['attention_mask']],-1).squeeze(0) for x in batch], batch_first=True)
    ins['decoder_attention_mask'] = pad_sequence([x['sent']['decoder_attention_mask'].squeeze(0) for x in batch], batch_first=True)
    ins['labels'] = pad_sequence([x['sent']['decoder_input_ids'].squeeze(0) for x in batch], batch_first=True)
    
    return ins 

def collate_fn_Recall(batch):
    ins = dict()
    # for x in batch
    ins['input_ids'] = pad_sequence([x['sent']['input_ids'].squeeze(0) for x in batch], batch_first=True)
    ins['attention_mask'] = pad_sequence([x['sent']['attention_mask'].squeeze(0) for x in batch], batch_first=True)
    ins['decoder_attention_mask'] = pad_sequence([x['dis_attn'].squeeze(0) for x in batch], batch_first=True)
    ins['labels'] = pad_sequence([x['dis_input_ids'].squeeze(0) for x in batch], batch_first=True)
    
    return ins 

def collate_fn_Recall_plus(batch):
    ins = dict()
    # for x in batch
    ins['input_ids'] = pad_sequence([torch.cat([x['prompt_ids'],x['sent']['input_ids']],-1).squeeze(0) for x in batch], batch_first=True)
    ins['attention_mask'] = pad_sequence([torch.cat([x['prompt_attn'],x['sent']['attention_mask']],-1).squeeze(0) for x in batch], batch_first=True)
    ins['decoder_attention_mask'] = pad_sequence([x['dis_attn'].squeeze(0) for x in batch], batch_first=True)
    ins['labels'] = pad_sequence([x['dis_input_ids'].squeeze(0) for x in batch], batch_first=True)
    
    return ins 

def collate_fn_Recall_help(batch):
    ins = dict()
    # for x in batch
    ins['input_ids'] = pad_sequence([x['sent']['input_ids'].squeeze(0) for x in batch], batch_first=True)
    ins['attention_mask'] = pad_sequence([x['sent']['attention_mask'].squeeze(0) for x in batch], batch_first=True)
    # ins['decoder_attention_mask'] = pad_sequence([x['dis_attn'].squeeze(0) for x in batch], batch_first=True)
    # ins['decoder_input_ids'] = pad_sequence([x['dis_input_ids'].squeeze(0) for x in batch], batch_first=True)
    ins['decoder_inputs_embeds'] = pad_sequence([x['dis_input_ids'].squeeze(0) for x in batch], batch_first=True)
    
    return ins 

def collate_fn_Recall_help_plus(batch):
    ins = dict()
    # for x in batch
    ins['input_ids'] = pad_sequence([torch.cat([x['prompt_ids'],x['sent']['input_ids']],-1).squeeze(0) for x in batch], batch_first=True)
    ins['attention_mask'] = pad_sequence([torch.cat([x['prompt_attn'],x['sent']['attention_mask']],-1).squeeze(0) for x in batch], batch_first=True)
    # ins['decoder_attention_mask'] = pad_sequence([x['dis_attn'].squeeze(0) for x in batch], batch_first=True)
    # ins['decoder_input_ids'] = pad_sequence([x['dis_input_ids'].squeeze(0) for x in batch], batch_first=True)
    ins['decoder_inputs_embeds'] = pad_sequence([x['dis_input_ids'].squeeze(0) for x in batch], batch_first=True)
    
    return ins

def collate_fn_Pipeline1(batch):
    ins = dict()
    # for x in batch
    ins['input_ids'] = pad_sequence([torch.cat([x['prompt_ids'],x['sent']['input_ids']],-1).squeeze(0) for x in batch], batch_first=True)
    ins['attention_mask'] = pad_sequence([torch.cat([x['prompt_attn'],x['sent']['attention_mask']],-1).squeeze(0) for x in batch], batch_first=True)
    ins['decoder_attention_mask'] = pad_sequence([x['label_name_attn'].squeeze(0) for x in batch], batch_first=True)
    ins['labels'] = pad_sequence([x['label_name'].squeeze(0) for x in batch], batch_first=True)
    
    return ins 

def collate_fn_Pipeline1_help(batch):
    ins = dict()
    # for x in batch
    ins['input_ids'] = pad_sequence([torch.cat([x['prompt_ids'],x['sent']['input_ids']],-1).squeeze(0) for x in batch], batch_first=True)
    ins['attention_mask'] = pad_sequence([torch.cat([x['prompt_attn'],x['sent']['attention_mask']],-1).squeeze(0) for x in batch], batch_first=True)
    # ins['decoder_attention_mask'] = pad_sequence([x['label_name_attn'].squeeze(0) for x in batch], batch_first=True)
    # ins['labels'] = pad_sequence([x['label_name'].squeeze(0) for x in batch], batch_first=True)
    ins['decoder_inputs_embeds'] = pad_sequence([x['REL_token'].squeeze(0) for x in batch], batch_first=True)
    return ins 

def collate_fn_Pipeline2(batch):
    ins = dict()
    # for x in batch
    ins['input_ids'] = pad_sequence([torch.cat([x['fin_discribe'],x['sent']['input_ids']],-1).squeeze(0) for x in batch], batch_first=True)
    ins['attention_mask'] = pad_sequence([torch.cat([x['fin_discribe_attn'],x['sent']['attention_mask']],-1).squeeze(0) for x in batch], batch_first=True)
    ins['decoder_attention_mask'] = pad_sequence([x['sent']['decoder_attention_mask'].squeeze(0) for x in batch], batch_first=True)
    ins['labels'] = pad_sequence([x['sent']['decoder_input_ids'].squeeze(0) for x in batch], batch_first=True)
    
    return ins 

def collate_fn_Pipeline_traindata(batch):
    ins = dict()
    # for x in batch
    ins['input_ids'] = pad_sequence([x['task1_input_ids'].squeeze(0) for x in batch], batch_first=True)
    ins['attention_mask'] = pad_sequence([x['task1_input_attn'].squeeze(0) for x in batch], batch_first=True)
    ins['decoder_attention_mask'] = pad_sequence([x['task1_decoder_attn'].squeeze(0) for x in batch], batch_first=True)
    ins['labels'] = pad_sequence([x['task1_decoder_ids'].squeeze(0) for x in batch], batch_first=True)
    
    ins['help_input_ids'] = pad_sequence([x['task2_input_ids'].squeeze(0) for x in batch], batch_first=True)
    ins['help_attention_mask'] = pad_sequence([x['task2_input_attn'].squeeze(0) for x in batch], batch_first=True)
    ins['help_decoder_attention_mask'] = pad_sequence([x['task2_decoder_attn'].squeeze(0) for x in batch], batch_first=True)
    ins['help_labels'] = pad_sequence([x['task2_decoder_ids'].squeeze(0) for x in batch], batch_first=True)

    return ins 

def collate_fn_Pipeline_meta_traindata(batch):
    ins = dict()
    # for x in batch
    ins['query_task1_input_ids'] = pad_sequence([x['sent']['task1_input_ids'].squeeze(0) for x in batch], batch_first=True)
    ins['query_task1_attention_mask'] = pad_sequence([x['sent']['task1_input_attn'].squeeze(0) for x in batch], batch_first=True)
    ins['query_task1_decoder_attention_mask'] = pad_sequence([x['sent']['task1_decoder_attn'].squeeze(0) for x in batch], batch_first=True)
    ins['query_task1_labels'] = pad_sequence([x['sent']['task1_decoder_ids'].squeeze(0) for x in batch], batch_first=True)
    
    ins['query_task2_input_ids'] = pad_sequence([x['sent']['task2_input_ids'].squeeze(0) for x in batch], batch_first=True)
    ins['query_task2_attention_mask'] = pad_sequence([x['sent']['task2_input_attn'].squeeze(0) for x in batch], batch_first=True)
    ins['query_task2_decoder_attention_mask'] = pad_sequence([x['sent']['task2_decoder_attn'].squeeze(0) for x in batch], batch_first=True)
    ins['query_task2_labels'] = pad_sequence([x['sent']['task2_decoder_ids'].squeeze(0) for x in batch], batch_first=True)

    ins['support_task1_input_ids'] = pad_sequence([sent['task1_input_ids'].squeeze(0) for x in batch for sent in x['candidate_sents']], batch_first=True)
    ins['support_task1_attention_mask'] = pad_sequence([sent['task1_input_attn'].squeeze(0) for x in batch for sent in x['candidate_sents']], batch_first=True)
    ins['support_task1_decoder_attention_mask'] = pad_sequence([sent['task1_decoder_attn'].squeeze(0) for x in batch for sent in x['candidate_sents']], batch_first=True)
    ins['support_task1_labels'] = pad_sequence([sent['task1_decoder_ids'].squeeze(0) for x in batch for sent in x['candidate_sents']], batch_first=True)   

    ins['support_task2_input_ids'] = pad_sequence([sent['task2_input_ids'].squeeze(0) for x in batch for sent in x['candidate_sents']], batch_first=True)
    ins['support_task2_attention_mask'] = pad_sequence([sent['task2_input_attn'].squeeze(0) for x in batch for sent in x['candidate_sents']], batch_first=True)
    ins['support_task2_decoder_attention_mask'] = pad_sequence([sent['task2_decoder_attn'].squeeze(0) for x in batch for sent in x['candidate_sents']], batch_first=True)
    ins['support_task2_labels'] = pad_sequence([sent['task2_decoder_ids'].squeeze(0) for x in batch for sent in x['candidate_sents']], batch_first=True)       
     
    return ins 

def collate_fn_Pipeline_ActiveMeta_traindata(batch):
    ins = dict()
    # for x in batch
    ins['input_ids'] = pad_sequence([x['task1_input_ids'].squeeze(0) for x in batch], batch_first=True)
    ins['attention_mask'] = pad_sequence([x['task1_input_attn'].squeeze(0) for x in batch], batch_first=True)
    ins['decoder_attention_mask'] = pad_sequence([x['task1_decoder_attn'].squeeze(0) for x in batch], batch_first=True)
    ins['labels'] = pad_sequence([x['task1_decoder_ids'].squeeze(0) for x in batch], batch_first=True)
    
    ins['help_input_ids'] = pad_sequence([x['task2_input_ids'].squeeze(0) for x in batch], batch_first=True)
    ins['help_attention_mask'] = pad_sequence([x['task2_input_attn'].squeeze(0) for x in batch], batch_first=True)
    ins['help_decoder_attention_mask'] = pad_sequence([x['task2_decoder_attn'].squeeze(0) for x in batch], batch_first=True)
    ins['help_labels'] = pad_sequence([x['task2_decoder_ids'].squeeze(0) for x in batch], batch_first=True)
    # ipdb.set_trace()
    ins['HEAD_pos_encode'] = [x['HEAD_pos_encode'] for x in batch]
    ins['TAIL_pos_encode'] = [x['TAIL_pos_encode'] for x in batch]
    ins['HEAD_pos_decode'] = [x['HEAD_pos_decode'] for x in batch]
    ins['TAIL_pos_decode'] = [x['TAIL_pos_decode'] for x in batch]
    return ins     

def collate_fn_Pipeline_devdata(batch):
    ins = dict()
    # for x in batch
    ins['input_ids'] = pad_sequence([x['task2_input_ids'].squeeze(0) for x in batch], batch_first=True)
    ins['attention_mask'] = pad_sequence([x['task2_input_attn'].squeeze(0) for x in batch], batch_first=True)
    ins['decoder_attention_mask'] = pad_sequence([x['task2_decoder_attn'].squeeze(0) for x in batch], batch_first=True)
    ins['labels'] = pad_sequence([x['task2_decoder_ids'].squeeze(0) for x in batch], batch_first=True)

    return ins 


def collate_fn_extract(batch):
    # pdb.set_trace()
    ins = dict()
    ins['input_ids'] = pad_sequence([x.squeeze(0) for x_list in batch for x in x_list['input_ids']], batch_first=True)
    ins['attention_mask'] = pad_sequence([x.squeeze(0) for x_list in batch for x in x_list['attention_mask']], batch_first=True)
    # ins['decoder_input_ids'] = pad_sequence([x['input_ids'].squeeze(0) for x in batch], batch_first=True)
    ins['decoder_input_ids'] = None
    ins['decoder_attention_mask'] = pad_sequence([x.squeeze(0) for x_list in batch for x in x_list['decoder_attention_mask']], batch_first=True)
    # ins['labels'] = pad_sequence([x['labels'].squeeze(0) for x in batch], batch_first=True)
    ins['labels'] = pad_sequence([x.squeeze(0) for x_list in batch for x in x_list['decoder_input_ids']], batch_first=True)
    if 'rel_pos' in batch[0]:
        ins['rel_pos'] = torch.tensor([x['rel_pos'] for x in batch])
        ins['rel_dec_proto'] = torch.tensor([x['rel_dec_proto'] for x in batch])

    if 'head_pos' in batch[0]:
        ins['head_pos'] = torch.tensor([x['head_pos'] for x in batch])
        ins['tail_pos'] = torch.tensor([x['tail_pos'] for x in batch])
        ins['head_dec_proto'] = torch.tensor([x['head_dec_proto'] for x in batch])
        ins['tail_dec_proto'] = torch.tensor([x['tail_dec_proto'] for x in batch])

    if 'promp_len' in batch[0]:
        ins['promp_len'] = torch.tensor([x['promp_len'] for x in batch])

    # if 'proto_pos' in batch[0]:  rel_pos
    #     ins['proto_pos'] = [x['proto_pos'] for x in batch]
    #     ins['proto_label'] = [x['proto_label'] for x in batch]
    # if 'labels' in batch[0]:
    #     ins['text_labels'] = [x['labels'] for x in batch]
    # if 'generate_input_ids' in batch[0]:
    #     ins['generate_input_ids'] = pad_sequence([x['generate_input_ids'].squeeze(0) for x in batch], batch_first=True,)
    #     ins['generate_attention_mask'] = pad_sequence([x['generate_attention_mask'].squeeze(0) for x in batch], batch_first=True)
    #     ins['generate_decoder_input_ids'] = None
    #     ins['generate_labels'] = pad_sequence([x['generate_decoder_labels'].squeeze(0) for x in batch], batch_first=True,)
    # if 'context_pos' in batch[0]:
    #     ins['input_ids_con'] = pad_sequence([x['input_ids_con'].squeeze(0) for x in batch], batch_first=True)
    #     ins['attention_mask_con'] = pad_sequence([x['attention_mask_con'].squeeze(0) for x in batch], batch_first=True)
    #     ins['input_ids_neg'] = pad_sequence([x['input_ids_neg'].squeeze(0) for x in batch], batch_first=True)
    #     ins['attention_mask_neg'] = pad_sequence([x['attention_mask_neg'].squeeze(0) for x in batch], batch_first=True)
    #     ins['context_pos'] = [x['context_pos'] for x in batch]
    #     ins['context_pos_neg'] = [x['context_pos_neg'] for x in batch]
    return ins

def collate_fn_pretrain(batch):
    ins = dict()
    ins['input_ids'] = pad_sequence([x.squeeze(0) for x_list in batch for x in x_list['input_ids']], batch_first=True)
    ins['attention_mask'] = pad_sequence([x.squeeze(0) for x_list in batch for x in x_list['attention_mask']], batch_first=True)
    ins['labels'] = pad_sequence([x.squeeze(0) for x_list in batch for x in x_list['labels']], batch_first=True)
    ins['decoder_attention_mask'] = pad_sequence([x.squeeze(0) for x_list in batch for x in x_list['decoder_attention_mask']], batch_first=True)

    return ins

def collate_fn_extract_rc(batch):
    ins = dict()
    ins['input_ids'] = pad_sequence([x.squeeze(0) for x_list in batch for x in x_list['input_ids']], batch_first=True)
    ins['attention_mask'] = pad_sequence([x.squeeze(0) for x_list in batch for x in x_list['attention_mask']], batch_first=True)
    # ins['decoder_input_ids'] = pad_sequence([x['input_ids'].squeeze(0) for x in batch], batch_first=True)
    ins['decoder_input_ids'] = None
    ins['decoder_attention_mask'] = pad_sequence([x.squeeze(0) for x_list in batch for x in x_list['decoder_attention_mask']], batch_first=True)
    # ins['labels'] = pad_sequence([x['labels'].squeeze(0) for x in batch], batch_first=True)
    ins['labels'] = pad_sequence([x.squeeze(0) for x_list in batch for x in x_list['decoder_input_ids']], batch_first=True)
    return ins

def collate_dev_extract(batch):
    ins = dict()
    ins['input_ids'] = pad_sequence([x['input_ids'].squeeze(0) for x in batch], batch_first=True)
    ins['attention_mask'] = pad_sequence([x['attention_mask'].squeeze(0) for x in batch], batch_first=True)
    # ins['decoder_input_ids'] = pad_sequence([x['input_ids'].squeeze(0) for x in batch], batch_first=True)
    ins['decoder_input_ids'] = None
    ins['decoder_attention_mask'] = pad_sequence([x['decoder_attention_mask'].squeeze(0) for x in batch], batch_first=True)
    # ins['labels'] = pad_sequence([x['labels'].squeeze(0) for x in batch], batch_first=True)
    ins['labels'] = pad_sequence([x['decoder_input_ids'].squeeze(0) for x in batch], batch_first=True)
    # if 'labels' in batch[0]:
    #     ins['text_labels'] = [x['labels'] for x in batch]
    return ins

def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ^^ safe to call this function even if cuda is not available


def find_sublist_index(items: list, query: list):
    length = len(query)

    for i in range(len(items) - length + 1):
        if items[i : i + length] == query:
            return i
    return -1

def find_sublist_start_end(items: list, query: list):
    length = len(query)

    for i in range(len(items) - length + 1):
        if items[i : i + length] == query:
            return i, i + length
    return -1, -1

def align_span_to_tokens(span: str, tokens: List[str]) -> Tuple[int, int]:
    # Eg align("John R. Allen, Jr.", ['John', 'R.', 'Allen', ',', 'Jr.'])
    char_word_map = {}
    num_chars = 0
    for i, w in enumerate(tokens):
        for _ in w:
            char_word_map[num_chars] = i
            num_chars += 1
    char_word_map[num_chars] = len(tokens)

    query = span.replace(" ", "")
    text = "".join(tokens)
    if '<unk>' in span:
        #Olav<unk>ygard   OlavØygard
        pattern = query.replace('<unk>','.')
        i, j = re.search(pattern,text).span()
        start = char_word_map[i]
        end = char_word_map[j]
    else:
        assert query in text
        i = text.find(query)
        start = char_word_map[i]
        end = char_word_map[i + len(query) - 1]
    assert 0 <= start <= end
    return start, end + 1

def find_span(span: str, tokens: List[str]) -> List[int]:
    #span: head, tokens: sent tokens
    if span == "":
        return []

    start = find_sublist_index(tokens, span.split())
    if start >= 0:
        return [start + i for i in range(len(span.split()))]
    else:
        start, end = align_span_to_tokens(span, tokens)
        return list(range(start, end))

class WikiProperty(BaseModel):
    """
    # https://query.wikidata.org
    # All properties with descriptions and aliases and types

    SELECT ?p ?pType ?pLabel ?pDescription ?pAltLabel WHERE {
        ?p wikibase:propertyType ?pType .
        SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
    }
    ORDER BY ASC(xsd:integer(STRAFTER(STR(?p), 'P')))
    """

    p: str
    pType: str
    pLabel: str
    pDescription: str
    pAltLabel: str

    @property
    def id(self) -> str:
        return self.p.split("/")[-1]

    @property
    def aliases(self) -> List[str]:
        names = [n.strip() for n in self.pAltLabel.split(",")]
        return sorted(set(names))

def load_wiki_relation_map(path: str) -> Dict[str, WikiProperty]:
    df = pd.read_csv(path)
    props = [WikiProperty(**r) for r in df.to_dict(orient="records")]
    return {p.id: p for p in props}


def load_label_to_properties(
    path: str, use_alias: bool = True
) -> Dict[str, WikiProperty]:
    relation_map = load_wiki_relation_map(path)
    mapping = {}
    for p in relation_map.values():
        if not p.pLabel in mapping.keys():
            mapping[p.pLabel] = p
    if use_alias:
        for p in relation_map.values():
            for a in p.aliases:
                if a not in mapping.keys():
                    mapping[a] = p
    return mapping

def mark_wiki_entity(edge):
    e1 = edge["left"]
    e2 = edge["right"]
    return e1, e2

def mark_fewrel_entity(edge):
    e1 = edge["h"][2][0]
    e2 = edge["t"][2][0]
    return e1, e2

def f1_score(common, pred, gold):
    p = float(common) / float(pred) if pred != 0 else 0
    r = float(common) / float(gold) if gold != 0 else 0
    f = 2 * p * r / (p + r) if p != 0 or r != 0 else 0
    return p, r, f

def score_rc1(key, prediction, NO_RELATION=0, tag='Zero_RC'):
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation = Counter()

    # Loop over the data to compute a score
    for row in range(len(key)):
        gold = key[row]
        guess = prediction[row]

        if gold == NO_RELATION and guess == NO_RELATION:
            pass
        elif gold == NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
        elif gold != NO_RELATION and guess == NO_RELATION:
            gold_by_relation[gold] += 1
        elif gold != NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1

    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro = float(sum(correct_by_relation.values())) / float(
            sum(guessed_by_relation.values()))
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(
            sum(gold_by_relation.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    print("SET NO_RELATION ID in {}: ".format(tag), NO_RELATION)
    print("Precision (micro): {:.3%}".format(prec_micro))
    print("   Recall (micro): {:.3%}".format(recall_micro))
    print("       F1 (micro): {:.3%}".format(f1_micro))
    return prec_micro, recall_micro, f1_micro

def score_rc(predicted, gold_idx, i=-1, empty_label=None):
    '''
    This evaluation function follows work from Sorokin and Gurevych(https://www.aclweb.org/anthology/D17-1188.pdf)
    code borrowed from the following link:
    https://github.com/UKPLab/emnlp2017-relation-extraction/blob/master/relation_extraction/evaluation/metrics.py
    '''
    if i == -1:
        i = len(predicted)

    complete_rel_set = set(gold_idx) - {empty_label}
    avg_prec = 0.0
    avg_rec = 0.0

    for r in complete_rel_set:
        tp, tp_fp, tp_fn = 0, 0, 0
        for pred,gold in zip(predicted,gold_idx):
            if pred == r:
                tp_fp+=1
            if gold == r:
                tp_fn += 1
            else:
                continue
            if pred ==gold:
                tp+=1

            # r_indices = (predicted[:i] == r)
            # tp = len((predicted_idx[:i][r_indices] == gold_idx[:i][r_indices]).nonzero()[0])
            # tp_fp = len(r_indices.nonzero()[0])
            # tp_fn = len((gold_idx == r).nonzero()[0])

        prec = (tp / tp_fp) if tp_fp > 0 else 0
        rec = tp / tp_fn
        avg_prec += prec
        avg_rec += rec
    f1 = 0
    avg_prec = avg_prec / len(set(predicted))
    avg_rec = avg_rec / len(complete_rel_set)
    if (avg_rec + avg_prec) > 0:
        f1 = 2.0 * avg_prec * avg_rec / (avg_prec + avg_rec)
    print("Precision (micro): {:.3%}".format(avg_prec))
    print("   Recall (micro): {:.3%}".format(avg_rec))
    print("       F1 (micro): {:.3%}".format(f1))
    return avg_prec, avg_rec, f1

def find_sublist_index(items: list, query: list):
    length = len(query)

    for i in range(len(items) - length + 1):
        if items[i : i + length] == query:
            return i
    return -1