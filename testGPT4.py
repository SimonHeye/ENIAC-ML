import torch
# from model import BaseT5
from encoding import *
from bertviz import head_view, model_view
from zsre_dataset import Dataset
from GPTtrainer import testGPT
from zsre_dataset import Dataset, Sentence
from torch.utils.data import DataLoader
from utils import collate_fn_pretrain, collate_fn_Baseline, find_sublist_index
import ipdb
import argparse
from openai import OpenAI



def safe_divide(a: float, b: float) -> float:
    if a == 0 or b == 0:
        return 0
    return a / b

def compute_score_weight(path_pred, path_gold):
    pred = Dataset.load(path_pred)
    gold = Dataset.load(path_gold)
    # ipdb.set_trace()
    num_pred = 0
    num_gold = 0
    num_correct = 0
    # num_trip = 0 #这里
    
    for i in range(len(gold.sents)):
        num_pred += len(pred.sents[i].triplets)
        num_gold += len(gold.sents[i].triplets)
        
        for p in pred.sents[i].triplets:
            for g in gold.sents[i].triplets:
                if (p.head, p.tail, p.label) == (g.head, g.tail, g.label):
                    num_correct += 1
                    
    precision = safe_divide(num_correct, num_pred)
    recall = safe_divide(num_correct, num_gold)
    accuracy = safe_divide(num_correct, num_pred)
    info = dict(
        # path_pred=path_pred,
        # path_gold=path_gold,
        precision=precision,
        recall=recall,
        accuracy=accuracy,
        score=safe_divide(2 * precision * recall, precision + recall),
    )
    return info


    #gpt
    # for i in range(len(gold.sents)):
    #     num_pred += len(pred.sents[i].triplets)
    #     num_gold += len(gold.sents[i].triplets)
    #     for p in pred.sents[i].triplets:
    #         for g in gold.sents[i].triplets:
    #             if (p.head, p.tail, p.label) == (g.head, g.tail, g.label):
    #                 num_correct += 1
    #     num_trip += len(pred.sents[i].triplets)  # 修改这一行

    #单多混合
    # for i in range(len(gold.sents)):
    #     num_pred += len(pred.sents[i].triplets)
    #     num_gold += len(gold.sents[i].triplets)
    #     # print(num_trip)
    #     for p in pred.sents[num_trip].triplets:
    #         for g in gold.sents[i].triplets:
    #             if (p.head, p.tail, p.label) == (g.head, g.tail, g.label):
    #                 num_correct += 1
    #     num_trip += len(gold.sents[i].triplets)

    precision = safe_divide(num_correct, num_pred)
    recall = safe_divide(num_correct, num_gold)
    accuracy = safe_divide(num_correct, num_pred)

    info = dict(
        # path_pred=path_pred,
        # path_gold=path_gold,
        precision=precision,
        recall=recall,
        accuracy=accuracy,
        score=safe_divide(2 * precision * recall, precision + recall),
    )
    return info

def compute_score_weight_plus(path_pred, path_gold):
    pred = Dataset.load(path_pred)
    gold = Dataset.load(path_gold)
    num_pred = 0
    num_gold = 0
    num_correct = 0
    num_trip = 0

    for i in range(len(gold.sents)):
        num_pred += len(pred.sents[i].triplets) if i < len(pred.sents) else 0
        num_gold += len(gold.sents[i].triplets)
        if i < len(pred.sents):
            for p in pred.sents[i].triplets:
                for g in gold.sents[i].triplets:
                    if (p.head, p.tail, p.label) == (g.head, g.tail, g.label):
                        num_correct += 1

    precision = safe_divide(num_correct, num_pred)
    recall = safe_divide(num_correct, num_gold)
    
    info = dict(
        precision=precision,
        recall=recall,
        score=safe_divide(2 * precision * recall, precision + recall),
    )
    return info


def init_args():
    parser = argparse.ArgumentParser("getoutput")
    parser.add_argument('--device', default='cuda:0', type=str)    
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--n_unseen', default=5, type=int)
    
    args = parser.parse_args()
    args.testdata_path = "D:\\META\\data\\zero_rte\\fewrel\\unseen_5_seed_0\\test.jsonl"
    

    args.task_type = "singal"
    args.path_pred = f"D:\\META\\outputs\\gpt4\\fewrel\\unseen_5_seed_0\\{args.task_type}\\"
    args.gold_spilit_path= f"D:\\META\\outputs\\gpt4\\fewrel\\unseen_5_seed_0\\{args.task_type}\\gold.json" 
    # args.gold_spilit_path= f"/user_data/wujy/SimonHeye/META/outputs/bsz-1_ep-1_noreptile/wiki/unseen_5_seed_0/multi/gold.json"    
    return args

def getoutput(config):
    own_encoder = ExtractEncoder_plus()
    Base_Trainer = testGPT(config, own_encoder)  
    path_pred = str(config.path_pred + 'finpred.jsonl')
    
    Base_Trainer.predict_withconstrain(config.testdata_path, config.task_type, config.gold_spilit_path, path_pred)

    results = compute_score_weight(path_pred, config.gold_spilit_path)
    print(json.dumps(results, indent=2))    


if __name__ == "__main__":
    opt = init_args()
    getoutput(opt)