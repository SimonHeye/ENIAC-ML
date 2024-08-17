import torch
# from model import BaseT5
from encoding import *
from bertviz import head_view, model_view
from transformers import BertTokenizer, BertModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
from models.T5withAdapter import T5ForConditionalGenerationWithAdapter
from zsre_dataset import Dataset
from trainer import TextGenerator_weight, Trainer, test
from zsre_dataset import Dataset, Sentence
from torch.utils.data import DataLoader
from utils import collate_fn_pretrain, collate_fn_Baseline, find_sublist_index
from model import BaseT5
import ipdb
import argparse



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
    parser.add_argument('--trainwelldown_model', default='/user_data/wujy/SimonHeye/META/outputs/bsz-1_ep-1_noreptile/fewrel/noactive/1712500116.131251/unseen_15_seed_0/best_extractor', type=str)
    parser.add_argument('--n_unseen', default=15, type=int)
    
    args = parser.parse_args()
    args.testdata_path = "../META/data/baselinedata/fewrel/unseen_5_seed_0/one_test.jsonl"
    args.relationfile_path = "../Meta_Extra_Generation/data/rebel_dataset/testdata/relations_count.tsv"
    

    args.task_type = "multi"
    args.path_pred = f"../outputs/bsz-4_ep-1_pipeline/fewrel/noactive/1712546678.1794293/unseen_15_seed_0/{args.task_type}/"
    # args.gold_spilit_path= f"../META/outputs/bsz-1_ep-1_noreptile/wiki/noactive/1713573648.3481178/unseen_15_seed_0/{args.task_type}/gold.json" 
    args.gold_spilit_path= f"../META/outputs/bsz-1_ep-1_noreptile/wiki/noactive/1713449268.7124434/unseen_5_seed_0/multi/gold.json"    
    return args

def getoutput(config):
    own_encoder = ExtractEncoder_plus()
    t5_tokenizer = T5Tokenizer.from_pretrained(config.trainwelldown_model)
    # t5_tokenizer.add_tokens(['[HEAD]', '[TAIL]', '[REL]', '[SENT]', '[Relation_discribe]'])

    t5_model = T5ForConditionalGeneration.from_pretrained(config.trainwelldown_model)
    # t5_model.resize_token_embeddings(len(t5_tokenizer))

    model = BaseT5(config, t5_tokenizer, t5_model, own_encoder)
    Base_Trainer = test(config, model, t5_tokenizer,own_encoder)  
    path_pred = str(config.path_pred + 'finpred.jsonl')
    

    Base_Trainer.Pipeline_predict(config.testdata_path, path_pred, model_path = config.trainwelldown_model, task_type = config.task_type, gold_spilit_path = config.gold_spilit_path)

    path_pred = "../META/outputs/bsz-1_ep-1_noreptile/wiki/noactive/1713449268.7124434/unseen_5_seed_0/multi/finpred_after.jsonl"
    results = compute_score_weight(path_pred, config.gold_spilit_path)
    print(json.dumps(results, indent=2))    


if __name__ == "__main__":
    opt = init_args()
    getoutput(opt)
