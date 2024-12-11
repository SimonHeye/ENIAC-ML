import json
from zsre_dataset import Dataset, Sentence
from torch.utils.data import DataLoader
import ipdb
from tqdm import tqdm
from openai import OpenAI
import random

class testGPT:
    def __init__(self, opt, own_encoder):
        self.opt = opt
        self.encoder = own_encoder
        self.api_key = "sk-GrMq0gCwQtaefKKsCjZbZR7m4fp8YxYqeVH5lLXIG3wW2kBS"
        self.api_base = "https://api.claudeshop.top/v1"
        self.raw_relation_label:list
        
        with open("D:\\META\\REL_discribe.jsonl","r") as f1:
            contents = f1.read()
            json_object = json.loads(contents)
            self.relation_discribe = json_object
    

    def predict_withconstrain(self, path_in, task_type, gold_spilit_path, path_out):
        data = Dataset.load(path_in)
        self.raw_relation_label = data.get_labels()
        
        if task_type=="multi":
            data.sents = [s for s in data.sents if len(s.triplets) > 1]  
        else:
            data.sents = [s for s in data.sents if len(s.triplets) == 1]
        data.save(gold_spilit_path)

        triple_list = []
        if task_type=="singal":
            triple_list = [trip for sent in data.sents for trip in sent.triplets]    
            texts = [trip.text for sent in data.sents for trip in sent.triplets]#所有句子构成的列表
        else:
            triple_list = [sent.triplets[0] for sent in data.sents]    
            texts = [sent.triplets[0].text for sent in data.sents]
        
        sents, sents_no = [], 0

        
        for text in tqdm(texts):
            temp_prompt = []
            # dis_id = text.triplets[0].label_id
            # dis_name = [self.relation_discribe[dis_id][0]]
            # ipdb.set_trace()
            candidate_rel = random.sample(self.raw_relation_label,self.opt.n_unseen) # ['participant in', 'position held', 'constellation', 'member of']
            rand_int = random.randint(0, self.opt.n_unseen) # 
            temp_prompt.extend(candidate_rel[:rand_int]) #['participant in', 'position held', 'constellation', 'member of']
            # temp_prompt.append(dis_name[0])
            temp_prompt.extend(candidate_rel[rand_int:])
            # ipdb.set_trace()
            
            client = OpenAI(api_key=self.api_key, base_url=self.api_base)
            
            # prompt =  '请从候选关系: '+', '.join(temp_prompt)+ '. '+ f'中，选择一个关系，并抽取出' + f'“{text}”' +'中，涉及到的头实体、尾实体、关系三元组,给我一个例如"[HEAD] A, [TAIL] B, [REL] C."的回复：'
            prompt = 'Please select a relation from the candidate relations: ' + ', '.join(temp_prompt) + '. ' + f', select a relation and extract the head entity, tail entity, and relation involved in ' + f'“{text}”' + 'Give me a response such as “[HEAD] A, [TAIL] B, [REL] C.”:'
            
            completion = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
            )
            
            # ipdb.set_trace()
            outputs = completion.choices[0].message.content
            
            triplet = self.encoder.safe_decode_raw(texts[sents_no], y=outputs)
            sents_no+=1

            sents.append(Sentence(triplets=[triplet]))
            Dataset(sents=sents).save(path_out)

    # def predict_multi(self, data, pred_labels, texts, sents, model_path, path_out, use_label_constraint=True, max_target_length=128, search_threshold=-0.9906, device=torch.device("cuda")):
    #     # data.sents = [s for s in data.sents if len(s.triplets) > 1]
    #     # split_data_path = '/user_data/wujy/SimonHeye/META/outputs/bsz-1_ep-1_noreptile/fewrel/noactive/1712383059.494721/unseen_5_seed_0/split_data.json'
    #     # data.save(split_data_path)
    #     stem = Path(path_out).stem
    #     path_raw = path_out.replace(stem, f"{stem}_raw")

    #     gen_model = TextGeneratormulti(
    #         model=T5ForConditionalGeneration.from_pretrained(model_path),
    #         tokenizer=T5Tokenizer.from_pretrained(model_path),
    #         max_length=max_target_length,
    #     )        
    #     # gen_model.model = gen_model.model.to(device)        
    #     # prompt = '[PROTO] ' + ', '.join(pred_labels) + '. '
    #     # prompt=''
    #     constraint = LabelConstraint(labels=pred_labels, tokenizer=self.tokenizer)
    #     searcher = TripletSearchDecoder(
    #         gen=gen_model, encoder=self.encoder, constraint=constraint, relation_discribe=self.relation_discribe, tokenizer=self.tokenizer
    #     )

    #     sents = [
    #         Sentence(tokens=s.tokens, triplets=searcher.run(s.text, s.triplets))
    #         for s in tqdm(data.sents)
    #     ]

    #     Dataset(sents=sents).save(path_raw)
    #     for s in sents:
    #         s.triplets = [t for t in s.triplets if t.score > search_threshold]
    #     Dataset(sents=sents).save(path_out)

         

