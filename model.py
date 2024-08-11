import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Config
from transformers import pipeline, set_seed
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel, GPT2Config
from torch.nn.utils.rnn import pad_sequence
from transformers.models import gpt2
import torch
from pydantic import BaseModel
import torch.nn.functional as F
import ipdb
from typing import Dict, List, Optional, Tuple



class BaseT5(nn.Module):
    def __init__(self, args, tokenizer, t5_model, encoder):
        super(BaseT5, self).__init__()
        self.args = args
        self.tokenizer=tokenizer
        self.encoder=encoder
        self.config = T5Config.from_pretrained(self.args.trainwelldown_model) #这里
        self.t5 = t5_model
        # self.t5 = T5ForConditionalGeneration.from_pretrained(self.args.t5_pretrain_model_path, config=self.config)
    def forward(self, input_ids, attention_mask, labels, decoder_attention_mask, is_pipeline2, batch_pos=None):
        outputs = self.t5(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=None,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
            
            is_pipeline2=is_pipeline2,
            batch_pos = batch_pos
        )
        # ipdb.set_trace()
        # return outputs.loss, None
        return outputs

    def predict(self, input_ids, attention_mask):
        return self.t5.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.args.max_len,
            num_beams=self.args.num_beams,
        )
    def save_pretrained(self,checkpoint_path):
        self.t5.save_pretrained(checkpoint_path)

class ProtoT5(nn.Module):
    def __init__(self, args, tokenizer, t5_model, encoder, use_proto=True):
        super(ProtoT5, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.config = T5Config.from_pretrained(self.args.trainwelldown_model)
        self.t5 = t5_model
        self.encoder = encoder
        self.leak_relu = torch.nn.LeakyReLU(0.1)
        self.ignore_index = -100
        self.use_proto_num = 0
        if use_proto:
            self.proj_head = nn.Linear(2 * self.config.d_model, self.config.d_model, bias=True)
            self.classifier_head = nn.Linear(self.config.d_model, 2, bias=True)
            self.proj_tail = nn.Linear(2 * self.config.d_model, self.config.d_model, bias=True)
            self.classifier_tail = nn.Linear(self.config.d_model, 2, bias=True)
            self.proj_rel = nn.Linear(2 * self.config.d_model, self.config.d_model, bias=True)
            self.classifier_rel = nn.Linear(self.config.d_model, 2, bias=True)

    def forward(self, input_ids, attention_mask, labels, decoder_attention_mask, decoder_input_ids=None, rel_pos=None, REL_addtoken_pos_decode=None, head_pos=None, tail_pos=None, 
                HEAD_addtoken_pos_encode=None, TAIL_addtoken_pos_encode=None, HEAD_addtoken_pos_decode=None, TAIL_addtoken_pos_decode=None, is_pipeline1=None, aux_loss_weight=0.1, **kwargs):
        # ipdb.set_trace()
        
        outputs = self.t5(
            input_ids, #'[PROTO] religion, religion. Context : In 1689, Konstanty was one of the judges who sentenced Kazimierz <unk> yszczy<unk> ski to death for atheism.</s>'
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,#空
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,#'Head Entity : Kazimierz <unk> yszczy<unk> ski, Tail Entity : atheism, Relation : religion.</s>'
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
            **kwargs
        )
        #outputs.keys():odict_keys(['loss', 'logits', 'past_key_values', 'decoder_hidden_states', 'decoder_attentions', 'cross_attentions', 'encoder_last_hidden_state', 'encoder_hidden_states', 'encoder_attentions']
        enc_hidden_states = outputs.encoder_hidden_states[-1] #torch.Size([1, 72, 768])
        dec_hidden_states = outputs.decoder_hidden_states[-1]
        
        # ipdb.set_trace()
        if rel_pos is not None and is_pipeline1 and aux_loss_weight>0.:#rel_pos: tensor([[[ 4, 10]]], device='cuda:0')
            try:
                B,rel_num,_ = rel_pos.shape # torch.Size([1, 1, 2])
                rel_pos = rel_pos.reshape(-1,2) #tensor([[ 4, 10]], device='cuda:0')
                temp_label = torch.tensor([1]*rel_pos.shape[0]).cuda()#[1]*第一维的大小=tensor([1], device='cuda:0')
                # rel_proto = torch.stack([torch.stack([torch.mean(enc_hidden_states[i,rel_pos[i,j,0]:rel_pos[i,j,1]],dim=0) for j in range(rel_num)]) for i in range(B)])
                rel_proto = torch.stack([torch.max(enc_hidden_states[i,rel_pos[i,0]:rel_pos[i,1]],dim=0)[0] if rel_pos[i,0]<rel_pos[i,1] else torch.zeros_like(enc_hidden_states[i,0]) for i in range(B*rel_num)]) # (enc_hidden_states[0,rel_pos[0,0]:rel_pos[0,1]]).size() = torch.Size([6, 768]); (torch.max(enc_hidden_states[0,rel_pos[0,0]:rel_pos[0,1]],dim=0)[0]).size() =  torch.Size([768]); rel_proto.size = torch.Size([1, 768])
                dec_proto = torch.stack([dec_hidden_states[i,0] for i in range(B) for j in range(rel_num)])#dec_proto,size()=torch.Size([1, 768])

                hidden_states = torch.cat([rel_proto,dec_proto],dim=-1)
                # states = self.leak_relu(self.proj_fc(hidden_states))
                states = torch.relu(self.proj_rel(hidden_states))
                logits = self.classifier_rel(states)

                aux_proto_loss = torch.sum(F.cross_entropy(logits.view(-1, 2), temp_label, reduction='none'))
                outputs.loss += aux_loss_weight * aux_proto_loss
            except:
                print(f"enc_hidden_states:{enc_hidden_states}")
                print(f"dec_hidden_states:{dec_hidden_states}")
                print(f"rel_pos:{rel_pos}")
                return outputs
            
        if head_pos is not None and not is_pipeline1 and aux_loss_weight>0.:
            try:
                B,head_num,_ = head_pos.shape #tensor([[[18, 19]]], device='cuda:0')
                temp_label = torch.tensor([1]*B*head_num*2).cuda()#temp_label = tensor([1, 1], device='cuda:0')

                head_pos = head_pos.reshape(-1,2) 
                tail_pos = tail_pos.reshape(-1,2)  
                
                head_proto = torch.stack([torch.max(enc_hidden_states[i,head_pos[i,0]:head_pos[i,1]],dim=0)[0] if head_pos[i,0]<head_pos[i,1] else torch.zeros_like(enc_hidden_states[i,0]) for i in range(B*head_num)]) # torch.Size([1, 768])
                tail_proto = torch.stack([torch.max(enc_hidden_states[i,tail_pos[i,0]:tail_pos[i,1]],dim=0)[0] if tail_pos[i,0]<tail_pos[i,1] else torch.zeros_like(enc_hidden_states[i,0]) for i in range(B*head_num)]) 
                
                # ipdb.set_trace()
                head_discribe_proto = torch.stack([enc_hidden_states[i,HEAD_addtoken_pos_encode[0]] for i in range(B) for j in range(head_num)])#这里
                tail_discribe_proto = torch.stack([enc_hidden_states[i,TAIL_addtoken_pos_encode[0]] for i in range(B) for j in range(head_num)])
                    
                head_dec_proto = torch.stack([dec_hidden_states[i,HEAD_addtoken_pos_decode[0]] for i in range(B) for j in range(head_num)])#这里
                tail_dec_proto = torch.stack([dec_hidden_states[i,TAIL_addtoken_pos_decode[0]] for i in range(B) for j in range(head_num)])

                head_hidden_states1 = torch.cat([head_proto,head_dec_proto],dim=-1)
                tail_hidden_states1 = torch.cat([tail_proto,tail_dec_proto],dim=-1)
                hidden_states1 = torch.stack([head_hidden_states1,tail_hidden_states1],dim=0)  # torch.Size([2, 1, 1536])
                states1 = torch.relu(self.proj_head(hidden_states1)) # torch.Size([2, 1, 768])
                logits1 = self.classifier_head(states1) # torch.Size([2, 1, 2])
                # raw_aux_proto_loss1 = F.cross_entropy(logits1.view(-1,2), temp_label, reduction='none')#logits.view(-1,2):torch.Size([2, 2]), temp_label:tensor([1, 1], device='cuda:0')               
                aux_proto_loss1 = torch.sum(F.cross_entropy(logits1.view(-1,2), temp_label, reduction='none'))#logits.view(-1,2):torch.Size([2, 2]), temp_label:tensor([1, 1], device='cuda:0')
                # outputs.loss += aux_loss_weight*aux_proto_loss1
                
                head_hidden_states2 = torch.cat([head_proto,head_discribe_proto],dim=-1)
                tail_hidden_states2 = torch.cat([tail_proto,tail_discribe_proto],dim=-1)
                hidden_states2 = torch.stack([head_hidden_states2,tail_hidden_states2],dim=0)  # torch.Size([2, 1, 1536])
                states2 = torch.relu(self.proj_head(hidden_states2)) # torch.Size([2, 1, 768])
                logits2 = self.classifier_head(states2) # torch.Size([2, 1, 2])
                raw_aux_proto_loss2 = F.cross_entropy(logits2.view(-1,2), temp_label, reduction='none')#logits.view(-1,2):torch.Size([2, 2]), temp_label:tensor([1, 1], device='cuda:0')                
                # aux_proto_loss2 = torch.sum(F.cross_entropy(logits2.view(-1,2), temp_label, reduction='none'))#logits.view(-1,2):torch.Size([2, 2]), temp_label:tensor([1, 1], device='cuda:0')
                # outputs.loss += aux_loss_weight*aux_proto_loss2
                
                # fin_aux_proto_loss2 = aux_proto_loss2/2
                # ipdb.set_trace()
                if raw_aux_proto_loss2[0]>0.2 and raw_aux_proto_loss2[1]>0.2:
                    self.use_proto_num+=1
                    outputs.loss += aux_loss_weight*aux_proto_loss1
                    # print(self.use_proto_num)
                    
                # log_prob1 = torch.log(raw_aux_proto_loss1)
                # prob2 = raw_aux_proto_loss2
                # kl_div_loss = F.kl_div(log_prob1, prob2, reduction='batchmean')
                # outputs.loss += aux_loss_weight * kl_div_loss * 0.01
                
            except:
                print(f"enc_hidden_states:{enc_hidden_states}")
                print(f"dec_hidden_states:{dec_hidden_states}")
                print(f"head_pos:{head_pos}")
                print(f"tail_pos:{tail_pos}")
                print(f"input_ids:{input_ids}")
                return outputs

        return outputs

    def predict(self, input_ids, attention_mask, labels, decoder_attention_mask, decoder_input_ids=None, rel_pos=None, rel_label_int=None, rel_dec_proto=None,
                **kwargs):
        outputs = self.t5(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )
        # if rel_pos is not None:
        #     B, rel_num = rel_pos.shape[0], rel_pos.shape[1]
        #     enc_hidden_states = outputs.encoder_hidden_states[-1]
        #     dec_hidden_states = outputs.decoder_hidden_states[-1]
        #     rel_proto = torch.stack([torch.stack(
        #         [torch.mean(enc_hidden_states[i, rel_pos[i, j, 0]:rel_pos[i, j, 1]], dim=0) for j in range(rel_num)])
        #                              for i in range(B)])
        return outputs

    def generate(self, input_ids, attention_mask, labels, decoder_attention_mask, decoder_input_ids=None, rel_pos=None, rel_label_int=None, rel_dec_proto=None,
                 **kwargs):
        outputs = self.t5(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True
        )
        return outputs

    def save_pretrained(self,checkpoint_path):
        self.t5.save_pretrained(checkpoint_path)
