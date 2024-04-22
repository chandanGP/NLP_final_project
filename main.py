import logging
import argparse
import math
import os
import torch
import random
import numpy as np
import json, jsonlines
import pickle
import time
import copy
import transformers
from transformers import BertTokenizer
from knowledge_neuron.knowledge_neurons.src.custom_bert import BertForMaskedLM
import torch.nn.functional as F
from knowledge_neuron.knowledge_neurons.src import *

from IPython.display import clear_output
import matplotlib.pyplot as plt
from datasets import load_dataset

class Params:
    def __init__(this):
        this.temp = None

def example2feature(example, max_seq_length, tokenizer):
    """Convert an example into input features"""
    features = []
    tokenslist = []

    ori_tokens = tokenizer.tokenize(example[0])
    # All templates are simple, almost no one will exceed the length limit.
    if len(ori_tokens) > max_seq_length - 2:
        ori_tokens = ori_tokens[:max_seq_length - 2]

    # add special tokens
    tokens = ["[CLS]"] + ori_tokens + ["[SEP]"]
    base_tokens = ["[UNK]"] + ["[UNK]"] * len(ori_tokens) + ["[UNK]"]
    segment_ids = [0] * len(tokens)

    # Generate id and attention mask
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    baseline_ids = tokenizer.convert_tokens_to_ids(base_tokens)
    input_mask = [1] * len(input_ids)

    # Pad [PAD] tokens (id in BERT-base-cased: 0) up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    baseline_ids += padding
    segment_ids += padding
    input_mask += padding

    assert len(baseline_ids) == max_seq_length
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    features = {
        'input_ids': input_ids,
        'input_mask': input_mask,
        'segment_ids': segment_ids,
        'baseline_ids': baseline_ids,
    }
    tokens_info = {
        "tokens":tokens,
        "gold_obj":example[1],
        "pred_obj": None
    }
    if len(example)>2 and example[2]!=None :
        tokens_info['relation'] = example[2]

    return features, tokens_info


def scaled_input(emb, batch_size, num_batch):
    # emb: (1, ffn_size)
    baseline = torch.zeros_like(emb)  # (1, ffn_size)

    num_points = batch_size * num_batch
    step = (emb - baseline) / num_points  # (1, ffn_size)

    res = torch.cat([torch.add(baseline, step * i) for i in range(num_points)], dim=0)  # (num_points, ffn_size)
    return res, step[0]


def convert_to_triplet_ig(ig_list):
    ig_triplet = []
    ig = np.array(ig_list)  # 12, 3072
    max_ig = ig.max()
    for i in range(ig.shape[0]):
        for j in range(ig.shape[1]):
            if ig[i][j] >= max_ig * 0.1:
                ig_triplet.append([i, j, ig[i][j]])
    return ig_triplet

def find_location(a,b):
    output = []
    for idx in range(len(a)):
        start_idx = None
        if a[idx] == b[0] :
            start_idx = idx
        if start_idx != None :
            all_same = True
            temp_lst1 = []
            for temp_idx1,temp_idx2 in enumerate(range(start_idx,start_idx+len(b))):
                if temp_idx1<len(b) and temp_idx2<len(a) and  a[temp_idx2] != b[temp_idx1] :
                    all_same=False
                    break
                temp_lst1.append(temp_idx2)
            if all_same :
                output.append(temp_lst1)
            start_idx=None
    return output

def get_BIRD_data(data,tokenizer,relation_data_path):
    subject_data = {}
    for file_name in os.listdir(relation_data_path):
        temp_data = []
        with open(relation_data_path+'/'+file_name,'r') as fp:
            for line in fp:
                temp_data.append(json.loads(line))
        for data_dict in temp_data:
            relation_type = data_dict['predicate_id']
            subject = data_dict['sub_label'].strip()
            temp1 = subject_data.get(relation_type,[])
            temp1.append(subject)
            subject_data[relation_type] = temp1
    
    relations_found = 0
    relations_not_found = 0
    ROME_data = {}
    for relation, list_of_bags in data.items():
        subject_list = subject_data.get(relation,None)
        if subject_list==None :
            relations_not_found+=1
            continue
        relations_found+=1
        temp_lst1 = []
        for bag in list_of_bags :
            temp_lst2 = []
            for prompt_data in bag:
                temp_data = []
                prompt = prompt_data[0]
                tokenized_prompt = tokenizer.tokenize(prompt)
                done_subject=set()
                for subject in subject_list:
                    index = prompt.find(subject)
                    if index != -1 and subject not in done_subject:
                        #temp_data.append(prompt[:index] +'[X]'+ prompt[index+len(subject):])
                        tokenized_subject = tokenizer.tokenize(subject)
                        temp_data.append([tokenized_prompt,tokenized_subject,find_location(tokenized_prompt,tokenized_subject)])
                        done_subject.add(subject)
                temp_lst2.append(prompt_data+temp_data)
            temp_lst1.append(temp_lst2)
        ROME_data[relation] = temp_lst1
    
    return ROME_data


def get_attribution_scores(input_prompt,true_label,model,tokenizer,arguments,tensor_format=True,relation_type=None):
    args = arguments
    eval_example = [input_prompt,true_label]
    if relation_type != None :
        eval_example.append(relation_type)
    eval_features, tokens_info = example2feature(eval_example, args.max_seq_length, tokenizer)

    device = arguments.device
    n_gpu = arguments.n_gpu

    # convert features to long type tensors
    baseline_ids, input_ids, input_mask, segment_ids = eval_features['baseline_ids'], eval_features['input_ids'], eval_features['input_mask'], eval_features['segment_ids']
    baseline_ids = torch.tensor(baseline_ids, dtype=torch.long).unsqueeze(0)
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    input_mask = torch.tensor(input_mask, dtype=torch.long).unsqueeze(0)
    segment_ids = torch.tensor(segment_ids, dtype=torch.long).unsqueeze(0)
    baseline_ids = baseline_ids.to(device)
    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)
    segment_ids = segment_ids.to(device)

    # record real input length
    input_len = int(input_mask[0].sum())

    # record [MASK]'s position
    tgt_pos = tokens_info['tokens'].index('[MASK]')

    # record various results
    res_dict = {
        'pred': [],
        'ig_pred': [],
        'ig_gold': [],
        'base': [],
        'pred_label':None
    }

    # original pred prob
    if args.get_pred:
        _, logits = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=0)  # (1, n_vocab)
        base_pred_prob = F.softmax(logits, dim=1)  # (1, n_vocab)
        res_dict['pred'].append(base_pred_prob.tolist())
        pred_label = int(torch.argmax(logits[0, :]))
        tokens_info['pred_obj'] = tokenizer.convert_ids_to_tokens(pred_label)
        res_dict['pred_label'] = tokens_info['pred_obj']
    

    if args.get_ig_pred or args.get_ig_gold or args.get_base :

        for tgt_layer in range(model.bert.config.num_hidden_layers):
            ffn_weights, logits = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=tgt_layer)  # (1, ffn_size), (1, n_vocab)
            pred_label = int(torch.argmax(logits[0, :]))  # scalar
            gold_label = tokenizer.convert_tokens_to_ids(tokens_info['gold_obj'])
            tokens_info['pred_obj'] = tokenizer.convert_ids_to_tokens(pred_label)
            res_dict['pred_label'] = tokens_info['pred_obj']
            scaled_weights, weights_step = scaled_input(ffn_weights, args.batch_size, args.num_batch)  # (num_points, ffn_size), (ffn_size)
            scaled_weights.requires_grad_(True)

            # integrated grad at the pred label for each layer
            if args.get_ig_pred:
                ig_pred = None
                for batch_idx in range(args.num_batch):
                    batch_weights = scaled_weights[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size]
                    _, grad = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=tgt_layer, tmp_score=batch_weights, tgt_label=pred_label)  # (batch, n_vocab), (batch, ffn_size)
                    grad = grad.sum(dim=0)  # (ffn_size)
                    ig_pred = grad if ig_pred is None else torch.add(ig_pred, grad)  # (ffn_size)
                ig_pred = ig_pred * weights_step  # (ffn_size)
                res_dict['ig_pred'].append(ig_pred.tolist())

            # integrated grad at the gold label for each layer
            if args.get_ig_gold:
                ig_gold = None
                for batch_idx in range(args.num_batch):
                    batch_weights = scaled_weights[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size]
                    _, grad = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=tgt_layer, tmp_score=batch_weights, tgt_label=gold_label)  # (batch, n_vocab), (batch, ffn_size)
                    grad = grad.sum(dim=0)  # (ffn_size)
                    ig_gold = grad if ig_gold is None else torch.add(ig_gold, grad)  # (ffn_size)
                ig_gold = ig_gold * weights_step  # (ffn_size)
                res_dict['ig_gold'].append(ig_gold.tolist())

            # base ffn_weights for each layer
            if args.get_base:
                res_dict['base'].append(ffn_weights.squeeze().tolist())
    
    if tensor_format :
        if len(res_dict['pred'])>0:
            res_dict['pred'] = torch.tensor(res_dict['pred'])[0][0]
        if len(res_dict['ig_pred'])>0:
            res_dict['ig_pred'] = torch.tensor(res_dict['ig_pred'])
        if len(res_dict['ig_gold'])>0:
            res_dict['ig_gold'] = torch.tensor(res_dict['ig_gold'])
        if len(res_dict['base'])>0:
            res_dict['base'] = torch.tensor(res_dict['base'])
    
    return res_dict


    


def get_knowledge_neurons(model,tokenizer,data,arguments,relations=None,save_folder_path=None,display=False):

    if relations!=None and len(relations) > 0 :
        temp_data_dict = {}
        for relation in relations :
            temp_data_dict[relation] = data[relation]
        data = temp_data_dict

    total_prompts = 0
    for relation, bag_list in data.items():
        for bag in bag_list :
            for prompt in bag:
                total_prompts+=1
    
    total_zero_neurons_bags  = 0
    total_non_zero_neuron_bags = 0
    results_dict = {}
    prompt_count = 0
    cur_printing_len = 0
    current_per_complited = -1
    per_st = time.time()
    
    if display :
        print('==== Identifying knowledge neurons ====')
        print('number of relations considering :',len(data))
        print('total number of prompts:',total_prompts)

    if display :
        print("", end='', flush=True)
    for relation, bag_list in data.items():

        if os.path.exists(save_folder_path+'/'+relation+'_kns.json') :
            print('saved data found for ',relation,' loading it')
            with open(save_folder_path+'/'+relation+'_kns.json','r') as fp :
                data_dict = json.load(fp)
                results_dict[relation] = data_dict[relation]
                continue


        print('getting knowledge neurons for ',relation)
        temp_list1 = []
        for bag in bag_list :
            temp_list2 = []
            refined_neurons_set = set()
            for prompt in bag :
                prompt_count+=1
                arguments.get_pred = True
                arguments.get_ig_pred = False
                arguments.get_ig_gold = True
                arguments.get_base = False
                output_dict = get_attribution_scores(prompt[0],prompt[1],model,tokenizer,arguments,relation_type=prompt[2])
                attribution_scores = output_dict['ig_gold']
                row_indices, column_indices = torch.where(attribution_scores >= arguments.relative_attribution_threshold*torch.max(attribution_scores))
                set_of_neurons = set()
                for i, j in zip(row_indices,column_indices):
                    set_of_neurons.add((int(i.item()),int(j.item())))
                
                if len(refined_neurons_set) > 0 :
                    refined_neurons_set = refined_neurons_set.intersection(set_of_neurons)
                
                else:
                    refined_neurons_set = set_of_neurons
                
                current_per = round((prompt_count/total_prompts) * 100,2)
                
                if display : #current_per > current_per_complited and display
                    per_et = time.time()
                    diff_time = per_et - per_st
                    per_st = time.time()
                    expected_remaining_time = diff_time * (total_prompts - prompt_count)
                    time_lst = []
                    time_taken = expected_remaining_time
                    time_lst = []
                    while time_taken >0 :
                        temp = time_taken % 60
                        time_taken = (time_taken - temp)//60
                        time_lst = [str(int(temp))] + time_lst
                    
                    printing_string = 'Completed:'+str(current_per)+' %'+'     Expected remaining time: '+":".join(time_lst)
                    print('\r' + ' ' * cur_printing_len + '\r', end='', flush=True)
                    print(printing_string,end='', flush=True)
                    cur_printing_len = len(printing_string)


            
            if len(refined_neurons_set) == 0:
                total_zero_neurons_bags+=1
            else:
                total_non_zero_neuron_bags+=1
            
            for prompt in bag :
                temp_lst3 = []
                for neuron in refined_neurons_set :
                    temp_lst3.append([neuron[0],neuron[1]])
                temp_list2.append(prompt+[temp_lst3])
            temp_list1.append(temp_list2)
        results_dict[relation] = temp_list1
        if save_folder_path != None :
            with open(save_folder_path+'/'+relation+'_kns.json','w') as fp :
                json.dump({relation:temp_list1},fp)
        print('\n')
    
    if display :
        print('\n')
    
    return results_dict
    

            


def edit_model_kn_method1(model,tokenizer,train_data,val_data,arguments, lambda_1 = 1, lambda_2 = 8,display=False):
    total_prompts = 0
    for relation, list_of_bags in train_data.items():
        for bag in list_of_bags :
            for prompt_data in bag :
                total_prompts+=1
    
    prompt_count = 0
    device = arguments.device
    model.train()
    cur_printing_len = 0
    for relation, list_of_bags in train_data.items():
        for bag in list_of_bags :
            per_st = time.time()
            for prompt_data in bag :
                prompt_count+=1
                input_prompt, target_word, relation_type, knowledge_neurons = prompt_data[0], prompt_data[1], prompt_data[2], prompt_data[3]

                arguments.get_pred = True
                arguments.get_ig_pred = False
                arguments.get_ig_gold = False
                arguments.get_base = False
                output_dict = get_attribution_scores(input_prompt,target_word,model,tokenizer,arguments,relation_type=relation_type)
                predicted_word = output_dict['pred_label']

                target_word_embeddings = model.bert.embeddings(torch.tensor([tokenizer.convert_tokens_to_ids(tokenizer.tokenize(target_word))],device=device))
                target_word_embeddings = torch.mean(target_word_embeddings.view(-1,target_word_embeddings.shape[-1]),dim=0)

                predicted_word_embeddings = model.bert.embeddings(torch.tensor([tokenizer.convert_tokens_to_ids(tokenizer.tokenize(predicted_word))],device=device))
                predicted_word_embeddings = torch.mean(predicted_word_embeddings.view(-1,predicted_word_embeddings.shape[-1]),dim=0)

                diff_embedding = (lambda_2 * target_word_embeddings) - (lambda_1 * predicted_word_embeddings)

                for neuron in knowledge_neurons:
                    layer_index, neuron_index = neuron[0], neuron[1]
                    mlp_layer = model.bert.encoder.layer[layer_index].output.dense
                    mlp_weights = mlp_layer.weight.data.clone().t()
                    mlp_weights[neuron_index, :] = mlp_weights[neuron_index, :]+diff_embedding
                    mlp_layer.weight.data = mlp_weights.t()
                
                current_per = round((prompt_count/total_prompts) * 100,2)
                if display : #current_per > current_per_complited and display
                    per_et = time.time()
                    diff_time = per_et - per_st
                    per_st = time.time()
                    expected_remaining_time = diff_time * (total_prompts - prompt_count)
                    time_lst = []
                    time_taken = expected_remaining_time
                    time_lst = []
                    while time_taken >0 :
                        temp = time_taken % 60
                        time_taken = (time_taken - temp)//60
                        time_lst = [str(int(temp))] + time_lst
                    
                    printing_string = 'Completed:'+str(current_per)+' %'+'     Expected remaining time: '+":".join(time_lst)
                    print('\r' + ' ' * cur_printing_len + '\r', end='', flush=True)
                    print(printing_string,end='', flush=True)
                    cur_printing_len = len(printing_string)
    print('\n')
    #getting train data accuracy
    train_correct_count = 0
    train_total_count = 0
    for relation, list_of_bags in train_data.items():
        for bag in list_of_bags :
            for prompt_data in bag :
                train_total_count +=1
                input_prompt, target_word, relation_type, knowledge_neurons = prompt_data[0], prompt_data[1], prompt_data[2], prompt_data[3]

                arguments.get_pred = True
                arguments.get_ig_pred = False
                arguments.get_ig_gold = False
                arguments.get_base = False
                output_dict = get_attribution_scores(input_prompt,target_word,model,tokenizer,arguments,relation_type=relation_type)
                predicted_word = output_dict['pred_label']

                if predicted_word == target_word :
                    train_correct_count +=1
    
    # getting validation accuracy
    valid_correct_count = 0
    valid_total_count = 0
    for relation, list_of_bags in val_data.items():
        for bag in list_of_bags :
            for prompt_data in bag :
                valid_total_count +=1
                input_prompt, target_word, relation_type, knowledge_neurons = prompt_data[0], prompt_data[1], prompt_data[2], prompt_data[3]

                arguments.get_pred = True
                arguments.get_ig_pred = False
                arguments.get_ig_gold = False
                arguments.get_base = False
                output_dict = get_attribution_scores(input_prompt,target_word,model,tokenizer,arguments,relation_type=relation_type)
                predicted_word = output_dict['pred_label']

                if predicted_word == target_word :
                    valid_correct_count +=1

    return model, train_correct_count/train_total_count, valid_correct_count/valid_total_count
                
                


def edit_model_kn_method2(model,tokenizer,train_data,val_data,arguments,num_epochs,model_save_path=None, display=False):
    device = arguments.device
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)
    loss_func = torch.nn.CrossEntropyLoss()

    print('======== Editing model ========')
    min_valid_loss = float('inf')
    model_saved = False
    for epoch in range(num_epochs):
        total_train_loss = 0
        total_val_loss = 0
        total_train_count = 0
        total_val_count = 0
        for relation, list_of_bags in train_data.items():
            for bag in list_of_bags :
                if device.type == 'cuda' :
                    torch.cuda.empty_cache()
                if len(bag) == 0 :
                    continue
                
                model.train()
                optimizer.zero_grad()

                input_ids_lst = []
                baseline_ids_lst = []
                input_mask_lst = []
                segment_ids_lst = []
                tgt_pos_lst = []
                labels = []
                total_train_count += 1
                for prompt_data in bag :
                    input_prompt, target_word, relation_type, knowledge_neurons = prompt_data[0], prompt_data[1], prompt_data[2], prompt_data[3]

                    eval_features, tokens_info = example2feature([input_prompt,target_word,relation_type], arguments.max_seq_length, tokenizer)
                    # convert features to long type tensors
                    baseline_ids, input_ids, input_mask, segment_ids = eval_features['baseline_ids'], eval_features['input_ids'], eval_features['input_mask'], eval_features['segment_ids']
                    baseline_ids = torch.tensor(baseline_ids, dtype=torch.long).unsqueeze(0)
                    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
                    input_mask = torch.tensor(input_mask, dtype=torch.long).unsqueeze(0)
                    segment_ids = torch.tensor(segment_ids, dtype=torch.long).unsqueeze(0)
                    baseline_ids = baseline_ids.to(device)
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)



                    # record [MASK]'s position
                    tgt_pos = tokens_info['tokens'].index('[MASK]')

                    baseline_ids_lst.append(baseline_ids)
                    input_ids_lst.append(input_ids)
                    input_mask_lst.append(input_mask)
                    segment_ids_lst.append(segment_ids)
                    tgt_pos_lst.append(tgt_pos)
                    labels.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(target_word)[-1]))
                
                baseline_ids = torch.cat(baseline_ids_lst,dim=0)
                input_ids = torch.cat(input_ids_lst,dim=0)
                input_mask = torch.cat(input_mask_lst,dim=0)
                segment_ids = torch.cat(segment_ids_lst,dim=0)
                tgt_pos = torch.tensor(tgt_pos_lst,device=device)
                labels = torch.tensor(labels,device=device)

                logits = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=0,return_all_logits=True)  # (1, n_vocab)
                logits = logits[torch.arange(logits.shape[0],device=device),tgt_pos]
                loss = loss_func(logits,labels)
                loss.backward()

                total_train_loss += loss.item()
                
                # selceting gradients only for required neurons
                parameter_dict = {}
                for neuron in bag[0][-1] :
                    layer = neuron[0]
                    neuron_index = neuron[1]
                    parameter_dict[(layer,neuron_index)] = model.bert.encoder.layer[layer].output.dense.weight.grad.clone().detach()
                
                with torch.no_grad() :
                    for param in model.parameters():
                        if param.grad is not None:
                            param.grad.zero_()

                    for neuron, parameter_grad in parameter_dict.items():
                        layer = neuron[0]
                        neuron_index = neuron[1]
                        model.bert.encoder.layer[layer].output.dense.weight.grad[:,neuron_index] = parameter_grad[:,neuron_index]
                
                optimizer.step()

        total_train_loss = total_train_loss / total_train_count

        model.eval()
        for relation, list_of_bags in val_data.items():
            for bag in list_of_bags :
                if device.type == 'cuda' :
                    torch.cuda.empty_cache()
                if len(bag) == 0 :
                    continue

                input_ids_lst = []
                baseline_ids_lst = []
                input_mask_lst = []
                segment_ids_lst = []
                tgt_pos_lst = []
                labels = []
                total_val_count += 1
                for prompt_data in bag :
                    input_prompt, target_word, relation_type, knowledge_neurons = prompt_data[0], prompt_data[1], prompt_data[2], prompt_data[3]

                    eval_features, tokens_info = example2feature([input_prompt,target_word,relation_type], arguments.max_seq_length, tokenizer)
                    # convert features to long type tensors
                    baseline_ids, input_ids, input_mask, segment_ids = eval_features['baseline_ids'], eval_features['input_ids'], eval_features['input_mask'], eval_features['segment_ids']
                    baseline_ids = torch.tensor(baseline_ids, dtype=torch.long).unsqueeze(0)
                    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
                    input_mask = torch.tensor(input_mask, dtype=torch.long).unsqueeze(0)
                    segment_ids = torch.tensor(segment_ids, dtype=torch.long).unsqueeze(0)
                    baseline_ids = baseline_ids.to(device)
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)



                    # record [MASK]'s position
                    tgt_pos = tokens_info['tokens'].index('[MASK]')

                    baseline_ids_lst.append(baseline_ids)
                    input_ids_lst.append(input_ids)
                    input_mask_lst.append(input_mask)
                    segment_ids_lst.append(segment_ids)
                    tgt_pos_lst.append(tgt_pos)
                    labels.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(target_word)[-1]))
                
                baseline_ids = torch.cat(baseline_ids_lst,dim=0)
                input_ids = torch.cat(input_ids_lst,dim=0)
                input_mask = torch.cat(input_mask_lst,dim=0)
                segment_ids = torch.cat(segment_ids_lst,dim=0)
                tgt_pos = torch.tensor(tgt_pos_lst,device=device)
                labels = torch.tensor(labels,device=device)

                logits = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=0,return_all_logits=True)  # (1, n_vocab)
                logits = logits[torch.arange(logits.shape[0],device=device),tgt_pos]
                loss = loss_func(logits,labels)

                total_val_loss += loss.item()
        
        total_val_loss = total_val_loss / total_val_count

        print('epoch:',epoch)
        print('train loss:',total_train_loss)
        print('validation loss:',total_val_loss)
        if total_val_loss <= min_valid_loss :
            min_valid_loss = total_val_loss
            if model_save_path!=None :
                torch.save(model.state_dict(), model_save_path+'/kn_model_finetuned.pth')
                model_saved = True
                print('model saved')
    print('======== editing done ========')
    if model_save_path != None and model_saved:
        model.load_state_dict(torch.load(model_save_path+'/kn_model_finetuned.pth'))

    #getting train data accuracy
    train_correct_count = 0
    train_total_count = 0
    for relation, list_of_bags in train_data.items():
        for bag in list_of_bags :
            for prompt_data in bag :
                train_total_count +=1
                input_prompt, target_word, relation_type, knowledge_neurons = prompt_data[0], prompt_data[1], prompt_data[2], prompt_data[3]

                arguments.get_pred = True
                arguments.get_ig_pred = False
                arguments.get_ig_gold = False
                arguments.get_base = False
                output_dict = get_attribution_scores(input_prompt,target_word,model,tokenizer,arguments,relation_type=relation_type)
                predicted_word = output_dict['pred_label']

                if predicted_word == target_word :
                    train_correct_count +=1
    
    # getting validation accuracy
    valid_correct_count = 0
    valid_total_count = 0
    for relation, list_of_bags in val_data.items():
        for bag in list_of_bags :
            for prompt_data in bag :
                valid_total_count +=1
                input_prompt, target_word, relation_type, knowledge_neurons = prompt_data[0], prompt_data[1], prompt_data[2], prompt_data[3]

                arguments.get_pred = True
                arguments.get_ig_pred = False
                arguments.get_ig_gold = False
                arguments.get_base = False
                output_dict = get_attribution_scores(input_prompt,target_word,model,tokenizer,arguments,relation_type=relation_type)
                predicted_word = output_dict['pred_label']

                if predicted_word == target_word :
                    valid_correct_count +=1
    

    return model, train_correct_count/train_total_count, valid_correct_count/valid_total_count



def exctract_activations(model,input_dict,argumrnts,layer_idx = None ):
    device = argumrnts.device
    output_dict = {}
    output_all = False
    if layer_idx == None :
        output_all = True
    
    with torch.no_grad() :
        input_ids = input_dict.get('input_ids',None) 
        attention_mask = input_dict.get('attention_mask',None)
        token_type_ids = input_dict.get('token_type_ids',None)
        tgt_pos = input_dict.get('tgt_pos',None)
        tgt_layer = input_dict.get('tgt_layer',None)
        tmp_score = input_dict.get('tmp_score',None)
        imp_pos = input_dict.get('imp_pos',None)
        imp_op = input_dict.get('imp_op',None)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=next(model.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        #embedding layer
        embedding_output = model.bert.embeddings(input_ids, token_type_ids)
        
        #encoder layer
        hidden_states, attention_mask = embedding_output, extended_attention_mask

        all_encoder_layers = []
        ffn_weights = None
        if imp_op == 'return':
            imp_weights = []
        for layer_index, layer_module in enumerate(model.bert.encoder.layer):
            if imp_pos is not None:
                imp_pos_at_this_layer = [x for x in imp_pos if x[0] == layer_index]
            else:
                imp_pos_at_this_layer = None
            if imp_op == 'return':
                if tgt_layer == layer_index:
                    #hidden_states, ffn_weights, imp_weights_l = layer_module(hidden_states, attention_mask, tgt_pos=tgt_pos, tmp_score=tmp_score, imp_pos=imp_pos_at_this_layer, imp_op=imp_op)
                    attention_output, att_score = layer_module.attention(hidden_states, attention_mask)

                    intermediate_output, imp_weights = layer_module.intermediate(attention_output, tgt_pos=tgt_pos, tmp_score=tmp_score, imp_pos=imp_pos_at_this_layer, imp_op=imp_op)
                    if output_all or layer_index in layer_idx:
                        output_dict[layer_index] = intermediate_output

                    layer_output = layer_module.output(intermediate_output, attention_output)

                    hidden_states, ffn_weights, imp_weights_l = layer_output, intermediate_output, imp_weights


                else:
                    #hidden_states, _, imp_weights_l = layer_module(hidden_states, attention_mask, tgt_pos=tgt_pos, imp_pos=imp_pos_at_this_layer, imp_op=imp_op)
                    attention_output, att_score = layer_module.attention(hidden_states, attention_mask)

                    intermediate_output, imp_weights = layer_module.intermediate(attention_output, tgt_pos=tgt_pos, tmp_score=tmp_score, imp_pos=imp_pos_at_this_layer, imp_op=imp_op)
                    if output_all or layer_index in layer_idx:
                        output_dict[layer_index] = intermediate_output
                    layer_output = layer_module.output(intermediate_output, attention_output)

                    hidden_states, _, imp_weights_l = layer_output, intermediate_output, imp_weights

                imp_weights.extend(imp_weights_l)
            else:
                if tgt_layer == layer_index:
                    #hidden_states, ffn_weights = layer_module(hidden_states, attention_mask, tgt_pos=tgt_pos, tmp_score=tmp_score, imp_pos=imp_pos_at_this_layer, imp_op=imp_op)
                    attention_output, att_score = layer_module.attention(hidden_states, attention_mask)
                    intermediate_output = layer_module.intermediate(attention_output, tgt_pos=tgt_pos, tmp_score=tmp_score, imp_pos=imp_pos_at_this_layer, imp_op=imp_op)
                    if output_all or layer_index in layer_idx:
                        output_dict[layer_index] = intermediate_output
                    layer_output = layer_module.output(intermediate_output, attention_output)
                    hidden_states, ffn_weights = layer_output, intermediate_output
                else:
                    #hidden_states, _ = layer_module(hidden_states, attention_mask, tgt_pos=tgt_pos, imp_pos=imp_pos_at_this_layer, imp_op=imp_op)
                    attention_output, att_score = layer_module.attention(hidden_states, attention_mask)
                    intermediate_output = layer_module.intermediate(attention_output, tgt_pos=tgt_pos, tmp_score=tmp_score, imp_pos=imp_pos_at_this_layer, imp_op=imp_op)
                    if output_all or layer_index in layer_idx:
                        output_dict[layer_index] = intermediate_output
                    layer_output = layer_module.output(intermediate_output, attention_output)
                    hidden_states, _ = layer_output, intermediate_output
    return output_dict




class Optimizing_BERT_MODEL_BIRD(torch.nn.Module):
    def __init__(this,main_bert_model,target_layer,target_token_index,total_number_layers=12):
        super().__init__()
        this.main_bert_model = main_bert_model
        this.z_dim = main_bert_model.bert.encoder.layer[0].output.dense.weight.shape[0]
        this.z = torch.nn.Parameter(torch.randn(this.z_dim))
        this.target_layer = target_layer 
        this.target_token_index = target_token_index
        this.total_number_layers = total_number_layers
    
    def forward(this,input_ids, token_type_ids=None, attention_mask=None, tgt_pos=None, tgt_layer=None, tmp_score=None, tgt_label=None, imp_pos=None, imp_op=None, return_all_logits=True):
        if tmp_score is not None:
            batch_size = tmp_score.shape[0]
            input_ids = input_ids.repeat(batch_size, 1)
            token_type_ids = token_type_ids.repeat(batch_size, 1)
            attention_mask = attention_mask.repeat(batch_size, 1)
        
        if imp_op == 'return':
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            if token_type_ids is None:
                token_type_ids = torch.zeros_like(input_ids)
            
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            extended_attention_mask = extended_attention_mask.to(dtype=next(this.parameters()).dtype)  # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

            embedding_output = this.main_bert_model.bert.embeddings(input_ids, token_type_ids)

            hidden_states, attention_mask = embedding_output, extended_attention_mask

            all_encoder_layers = []
            ffn_weights = None
            if imp_op == 'return':
                imp_weights = []
            
            for layer_index in range(this.target_layer):
                layer_module = this.main_bert_model.bert.encoder.layer[layer_index]
                if imp_pos is not None:
                    imp_pos_at_this_layer = [x for x in imp_pos if x[0] == layer_index]
                else:
                    imp_pos_at_this_layer = None
                
                if tgt_layer == layer_index:
                    hidden_states, ffn_weights, imp_weights_l = layer_module(hidden_states, attention_mask, tgt_pos=tgt_pos, tmp_score=tmp_score, imp_pos=imp_pos_at_this_layer, imp_op=imp_op)
                else:
                    hidden_states, _, imp_weights_l = layer_module(hidden_states, attention_mask, tgt_pos=tgt_pos, imp_pos=imp_pos_at_this_layer, imp_op=imp_op)
                imp_weights.extend(imp_weights_l)

            layer_module = this.main_bert_model.bert.encoder.layer[this.target_layer]
            if imp_pos is not None:
                imp_pos_at_this_layer = [x for x in imp_pos if x[0] == layer_index]
            else:
                imp_pos_at_this_layer = None
            
            attention_output, att_score = layer_module.attention(hidden_states, attention_mask)
            intermediate_output, imp_weights = layer_module.intermediate(attention_output, tgt_pos=tgt_pos, tmp_score=tmp_score, imp_pos=imp_pos_at_this_layer, imp_op=imp_op)
            layer_output = layer_module.output(intermediate_output, attention_output)
            layer_output[0,this.target_token_index] = this.z

            hidden_states, ffn_weights, imp_weights_l = layer_output, intermediate_output, imp_weights 

            for layer_index in range(this.target_layer+1,this.total_number_layers):
                layer_module = this.main_bert_model.bert.encoder.layer[layer_index]
                if imp_pos is not None:
                    imp_pos_at_this_layer = [x for x in imp_pos if x[0] == layer_index]
                else:
                    imp_pos_at_this_layer = None
                
                if tgt_layer == layer_index:
                    hidden_states, ffn_weights, imp_weights_l = layer_module(hidden_states, attention_mask, tgt_pos=tgt_pos, tmp_score=tmp_score, imp_pos=imp_pos_at_this_layer, imp_op=imp_op)
                else:
                    hidden_states, _, imp_weights_l = layer_module(hidden_states, attention_mask, tgt_pos=tgt_pos, imp_pos=imp_pos_at_this_layer, imp_op=imp_op)
                imp_weights.extend(imp_weights_l)

            all_encoder_layers.append(hidden_states)
            encoded_layers, ffn_weights, imp_weights = all_encoder_layers, ffn_weights, imp_weights
            sequence_output = encoded_layers[-1]

            last_hidden, ffn_weights, imp_weights = sequence_output, ffn_weights, imp_weights 

            return this.main_bert_model.cls(last_hidden)
        else:
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            if token_type_ids is None:
                token_type_ids = torch.zeros_like(input_ids)
            
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            extended_attention_mask = extended_attention_mask.to(dtype=next(this.parameters()).dtype)  # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

            embedding_output = this.main_bert_model.bert.embeddings(input_ids, token_type_ids)

            hidden_states, attention_mask = embedding_output, extended_attention_mask

            all_encoder_layers = []
            ffn_weights = None
            for layer_index in range(this.target_layer):
                layer_module = this.main_bert_model.bert.encoder.layer[layer_index]
                if imp_pos is not None:
                    imp_pos_at_this_layer = [x for x in imp_pos if x[0] == layer_index]
                else:
                    imp_pos_at_this_layer = None
                if tgt_layer == layer_index:
                    hidden_states, ffn_weights = layer_module(hidden_states, attention_mask, tgt_pos=tgt_pos, tmp_score=tmp_score, imp_pos=imp_pos_at_this_layer, imp_op=imp_op)
                else:
                    hidden_states, _ = layer_module(hidden_states, attention_mask, tgt_pos=tgt_pos, imp_pos=imp_pos_at_this_layer, imp_op=imp_op)
            
            layer_module = this.main_bert_model.bert.encoder.layer[this.target_layer]
            if imp_pos is not None:
                imp_pos_at_this_layer = [x for x in imp_pos if x[0] == layer_index]
            else:
                imp_pos_at_this_layer = None
            
            attention_output, att_score = layer_module.attention(hidden_states, attention_mask)
            intermediate_output = layer_module.intermediate(attention_output, tgt_pos=tgt_pos, tmp_score=tmp_score, imp_pos=imp_pos, imp_op=imp_op)
            layer_output = layer_module.output(intermediate_output, attention_output)
            layer_output[0,this.target_token_index] = this.z
            hidden_states, ffn_weights = layer_output, intermediate_output

            for layer_index in range(this.target_layer+1,this.total_number_layers):
                layer_module = this.main_bert_model.bert.encoder.layer[layer_index]
                if imp_pos is not None:
                    imp_pos_at_this_layer = [x for x in imp_pos if x[0] == layer_index]
                else:
                    imp_pos_at_this_layer = None
                if tgt_layer == layer_index:
                    hidden_states, ffn_weights = layer_module(hidden_states, attention_mask, tgt_pos=tgt_pos, tmp_score=tmp_score, imp_pos=imp_pos_at_this_layer, imp_op=imp_op)
                else:
                    hidden_states, _ = layer_module(hidden_states, attention_mask, tgt_pos=tgt_pos, imp_pos=imp_pos_at_this_layer, imp_op=imp_op)
           
            all_encoder_layers.append(hidden_states)

            encoded_layers, ffn_weights = all_encoder_layers, ffn_weights
            sequence_output = encoded_layers[-1]
            last_hidden, ffn_weights = sequence_output, ffn_weights
            return this.main_bert_model.cls(last_hidden)






def get_z(model,tokenizer,prompt_data,num_epochs,arguments):
    device = arguments.device
    n_gpu = arguments.n_gpu
    model.train()
    model = model.to(device)
    optimizer = torch.optim.Adam([model.z],lr = 0.5, weight_decay = 0.0015)
    neuron_data = prompt_data[-1]
    subject_tuple = tuple(neuron_data[-2])
    eval_example = [prompt_data[0],prompt_data[1],prompt_data[2]]
    eval_features, tokens_info = example2feature(eval_example, arguments.max_seq_length, tokenizer)
    # convert features to long type tensors
    baseline_ids, input_ids, input_mask, segment_ids = eval_features['baseline_ids'], eval_features['input_ids'], eval_features['input_mask'], eval_features['segment_ids']
    baseline_ids = torch.tensor(baseline_ids, dtype=torch.long).unsqueeze(0)
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    input_mask = torch.tensor(input_mask, dtype=torch.long).unsqueeze(0)
    segment_ids = torch.tensor(segment_ids, dtype=torch.long).unsqueeze(0)
    baseline_ids = baseline_ids.to(device)
    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)
    segment_ids = segment_ids.to(device)

    # record real input length
    input_len = int(input_mask[0].sum())
    tgt_pos = None
    label_pos = tokens_info['tokens'].index('[MASK]')
    label = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(prompt_data[1])[-1])

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        model_output = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=0)
        log_prob = F.log_softmax(model_output[0][label_pos])
        loss = -1*log_prob[label]
        loss.backward()

        optimizer.step()
    return model.z.data
        



def edit_model_BIRD(model,tokenizer,train_data,val_data,arguments,num_epochs=20,model_save_path=None, display=False):

    device = arguments.device
    n_gpu = arguments.n_gpu
    args = arguments
    
    #extracting second moment statistics
    wikipedia_data = load_dataset("wikipedia", "20220301.en")
    wiki_article_index = list(range(len(wikipedia_data['train'])))
    random.shuffle(wiki_article_index)
    second_moment_stat = {}
    total_k_count = 0
    prompt_count = 0
    cur_printing_len = 0
    total_prompts = 100000
    per_st = time.time()
    print('======== extracting second order moments ========')
    for article_index in wiki_article_index :
        article_data = wikipedia_data['train'][article_index]['text'].split('\n')
        
        input_ids_lst = []
        baseline_ids_lst = []
        input_mask_lst = []
        segment_ids_lst = []
        labels = []
        input_len_lst = []
        for input_prompt in article_data :
            eval_example = [input_prompt,None,None]
            eval_features, tokens_info = example2feature(eval_example, args.max_seq_length, tokenizer)
            # convert features to long type tensors
            baseline_ids, input_ids, input_mask, segment_ids = eval_features['baseline_ids'], eval_features['input_ids'], eval_features['input_mask'], eval_features['segment_ids']
            baseline_ids = torch.tensor(baseline_ids, dtype=torch.long).unsqueeze(0)
            input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
            input_mask = torch.tensor(input_mask, dtype=torch.long).unsqueeze(0)
            segment_ids = torch.tensor(segment_ids, dtype=torch.long).unsqueeze(0)
            baseline_ids = baseline_ids.to(device)
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)

            # record real input length
            input_len = int(input_mask[0].sum())

            baseline_ids_lst.append(baseline_ids)
            input_ids_lst.append(input_ids)
            input_mask_lst.append(input_mask)
            segment_ids_lst.append(segment_ids)
            input_len_lst.append(input_len)

        baseline_ids = torch.cat(baseline_ids_lst,dim=0)
        input_ids = torch.cat(input_ids_lst,dim=0)
        input_mask = torch.cat(input_mask_lst,dim=0)
        segment_ids = torch.cat(segment_ids_lst,dim=0)
        tgt_pos = None
        #input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=0
        input_dict = {'input_ids':input_ids,'attention_mask':input_mask,'token_type_ids':segment_ids,'tgt_pos':tgt_pos,'tgt_layer':0}
        activations = exctract_activations(model,input_dict,arguments)
        total_k_count += activations[0].view(-1,activations[0].shape[-1]).shape[0]
        prompt_count += total_k_count
        if prompt_count >= 100000 :
            prompt_count = 100000
        for k, v in activations.items():
            k_mat = v.view(-1,v.shape[-1])
            if total_k_count >= 100000 :
                extra_vects = total_k_count - 100000
                if extra_vects > 0 :
                    k_mat = k_mat[:-extra_vects]
            som_stat = (1/100) * (k_mat.t() @ k_mat)
            temp_som = second_moment_stat.get(k,None)
            if temp_som == None :
                second_moment_stat[k] = som_stat
            else:
                second_moment_stat[k] += som_stat
        
        current_per = round((prompt_count/total_prompts) * 100,2)
        if display : #current_per > current_per_complited and display
            per_et = time.time()
            diff_time = per_et - per_st
            per_st = time.time()
            expected_remaining_time = diff_time * (total_prompts - prompt_count)
            time_lst = []
            time_taken = expected_remaining_time
            time_lst = []
            while time_taken >0 :
                temp = time_taken % 60
                time_taken = (time_taken - temp)//60
                time_lst = [str(int(temp))] + time_lst
            
            printing_string = 'Completed:'+str(current_per)+' %'+'     Expected remaining time: '+":".join(time_lst)
            print('\r' + ' ' * cur_printing_len + '\r', end='', flush=True)
            print(printing_string,end='', flush=True)
            cur_printing_len = len(printing_string)
        
        if total_k_count >= 100000 :
            break
    if display :
        print(' ')
        print('Done')
        print(' ')
    for k, v in second_moment_stat.items():
        second_moment_stat[k] = torch.inverse((1/1000) * v)
    #print(second_moment_stat[0],len(second_moment_stat))
    if device.type == 'cuda' :
        torch.cuda.empty_cache()
    #computing k* for all subject tokens
    print('======= computing k* for all subjects ========')
    prompt_count = 0
    cur_printing_len = 0
    #
    total_prompts = 0
    for relation, list_of_bags in train_data.items():
        for bag in list_of_bags :
            for prompt_data in bag :
                total_prompts+=1
    #
    subject_tokens_data = {}
    for relation, list_of_bags in train_data.items():
        for bag in list_of_bags:
            per_st = time.time()
            for prompt_data in bag:
                prompt_count+=1
                neuron_data = prompt_data[-1]
                subject_tuple = tuple(neuron_data[-2])
                eval_example = [prompt_data[0],None,None]
                eval_features, tokens_info = example2feature(eval_example, args.max_seq_length, tokenizer)
                # convert features to long type tensors
                baseline_ids, input_ids, input_mask, segment_ids = eval_features['baseline_ids'], eval_features['input_ids'], eval_features['input_mask'], eval_features['segment_ids']
                baseline_ids = torch.tensor(baseline_ids, dtype=torch.long).unsqueeze(0)
                input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
                input_mask = torch.tensor(input_mask, dtype=torch.long).unsqueeze(0)
                segment_ids = torch.tensor(segment_ids, dtype=torch.long).unsqueeze(0)
                baseline_ids = baseline_ids.to(device)
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)

                # record real input length
                input_len = int(input_mask[0].sum())
                tgt_pos = None
                input_dict = {'input_ids':input_ids,'attention_mask':input_mask,'token_type_ids':segment_ids,'tgt_pos':tgt_pos,'tgt_layer':0}
                activations = exctract_activations(model,input_dict,arguments)
                subject_data_dict = subject_tokens_data.get(subject_tuple,dict())
                for subject_indices in neuron_data[-1] :
                    last_index = subject_indices[-1]
                    for layer_index, act in activations.items():
                        cur_subject_token = act[-1][last_index]
                        temp_subject_data = subject_data_dict.get(layer_index,None)
                        if temp_subject_data == None :
                            subject_data_dict[layer_index] = [cur_subject_token,1]
                        else:
                            subject_data_dict[layer_index] = [temp_subject_data[0]+cur_subject_token,temp_subject_data[1]+1]
                subject_tokens_data[subject_tuple] = subject_data_dict

                current_per = round((prompt_count/total_prompts) * 100,2)
                if display : #current_per > current_per_complited and display
                    per_et = time.time()
                    diff_time = per_et - per_st
                    per_st = time.time()
                    expected_remaining_time = diff_time * (total_prompts - prompt_count)
                    time_lst = []
                    time_taken = expected_remaining_time
                    time_lst = []
                    while time_taken >0 :
                        temp = time_taken % 60
                        time_taken = (time_taken - temp)//60
                        time_lst = [str(int(temp))] + time_lst
                    
                    printing_string = 'Completed:'+str(current_per)+' %'+'     Expected remaining time: '+":".join(time_lst)
                    print('\r' + ' ' * cur_printing_len + '\r', end='', flush=True)
                    print(printing_string,end='', flush=True)
                    cur_printing_len = len(printing_string)
                
    if display :
        print(' ')
        print('Done')
        print(' ')
    for subject_tuple, data_dict in subject_tokens_data.items():
        temp_dict = dict()
        for layer_index, token_data in data_dict.items():
            temp_dict[layer_index] = token_data[0]/token_data[1]
        subject_tokens_data[subject_tuple] = temp_dict
    if device.type == 'cuda' :
        torch.cuda.empty_cache()
    
    #editing model
    print('======= editing model ========')
    prompt_count = 0
    cur_printing_len = 0

    for relation, list_of_bags in train_data.items():
        for bag in list_of_bags:
            per_st = time.time()
            for prompt_data in bag:
                prompt_count+=1
                if device.type == 'cuda' :
                    torch.cuda.empty_cache()
                tokenzed_subject_data = prompt_data[-1]
                sybject_tuple = tuple(tokenzed_subject_data[-2])
                for neuron_data in prompt_data[3] :
                    layer_index = neuron_data[0]
                    k = subject_tokens_data[subject_tuple][layer_index]
                    temp_1 = (second_moment_stat[layer_index] @ k.unsqueeze(1)).t()
                    for subject_loc_data in tokenzed_subject_data[-1]:
                        subject_loc = subject_loc_data[-1]
                        new_model = Optimizing_BERT_MODEL_BIRD(main_bert_model=model,target_layer=layer_index,target_token_index=subject_loc)
                        v = get_z(new_model,tokenizer,prompt_data,num_epochs,arguments)
                        mlp_layer = model.bert.encoder.layer[layer_index].output.dense
                        mlp_weights = mlp_layer.weight.data.clone()
                        lambda_1 = (v.unsqueeze(1) - (mlp_weights@k.unsqueeze(1)))/(torch.sum(temp_1[0]*k))
                        mlp_layer.weight.data = mlp_weights + (lambda_1 @ temp_1)
                current_per = round((prompt_count/total_prompts) * 100,2)
                if display : #current_per > current_per_complited and display
                    per_et = time.time()
                    diff_time = per_et - per_st
                    per_st = time.time()
                    expected_remaining_time = diff_time * (total_prompts - prompt_count)
                    time_lst = []
                    time_taken = expected_remaining_time
                    time_lst = []
                    while time_taken >0 :
                        temp = time_taken % 60
                        time_taken = (time_taken - temp)//60
                        time_lst = [str(int(temp))] + time_lst
                    
                    printing_string = 'Completed:'+str(current_per)+' %'+'     Expected remaining time: '+":".join(time_lst)
                    print('\r' + ' ' * cur_printing_len + '\r', end='', flush=True)
                    print(printing_string,end='', flush=True)
                    cur_printing_len = len(printing_string)
    if display :
        print(' ')
        print('Done')
        print(' ')
    #getting train data accuracy
    train_correct_count = 0
    train_total_count = 0
    for relation, list_of_bags in train_data.items():
        for bag in list_of_bags :
            for prompt_data in bag :
                train_total_count +=1
                input_prompt, target_word, relation_type, knowledge_neurons = prompt_data[0], prompt_data[1], prompt_data[2], prompt_data[3]

                arguments.get_pred = True
                arguments.get_ig_pred = False
                arguments.get_ig_gold = False
                arguments.get_base = False
                output_dict = get_attribution_scores(input_prompt,target_word,model,tokenizer,arguments,relation_type=relation_type)
                predicted_word = output_dict['pred_label']

                if predicted_word == target_word :
                    train_correct_count +=1
    
    # getting validation accuracy
    valid_correct_count = 0
    valid_total_count = 0
    for relation, list_of_bags in val_data.items():
        for bag in list_of_bags :
            for prompt_data in bag :
                valid_total_count +=1
                input_prompt, target_word, relation_type, knowledge_neurons = prompt_data[0], prompt_data[1], prompt_data[2], prompt_data[3]

                arguments.get_pred = True
                arguments.get_ig_pred = False
                arguments.get_ig_gold = False
                arguments.get_base = False
                output_dict = get_attribution_scores(input_prompt,target_word,model,tokenizer,arguments,relation_type=relation_type)
                predicted_word = output_dict['pred_label']

                if predicted_word == target_word :
                    valid_correct_count +=1
    
    relations = list(train_data.keys())
    relations = [str(i) for i in relations]

    if model_save_path != None :
        model.load_state_dict(torch.load(model_save_path+'/model_edited_BIRD_'+','.join(relations)+'_.pth'))

    return model, train_correct_count/train_total_count, valid_correct_count/valid_total_count



class Optimizing_BERT_MODEL_T_PATCHER(torch.nn.Module):
    def __init__(this,main_bert_model,target_layer = None,total_number_layers=12):
        super().__init__()
        this.main_bert_model = main_bert_model
        this.target_layer = target_layer 
        if this.target_layer == None :
            this.target_layer = [total_number_layers - 1]
        this.total_number_layers = total_number_layers
    
    
    def forward(this,input_ids, pos_samples_len, neg_samples_len, token_type_ids=None, attention_mask=None, tgt_pos=None, tgt_layer=None, tmp_score=None, tgt_label=None, imp_pos=None, imp_op=None, return_all_logits=True):
        pos_tgt_pos = tgt_pos[:pos_samples_len]
        neg_tgt_pos = tgt_pos[pos_samples_len:]

        if tmp_score is not None:
            batch_size = tmp_score.shape[0]
            input_ids = input_ids.repeat(batch_size, 1)
            token_type_ids = token_type_ids.repeat(batch_size, 1)
            attention_mask = attention_mask.repeat(batch_size, 1)
        
        input_quiry_dict = dict()
        #negative_quiry_dict = dict()
        #diff_quiry_dict = dict()

        torch.cuda.empty_cache()
        
        if imp_op == 'return':
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            if token_type_ids is None:
                token_type_ids = torch.zeros_like(input_ids)
            
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            extended_attention_mask = extended_attention_mask.to(dtype=next(this.parameters()).dtype)  # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

            embedding_output = this.main_bert_model.bert.embeddings(input_ids, token_type_ids)

            hidden_states, attention_mask = embedding_output, extended_attention_mask

            all_encoder_layers = []
            ffn_weights = None
            if imp_op == 'return':
                imp_weights = []
            
            for layer_index in range(this.total_number_layers):
                layer_module = this.main_bert_model.bert.encoder.layer[layer_index]
                if imp_pos is not None:
                    imp_pos_at_this_layer = [x for x in imp_pos if x[0] == layer_index]
                else:
                    imp_pos_at_this_layer = None
                
                if tgt_layer == layer_index:
                    #hidden_states, ffn_weights, imp_weights_l = layer_module(hidden_states, attention_mask, tgt_pos=tgt_pos, tmp_score=tmp_score, imp_pos=imp_pos_at_this_layer, imp_op=imp_op)
                    if imp_pos is not None:
                        imp_pos_at_this_layer = [x for x in imp_pos if x[0] == layer_index]
                    else:
                        imp_pos_at_this_layer = None
                    
                    attention_output, att_score = layer_module.attention(hidden_states, attention_mask)

                    intermediate_output, imp_weights, dense_layer_output = layer_module.intermediate(attention_output, tgt_pos=tgt_pos, tmp_score=tmp_score, imp_pos=imp_pos_at_this_layer, imp_op=imp_op,dense_layer_output = True)
                    if layer_index in this.target_layer :
                        input_quiry_dict[layer_index] = dense_layer_output[:pos_samples_len].clone().detach()
                        '''
                        negative_quiry_dict[layer_index] = dense_layer_output[pos_samples_len:].clone().detach()
                        pos_input_quiries = attention_output[torch.arange(pos_samples_len,device=input_ids.device),pos_tgt_pos]
                        neg_input_quiries = attention_output[torch.arange(pos_samples_len,attention_output.shape[0],device=input_ids.device),neg_tgt_pos]
                        results = pos_input_quiries.unsqueeze(1) - neg_input_quiries.unsqueeze(0)
                        results = results.view(-1,results.shape[-1])
                        diff_quiry_output = layer_module.intermediate.dense(results)
                        diff_quiry_dict[layer_index] = diff_quiry_output
                        '''

                        
                    layer_output = layer_module.output(intermediate_output, attention_output)

                    hidden_states, ffn_weights, imp_weights_l = layer_output, intermediate_output, imp_weights 
                        
                
                else:
                    #hidden_states, _, imp_weights_l = layer_module(hidden_states, attention_mask, tgt_pos=tgt_pos, imp_pos=imp_pos_at_this_layer, imp_op=imp_op)
                    if imp_pos is not None:
                        imp_pos_at_this_layer = [x for x in imp_pos if x[0] == layer_index]
                    else:
                        imp_pos_at_this_layer = None
                    
                    attention_output, att_score = layer_module.attention(hidden_states, attention_mask)

                    intermediate_output, imp_weights, dense_layer_output = layer_module.intermediate(attention_output, tgt_pos=tgt_pos, tmp_score=tmp_score, imp_pos=imp_pos_at_this_layer, imp_op=imp_op, dense_layer_output = True)
                    if layer_index in this.target_layer :
                        input_quiry_dict[layer_index] = dense_layer_output[:pos_samples_len].clone().detach()
                        '''
                        negative_quiry_dict[layer_index] = dense_layer_output[pos_samples_len:].clone().detach()
                        pos_input_quiries = attention_output[torch.arange(pos_samples_len,device=input_ids.device),pos_tgt_pos]
                        neg_input_quiries = attention_output[torch.arange(pos_samples_len,attention_output.shape[0],device=input_ids.device),neg_tgt_pos]
                        results = pos_input_quiries.unsqueeze(1) - neg_input_quiries.unsqueeze(0)
                        results = results.view(-1,results.shape[-1])
                        diff_quiry_output = layer_module.intermediate.dense(results)
                        diff_quiry_dict[layer_index] = diff_quiry_output
                        '''
                    layer_output = layer_module.output(intermediate_output, attention_output)

                    hidden_states, _, imp_weights_l = layer_output, intermediate_output, imp_weights 
                imp_weights.extend(imp_weights_l)

            all_encoder_layers.append(hidden_states)
            encoded_layers, ffn_weights, imp_weights = all_encoder_layers, ffn_weights, imp_weights
            sequence_output = encoded_layers[-1]

            last_hidden, ffn_weights, imp_weights = sequence_output, ffn_weights, imp_weights 

            return this.main_bert_model.cls(last_hidden)
        else:
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            if token_type_ids is None:
                token_type_ids = torch.zeros_like(input_ids)
            
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            extended_attention_mask = extended_attention_mask.to(dtype=next(this.parameters()).dtype)  # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

            embedding_output = this.main_bert_model.bert.embeddings(input_ids, token_type_ids)

            hidden_states, attention_mask = embedding_output, extended_attention_mask

            all_encoder_layers = []
            ffn_weights = None
            for layer_index in range(this.total_number_layers):
                layer_module = this.main_bert_model.bert.encoder.layer[layer_index]
                if imp_pos is not None:
                    imp_pos_at_this_layer = [x for x in imp_pos if x[0] == layer_index]
                else:
                    imp_pos_at_this_layer = None
                if tgt_layer == layer_index:
                    #hidden_states, ffn_weights = layer_module(hidden_states, attention_mask, tgt_pos=tgt_pos, tmp_score=tmp_score, imp_pos=imp_pos_at_this_layer, imp_op=imp_op)

                    if imp_pos is not None:
                        imp_pos_at_this_layer = [x for x in imp_pos if x[0] == layer_index]
                    else:
                        imp_pos_at_this_layer = None
                    
                    attention_output, att_score = layer_module.attention(hidden_states, attention_mask)

                    intermediate_output, dense_layer_output = layer_module.intermediate(attention_output, tgt_pos=tgt_pos, tmp_score=tmp_score, imp_pos=imp_pos, imp_op=imp_op, dense_layer_output = True)
                    if layer_index in this.target_layer :
                        input_quiry_dict[layer_index] = dense_layer_output[:pos_samples_len].clone().detach()
                        '''
                        negative_quiry_dict[layer_index] = dense_layer_output[pos_samples_len:].clone().detach()
                        pos_input_quiries = attention_output[torch.arange(pos_samples_len,device=input_ids.device),pos_tgt_pos]
                        neg_input_quiries = attention_output[torch.arange(pos_samples_len,attention_output.shape[0],device=input_ids.device),neg_tgt_pos]
                        results = pos_input_quiries.unsqueeze(1) - neg_input_quiries.unsqueeze(0)
                        results = results.view(-1,results.shape[-1])
                        diff_quiry_output = layer_module.intermediate.dense(results)
                        diff_quiry_dict[layer_index] = diff_quiry_output
                        '''
                    layer_output = layer_module.output(intermediate_output, attention_output)
                    hidden_states, ffn_weights = layer_output, intermediate_output    
            
                else:
                    #hidden_states, _ = layer_module(hidden_states, attention_mask, tgt_pos=tgt_pos, imp_pos=imp_pos_at_this_layer, imp_op=imp_op)
                    if imp_pos is not None:
                        imp_pos_at_this_layer = [x for x in imp_pos if x[0] == layer_index]
                    else:
                        imp_pos_at_this_layer = None
                    
                    attention_output, att_score = layer_module.attention(hidden_states, attention_mask)

                    intermediate_output, dense_layer_output = layer_module.intermediate(attention_output, tgt_pos=tgt_pos, tmp_score=tmp_score, imp_pos=imp_pos, imp_op=imp_op, dense_layer_output = True)
                    if layer_index in this.target_layer :
                        input_quiry_dict[layer_index] = dense_layer_output[:pos_samples_len].clone().detach()
                        '''
                        negative_quiry_dict[layer_index] = dense_layer_output[pos_samples_len:].clone().detach()
                        pos_input_quiries = attention_output[torch.arange(pos_samples_len,device=input_ids.device),pos_tgt_pos]
                        neg_input_quiries = attention_output[torch.arange(pos_samples_len,attention_output.shape[0],device=input_ids.device),neg_tgt_pos]
                        results = pos_input_quiries.unsqueeze(1) - neg_input_quiries.unsqueeze(0)
                        results = results.view(-1,results.shape[-1])
                        diff_quiry_output = layer_module.intermediate.dense(results)
                        diff_quiry_dict[layer_index] = diff_quiry_output
                        '''
                    layer_output = layer_module.output(intermediate_output, attention_output)
                    hidden_states, _ = layer_output, intermediate_output
           
            all_encoder_layers.append(hidden_states)

            encoded_layers, ffn_weights = all_encoder_layers, ffn_weights
            sequence_output = encoded_layers[-1]
            last_hidden, ffn_weights = sequence_output, ffn_weights
            return this.main_bert_model.cls(last_hidden), input_quiry_dict#, negative_quiry_dict, diff_quiry_dict




def activation_loss(layer_output,tgt_pos,neurons):
    return torch.exp(-layer_output[torch.arange(layer_output.shape[0],device=layer_output.device),tgt_pos][:,neurons]).sum()





class Neg_samples:
    def __init__(this,data,not_relation,not_bag_index):
        this.samples = []
        for relation, list_of_bags in data.items():
            for bag_idx, bag in enumerate(list_of_bags) :
                if (relation != not_relation) or (relation == not_relation and bag_idx != not_bag_index):
                    for data_point in bag :
                        this.samples.append(data_point)
    
    def sample(this,num_samples,tokenizer,arguments):
        num_samples = min(num_samples,len(this.samples))
        device = arguments.device
        input_ids_lst = []
        baseline_ids_lst = []
        input_mask_lst = []
        segment_ids_lst = []
        tgt_pos_lst = []
        labels = []
        for prompt_data in random.sample(this.samples,num_samples) :
            input_prompt, target_word, relation_type, knowledge_neurons = prompt_data[0], prompt_data[1], prompt_data[2], prompt_data[3]

            eval_features, tokens_info = example2feature([input_prompt,target_word,relation_type], arguments.max_seq_length, tokenizer)
            # convert features to long type tensors
            baseline_ids, input_ids, input_mask, segment_ids = eval_features['baseline_ids'], eval_features['input_ids'], eval_features['input_mask'], eval_features['segment_ids']
            baseline_ids = torch.tensor(baseline_ids, dtype=torch.long).unsqueeze(0)
            input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
            input_mask = torch.tensor(input_mask, dtype=torch.long).unsqueeze(0)
            segment_ids = torch.tensor(segment_ids, dtype=torch.long).unsqueeze(0)
            baseline_ids = baseline_ids.to(device)
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)



            # record [MASK]'s position
            tgt_pos = tokens_info['tokens'].index('[MASK]')

            baseline_ids_lst.append(baseline_ids)
            input_ids_lst.append(input_ids)
            input_mask_lst.append(input_mask)
            segment_ids_lst.append(segment_ids)
            tgt_pos_lst.append(tgt_pos)
            labels.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(target_word)[-1]))
        
        if num_samples == 0 :
            return {'neg_baseline_ids':torch.tensor(baseline_ids_lst),'neg_input_ids':torch.tensor(input_ids_lst),'neg_input_mask':torch.tensor(input_mask_lst),'neg_segment_ids':torch.tensor(segment_ids_lst),'neg_labels':torch.tensor(labels),'neg_tgt_pos':torch.tensor(tgt_pos_lst)}
        
        baseline_ids = torch.cat(baseline_ids_lst,dim=0)
        input_ids = torch.cat(input_ids_lst,dim=0)
        input_mask = torch.cat(input_mask_lst,dim=0)
        segment_ids = torch.cat(segment_ids_lst,dim=0)
        tgt_pos = torch.tensor(tgt_pos_lst,device=device)
        labels = torch.tensor(labels,device=device)
        return {'neg_baseline_ids':baseline_ids,'neg_input_ids':input_ids,'neg_input_mask':input_mask,'neg_segment_ids':segment_ids,'neg_labels':labels,'neg_tgt_pos':tgt_pos}
    






def memory_loss(neg_quiries_output,diff_output_quiry,neg_tgt_pos,neurons,k, beta= -3,gamma = 3):
    lm1 = torch.exp((neg_quiries_output[:,neg_tgt_pos][:,neurons]).view(-1) - beta)
    top_lm1_indices = torch.sort(lm1).indices[-k:]
    lm1 = torch.mean(lm1[top_lm1_indices])
    lm2 = torch.exp(diff_output_quiry[:,neurons].view(-1) - gamma)
    top_lm2_indices = torch.sort(lm2).indices[-k:]
    lm2 = torch.mean(lm2[top_lm2_indices])
    return lm1+lm2



def convert_raw_batch_to_tensor(data_bag,tokenizer,arguments):
    
    device = arguments.device
    input_ids_lst = []
    baseline_ids_lst = []
    input_mask_lst = []
    segment_ids_lst = []
    tgt_pos_lst = []
    labels = []
    for prompt_data in data_bag :
        input_prompt, target_word, relation_type, knowledge_neurons = prompt_data[0], prompt_data[1], prompt_data[2], prompt_data[3]

        eval_features, tokens_info = example2feature([input_prompt,target_word,relation_type], arguments.max_seq_length, tokenizer)
        # convert features to long type tensors
        baseline_ids, input_ids, input_mask, segment_ids = eval_features['baseline_ids'], eval_features['input_ids'], eval_features['input_mask'], eval_features['segment_ids']
        baseline_ids = torch.tensor(baseline_ids, dtype=torch.long).unsqueeze(0)
        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        input_mask = torch.tensor(input_mask, dtype=torch.long).unsqueeze(0)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long).unsqueeze(0)
        baseline_ids = baseline_ids.to(device)
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)



        # record [MASK]'s position
        tgt_pos = tokens_info['tokens'].index('[MASK]')

        baseline_ids_lst.append(baseline_ids)
        input_ids_lst.append(input_ids)
        input_mask_lst.append(input_mask)
        segment_ids_lst.append(segment_ids)
        tgt_pos_lst.append(tgt_pos)
        labels.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(target_word)[-1]))
    
    if len(baseline_ids_lst) == 0 :
        return {'baseline_ids':torch.tensor(baseline_ids_lst),'input_ids':torch.tensor(input_ids_lst),'input_mask':torch.tensor(input_mask_lst),'segment_ids':torch.tensor(segment_ids_lst),'labels':torch.tensor(labels),'tgt_pos':torch.tensor(tgt_pos_lst)}
    
    baseline_ids = torch.cat(baseline_ids_lst,dim=0)
    input_ids = torch.cat(input_ids_lst,dim=0)
    input_mask = torch.cat(input_mask_lst,dim=0)
    segment_ids = torch.cat(segment_ids_lst,dim=0)
    tgt_pos = torch.tensor(tgt_pos_lst,device=device)
    labels = torch.tensor(labels,device=device)
    return {'baseline_ids':baseline_ids,'input_ids':input_ids,'input_mask':input_mask,'segment_ids':segment_ids,'labels':labels,'tgt_pos':tgt_pos}




def add_patch(model,tokenizer,train_data_bag,valid_data_bag,neg_sampler,arguments,act_loss_const = 1,memory_loss_const = 10,memory_k = 1000,neurons_list=None,num_patches=1,num_epochs=20,model_save_path=None, display=False):
    if display :
        print('======== inside add_patch ========')
    device = arguments.device
    n_gpu = arguments.n_gpu
    if neurons_list == None :
        neurons_list = torch.tensor([-i for i in range(1,num_patches+1)],device=device)
        #introducing new patches
        a = model.bert.encoder.layer[-1].intermediate.dense
        b = torch.nn.Linear(a.weight.data.shape[1],a.weight.data.shape[0]+num_patches).to(device)
        with torch.no_grad() :
            b.weight[:-num_patches] = a.weight.data.clone()
            b.bias[:-num_patches] = a.bias.data.clone()
        model.bert.encoder.layer[-1].intermediate.dense = b
        c = model.bert.encoder.layer[-1].output.dense
        d = torch.nn.Linear(c.weight.data.shape[1]+num_patches,c.weight.data.shape[0]).to(device)
        with torch.no_grad() :
            d.weight[:,:-num_patches] = c.weight.data.clone()
            d.bias[:] = c.bias.data.clone()
        model.bert.encoder.layer[-1].output.dense = d
    
    if display :
        print('neurons introduced')
        print('intermediate layer:',model.bert.encoder.layer[-1].intermediate.dense.weight.data.shape)
        print('intermediate layer:',model.bert.encoder.layer[-1].intermediate.dense.bias.data.shape)
        print('output layer:',model.bert.encoder.layer[-1].output.dense.weight.data.shape)
        print('output layer:',model.bert.encoder.layer[-1].output.dense.bias.data.shape)
        print(' ')
    
    #initialising T-Patcher optimizing model
    model_t_patcher = Optimizing_BERT_MODEL_T_PATCHER(model).to(device)
    optimizer = torch.optim.Adam(model_t_patcher.parameters(),lr = 0.001)
    edit_loss = torch.nn.CrossEntropyLoss()
    
    input_ids_lst = []
    baseline_ids_lst = []
    input_mask_lst = []
    segment_ids_lst = []
    tgt_pos_lst = []
    labels = []
    for prompt_data in train_data_bag :
        input_prompt, target_word, relation_type, knowledge_neurons = prompt_data[0], prompt_data[1], prompt_data[2], prompt_data[3]

        eval_features, tokens_info = example2feature([input_prompt,target_word,relation_type], arguments.max_seq_length, tokenizer)
        # convert features to long type tensors
        baseline_ids, input_ids, input_mask, segment_ids = eval_features['baseline_ids'], eval_features['input_ids'], eval_features['input_mask'], eval_features['segment_ids']
        baseline_ids = torch.tensor(baseline_ids, dtype=torch.long).unsqueeze(0)
        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        input_mask = torch.tensor(input_mask, dtype=torch.long).unsqueeze(0)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long).unsqueeze(0)
        baseline_ids = baseline_ids.to(device)
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)



        # record [MASK]'s position
        tgt_pos = tokens_info['tokens'].index('[MASK]')

        baseline_ids_lst.append(baseline_ids)
        input_ids_lst.append(input_ids)
        input_mask_lst.append(input_mask)
        segment_ids_lst.append(segment_ids)
        tgt_pos_lst.append(tgt_pos)
        labels.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(target_word)[-1]))
    
    baseline_ids = torch.cat(baseline_ids_lst,dim=0)
    input_ids = torch.cat(input_ids_lst,dim=0)
    input_mask = torch.cat(input_mask_lst,dim=0)
    segment_ids = torch.cat(segment_ids_lst,dim=0)
    tgt_pos = torch.tensor(tgt_pos_lst,device=device)
    labels = torch.tensor(labels,device=device)
    pos_samples_len = input_ids.shape[0]
    pos_tgt_pos = tgt_pos.clone()

    val_data_batch = convert_raw_batch_to_tensor(valid_data_bag,tokenizer,arguments)
    val_baseline_ids, val_input_ids, val_input_mask, val_segment_ids, val_tgt_pos, val_labels = val_data_batch['baseline_ids'], val_data_batch['input_ids'], val_data_batch['input_mask'], val_data_batch['segment_ids'], val_data_batch['tgt_pos'], val_data_batch['labels']

    train_loss = None
    valid_loss = None
    min_valid_loss = float('inf')
    min_train_loss = float('inf')
    model_saved = False
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        if display :
            print('epoch:',epoch)
        model_t_patcher.train()
        optimizer.zero_grad()
        #getting negative samples
        neg_sampler_output = neg_sampler.sample(0,tokenizer,arguments)
        neg_baseline_ids, neg_input_ids, neg_input_mask, neg_segment_ids, neg_tgt_pos, neg_labels = neg_sampler_output['neg_baseline_ids'], neg_sampler_output['neg_input_ids'], neg_sampler_output['neg_input_mask'], neg_sampler_output['neg_segment_ids'], neg_sampler_output['neg_tgt_pos'], neg_sampler_output['neg_labels']

        neg_samples_len = neg_input_ids.shape[0]

        if neg_input_ids.shape[0] > 0 :
            #concatinating negative samples
            baseline_ids = torch.cat([baseline_ids,neg_baseline_ids],dim=0)
            input_ids = torch.cat([input_ids,neg_input_ids],dim=0)
            input_mask = torch.cat([input_mask,neg_input_mask],dim=0)
            segment_ids = torch.cat([segment_ids,neg_segment_ids],dim=0)
            tgt_pos = torch.cat([tgt_pos,neg_tgt_pos],dim=0)
            labels = torch.cat([labels,neg_labels],dim=0)
        
        prediction_output, input_queiries = model_t_patcher(input_ids=input_ids, pos_samples_len=pos_samples_len, neg_samples_len = neg_samples_len, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos)
        logits = prediction_output[torch.arange(prediction_output.shape[0],device=device),tgt_pos]
        loss = edit_loss(logits,labels) 
        for layer in input_queiries.keys():
            loss += (act_loss_const * activation_loss(input_queiries[layer],pos_tgt_pos,neurons_list)) #+ (memory_loss_const * memory_loss(negative_quiry_dict[layer],diff_quiry_dict[layer],neg_tgt_pos,neurons_list,memory_k))
        loss.backward()
        optimizer.step()
        train_loss = loss.item()

        model_t_patcher.eval()
        
        if display :
            print('training loss:',train_loss)
        if train_loss <= min_train_loss :
            min_train_loss = train_loss 
            if model_save_path != None :
                torch.save(model_t_patcher.main_bert_model.state_dict(), model_save_path+'/T_Patcher_model_temp_save.pth')
                model_saved = True
                if display :
                    print('model saved')
        if display :
            print(' ')
    
    if model_save_path != None  and model_saved:
        model.load_state_dict(torch.load(model_save_path+'/T_Patcher_model_temp_save.pth'))  
    if display :  
        print('======== done ========')
        print(' ')
    return model




def add_patch_modified1(model,tokenizer,train_data,valid_data,neg_sampler,arguments,act_loss_const = 1,memory_loss_const = 10,memory_k = 1000,neurons_list=None,num_patches=1,num_epochs=20,model_save_path=None, display=False):
    if display :
        print('======== Model editing with modified T-Patcher ========')
    device = arguments.device
    n_gpu = arguments.n_gpu
    if neurons_list == None :
        neurons_list = torch.tensor([-i for i in range(1,num_patches+1)],device=device)
        #introducing new patches
        a = model.bert.encoder.layer[-1].intermediate.dense
        b = torch.nn.Linear(a.weight.data.shape[1],a.weight.data.shape[0]+num_patches).to(device)
        with torch.no_grad() :
            b.weight[:-num_patches] = a.weight.data.clone()
            b.bias[:-num_patches] = a.bias.data.clone()
        model.bert.encoder.layer[-1].intermediate.dense = b
        c = model.bert.encoder.layer[-1].output.dense
        d = torch.nn.Linear(c.weight.data.shape[1]+num_patches,c.weight.data.shape[0]).to(device)
        with torch.no_grad() :
            d.weight[:,:-num_patches] = c.weight.data.clone()
            d.bias[:] = c.bias.data.clone()
        model.bert.encoder.layer[-1].output.dense = d
    
    if display :
        print('neurons introduced')
        print('intermediate layer:',model.bert.encoder.layer[-1].intermediate.dense.weight.data.shape)
        print('intermediate layer:',model.bert.encoder.layer[-1].intermediate.dense.bias.data.shape)
        print('output layer:',model.bert.encoder.layer[-1].output.dense.weight.data.shape)
        print('output layer:',model.bert.encoder.layer[-1].output.dense.bias.data.shape)
        print(' ')
    
    #initialising T-Patcher optimizing model
    model_t_patcher = Optimizing_BERT_MODEL_T_PATCHER(model).to(device)
    optimizer = torch.optim.Adam(model_t_patcher.parameters(),lr = 0.001)
    edit_loss = torch.nn.CrossEntropyLoss()
    
    train_loss = None
    valid_loss = None
    min_valid_loss = float('inf')
    min_train_loss = float('inf')
    model_saved = False
    model_t_patcher.train()
    for epoch in range(num_epochs):
        st=time.time()
        torch.cuda.empty_cache()
        if display :
            print('epoch:',epoch)
        for relation , list_of_bags in train_data.items():
            for bag in list_of_bags :
                train_data_batch = convert_raw_batch_to_tensor(bag,tokenizer,arguments)
                baseline_ids, input_ids, input_mask, segment_ids, tgt_pos, labels = train_data_batch['baseline_ids'], train_data_batch['input_ids'], train_data_batch['input_mask'], train_data_batch['segment_ids'], train_data_batch['tgt_pos'], train_data_batch['labels']
                pos_samples_len = input_ids.shape[0]
                pos_tgt_pos = tgt_pos.clone()
                optimizer.zero_grad()
                neg_samples_len = 0
                if input_ids.shape[0] > 0 :
                    prediction_output, input_queiries = model_t_patcher(input_ids=input_ids, pos_samples_len=pos_samples_len, neg_samples_len = neg_samples_len, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos)
                    logits = prediction_output[torch.arange(prediction_output.shape[0],device=device),tgt_pos]
                    loss = edit_loss(logits,labels) 
                    '''
                    for layer in input_queiries.keys():
                        loss += (act_loss_const * activation_loss(input_queiries[layer],pos_tgt_pos,neurons_list)) #+ (memory_loss_const * memory_loss(negative_quiry_dict[layer],diff_quiry_dict[layer],neg_tgt_pos,neurons_list,memory_k))
                    '''
                    loss.backward()
                    optimizer.step()
                    train_loss = loss.item()

        model_t_patcher.eval()
        for relation , list_of_bags in valid_data.items():
            for bag in list_of_bags :
                train_data_batch = convert_raw_batch_to_tensor(bag,tokenizer,arguments)
                baseline_ids, input_ids, input_mask, segment_ids, tgt_pos, labels = train_data_batch['baseline_ids'], train_data_batch['input_ids'], train_data_batch['input_mask'], train_data_batch['segment_ids'], train_data_batch['tgt_pos'], train_data_batch['labels']
                pos_samples_len = input_ids.shape[0]
                pos_tgt_pos = tgt_pos.clone()
                
                neg_samples_len = 0
                if input_ids.shape[0] > 0 :
                
                    prediction_output, input_queiries = model_t_patcher(input_ids=input_ids, pos_samples_len=pos_samples_len, neg_samples_len = neg_samples_len, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos)
                    logits = prediction_output[torch.arange(prediction_output.shape[0],device=device),tgt_pos]
                    loss = edit_loss(logits,labels) 
                    '''
                    for layer in input_queiries.keys():
                        loss += (act_loss_const * activation_loss(input_queiries[layer],pos_tgt_pos,neurons_list)) #+ (memory_loss_const * memory_loss(negative_quiry_dict[layer],diff_quiry_dict[layer],neg_tgt_pos,neurons_list,memory_k))
                    '''
                    valid_loss = loss.item()
        
        et = time.time()
        if display :
            print('training loss:',train_loss)
            print('valid loss:',valid_loss)

        if valid_loss <= min_valid_loss :
            min_valid_loss = valid_loss 
            if model_save_path != None :
                torch.save(model_t_patcher.main_bert_model.state_dict(), model_save_path+'/T_Patcher_model_modified.pth')
                model_saved = True
                if display :
                    print('model saved')
        if display :
            diff_time = et - st
            time_lst = []
            time_taken = diff_time
            time_lst = []
            while time_taken >0 :
                temp = time_taken % 60
                time_taken = (time_taken - temp)//60
                time_lst = [str(int(temp))] + time_lst
            print('time taken:',':'.join(time_lst))
            print(' ')
    
    if model_save_path != None  and model_saved:
        model.load_state_dict(torch.load(model_save_path+'/T_Patcher_model_modified.pth'))  
    if display :  
        print('======== done ========')
        print(' ')
    return model


def add_patch_modified2(model,tokenizer,train_data,valid_data,neg_sampler,arguments,act_loss_const = 1,memory_loss_const = 10,memory_k = 1000,neurons_list=None,num_patches=1,num_epochs=20,model_save_path=None, display=False):
    if display :
        print('======== Model editing with modified T-Patcher ========')
    device = arguments.device
    n_gpu = arguments.n_gpu
    if neurons_list == None :
        neurons_list = torch.tensor([-i for i in range(1,num_patches+1)],device=device)
        #introducing new patches
        a = model.bert.encoder.layer[-1].intermediate.dense
        b = torch.nn.Linear(a.weight.data.shape[1],a.weight.data.shape[0]+num_patches).to(device)
        with torch.no_grad() :
            b.weight[:-num_patches] = a.weight.data.clone()
            b.bias[:-num_patches] = a.bias.data.clone()
        model.bert.encoder.layer[-1].intermediate.dense = b
        c = model.bert.encoder.layer[-1].output.dense
        d = torch.nn.Linear(c.weight.data.shape[1]+num_patches,c.weight.data.shape[0]).to(device)
        with torch.no_grad() :
            d.weight[:,:-num_patches] = c.weight.data.clone()
            d.bias[:] = c.bias.data.clone()
        model.bert.encoder.layer[-1].output.dense = d
    
    if display :
        print('neurons introduced')
        print('intermediate layer:',model.bert.encoder.layer[-1].intermediate.dense.weight.data.shape)
        print('intermediate layer:',model.bert.encoder.layer[-1].intermediate.dense.bias.data.shape)
        print('output layer:',model.bert.encoder.layer[-1].output.dense.weight.data.shape)
        print('output layer:',model.bert.encoder.layer[-1].output.dense.bias.data.shape)
        print(' ')
    
    total_neurons = model.bert.encoder.layer[-1].output.dense.weight.data.shape[1]
    neuron_list = [[11,i] for i in range(total_neurons-num_patches,total_neurons)]

    new_train_data = dict()
    new_val_data = dict()
    for relation, list_of_bags in train_data.items():
        temp_lst1 = []
        for bag in list_of_bags :
            temp_lst2 = []
            for prompt_data in bag:
                temp_lst2.append([prompt_data[0],prompt_data[1],prompt_data[2],copy.deepcopy(neuron_list)])
            temp_lst1.append(temp_lst2)
        new_train_data[relation] = temp_lst1
    
    for relation, list_of_bags in valid_data.items():
        temp_lst1 = []
        for bag in list_of_bags :
            temp_lst2 = []
            for prompt_data in bag:
                temp_lst2.append([prompt_data[0],prompt_data[1],prompt_data[2],copy.deepcopy(neuron_list)])
            temp_lst1.append(temp_lst2)
        new_val_data[relation] = temp_lst1
    
    
    model, train_acc, valid_acc = edit_model_kn_method2(model,
                                                            tokenizer,
                                                            new_train_data,
                                                            new_val_data,
                                                            arguments,
                                                            num_epochs,
                                                            model_save_path=arguments.model_save_path,
                                                            display=True)
    
    if display :  
        print('======== done ========')
        print(' ')
    return model



def edit_model_T_Patcher(model,tokenizer,train_data,val_data,arguments,num_epochs=20,model_save_path=None, display=False):
    total_prompts = 0
    for relation, list_of_bags in train_data.items():
        for bag in list_of_bags :
            for prompt_data in bag :
                total_prompts+=1
    
    prompt_count = 0
    device = arguments.device
    model.train()
    cur_printing_len = 0
    
    if display :
        print('======== Editing model using T-Patcher ========')
    for relation, list_of_bags in train_data.items():
        for bag_idx, bag in enumerate(list_of_bags):
            if len(bag)>0 :
                torch.cuda.empty_cache()
                per_st = time.time()
                prompt_count+=len(bag)
                neg_samples = Neg_samples(train_data,relation,bag_idx)
                model = add_patch(model,
                    tokenizer,
                    bag,
                    val_data[relation][bag_idx],
                    neg_sampler = neg_samples,
                    num_patches=1,
                    arguments=arguments,
                    display=False,
                    model_save_path= model_save_path,
                    num_epochs=num_epochs)
                current_per = round((prompt_count/total_prompts) * 100,2)
                if display : #current_per > current_per_complited and display
                    per_et = time.time()
                    diff_time = per_et - per_st
                    per_st = time.time()
                    expected_remaining_time = diff_time * (total_prompts - prompt_count)
                    time_lst = []
                    time_taken = expected_remaining_time
                    time_lst = []
                    while time_taken >0 :
                        temp = time_taken % 60
                        time_taken = (time_taken - temp)//60
                        time_lst = [str(int(temp))] + time_lst
                    
                    printing_string = 'Completed:'+str(current_per)+' %'+'     Expected remaining time: '+":".join(time_lst)
                    print('\r' + ' ' * cur_printing_len + '\r', end='', flush=True)
                    print(printing_string,end='', flush=True)
                    cur_printing_len = len(printing_string)
    
    #getting train data accuracy
    train_correct_count = 0
    train_total_count = 0
    for relation, list_of_bags in train_data.items():
        for bag in list_of_bags :
            for prompt_data in bag :
                train_total_count +=1
                input_prompt, target_word, relation_type, knowledge_neurons = prompt_data[0], prompt_data[1], prompt_data[2], prompt_data[3]

                arguments.get_pred = True
                arguments.get_ig_pred = False
                arguments.get_ig_gold = False
                arguments.get_base = False
                output_dict = get_attribution_scores(input_prompt,target_word,model,tokenizer,arguments,relation_type=relation_type)
                predicted_word = output_dict['pred_label']

                if predicted_word == target_word :
                    train_correct_count +=1
    
    # getting validation accuracy
    valid_correct_count = 0
    valid_total_count = 0
    for relation, list_of_bags in val_data.items():
        for bag in list_of_bags :
            for prompt_data in bag :
                valid_total_count +=1
                input_prompt, target_word, relation_type, knowledge_neurons = prompt_data[0], prompt_data[1], prompt_data[2], prompt_data[3]

                arguments.get_pred = True
                arguments.get_ig_pred = False
                arguments.get_ig_gold = False
                arguments.get_base = False
                output_dict = get_attribution_scores(input_prompt,target_word,model,tokenizer,arguments,relation_type=relation_type)
                predicted_word = output_dict['pred_label']

                if predicted_word == target_word :
                    valid_correct_count +=1
    
    if display :
        print(' ')
        print('======== Done editing ========')

    return model, train_correct_count/train_total_count, valid_correct_count/valid_total_count
    
            



def get_accuracy(model,tokenizer,data,arguments):
    model.eval()
    with torch.no_grad():
        correct_count = 0
        total_count = 0
        for relation, list_of_bags in data.items():
            for bag in list_of_bags :
                for prompt_data in bag :
                    total_count +=1
                    input_prompt, target_word, relation_type = prompt_data[0], prompt_data[1], prompt_data[2]

                    arguments.get_pred = True
                    arguments.get_ig_pred = False
                    arguments.get_ig_gold = False
                    arguments.get_base = False
                    output_dict = get_attribution_scores(input_prompt,target_word,model,tokenizer,arguments,relation_type=relation_type)
                    predicted_word = output_dict['pred_label']

                    if predicted_word == target_word :
                        correct_count +=1
    return correct_count/total_count



def main(model,tokenizer,data_dict,arguments,edit_type='BIRD+T_PATCHER',display=False):
    processed_data_dict = {}
    total_correct_predictions = 0
    total_wrong_predictions = 0
    correctly_predicted_data = {}
    if arguments.phaseone_processed_data_path==None or not os.path.exists(arguments.phaseone_processed_data_path+'/phaseone_processed_correct_data.json') :
        if display :
            print('======== Phase one processed data not found processing data ========')
        for relation, eval_bag_list in data_dict.items():
            temp_list1 = []
            temp_lst1_c = []
            for bag_idx, eval_bag in enumerate(eval_bag_list):
                temp_list2 = []
                temp_lst2_c = []
                for eval_example in eval_bag:
                    arguments.get_pred = True
                    arguments.get_ig_pred = False
                    arguments.get_ig_gold = False
                    arguments.get_base = False
                    output_dict = get_attribution_scores(eval_example[0],eval_example[1],model,tokenizer,arguments,relation_type=eval_example[2])
                    
                    if eval_example[1] == output_dict['pred_label'] :
                        total_correct_predictions+=1
                        temp_lst2_c.append(eval_example)
                    else:
                        total_wrong_predictions+=1
                        temp_list2.append(eval_example)

                temp_list1.append(temp_list2)
                temp_lst1_c.append(temp_lst2_c)
            processed_data_dict[relation]=temp_list1
            correctly_predicted_data[relation] = temp_lst1_c
        
        if arguments.phaseone_processed_data_path!=None :
            with open(arguments.phaseone_processed_data_path+'/phaseone_processed_correct_data.json','w') as fp :
                    json.dump(correctly_predicted_data,fp)

            with open(arguments.phaseone_processed_data_path+'/phaseone_processed_data.json','w') as fp :
                    json.dump(processed_data_dict,fp)
                    print('Phase one processed data saved')
    else:
        if display :
            print('======== Phase one processed data found loading data ========')
        with open(arguments.phaseone_processed_data_path+'/phaseone_processed_data.json','r') as fp :
            processed_data_dict = json.load(fp)
        with open(arguments.phaseone_processed_data_path+'/phaseone_processed_correct_data.json','r') as fp :
            correctly_predicted_data = json.load(fp)

    
    if display :
        print('======== phase two analysis of data ========')
    
    processed_data_dict = get_knowledge_neurons(model,tokenizer,processed_data_dict,arguments,relations=arguments.relations,save_folder_path=arguments.kns_results_folder,display=True)
    
    train_data = {}
    valid_data = {}
    for relation, list_of_bags in processed_data_dict.items():
        train_lst1 = []
        valid_lst1 = []
        for bag in list_of_bags :
            train_lst2 = []
            valid_lst2 = []
            if len(bag) <= 1 :
                train_lst2 = bag
            else:
                valid_lst2.append(bag[0])
                train_lst2 = bag[1:]
            train_lst1.append(train_lst2)
            valid_lst1.append(valid_lst2)
        train_data[relation] = train_lst1
        valid_data[relation] = valid_lst1
    
    
    print('======== Splitting data ========')
    number_zero_neuron_prompts = 0
    number_non_zero_neuron_prompts = 0
    zero_neurons_data = {}
    non_zero_neurons_data = {}
    for relation, list_of_bags in processed_data_dict.items():
        zero_neuron_lst1 = []
        non_zero_neuron_lst1 = []
        for bag in list_of_bags :
            zero_neuron_lst2 = []
            non_zero_neuron_lst2 = []
            for prompt_data in bag :
                if len(prompt_data[-1]) >= 2 :
                    non_zero_neuron_lst2.append(prompt_data)
                    number_non_zero_neuron_prompts+=1
                else:
                    zero_neuron_lst2.append(prompt_data)
                    number_zero_neuron_prompts+=1
            zero_neuron_lst1.append(zero_neuron_lst2)
            non_zero_neuron_lst1.append(non_zero_neuron_lst2)
        
        zero_neurons_data[relation] = zero_neuron_lst1
        non_zero_neurons_data[relation] = non_zero_neuron_lst1

    print('======== Editing model =======')
    edit_train_data = {}
    edit_valid_data = {}
    for relation, list_of_bags in non_zero_neurons_data.items():
        train_lst1 = []
        valid_lst1 = []
        for bag in list_of_bags :
            train_lst2 = []
            valid_lst2 = []
            if len(bag) <= 1 :
                train_lst2 = bag
            else:
                valid_lst2.append(bag[0])
                train_lst2 = bag[1:]
            train_lst1.append(train_lst2)
            valid_lst1.append(valid_lst2)
        edit_train_data[relation] = train_lst1
        edit_valid_data[relation] = valid_lst1
    
    new_train_data = {}
    new_valid_data = {}
    for relation, list_of_bags in zero_neurons_data.items():
        train_lst1 = []
        valid_lst1 = []
        for bag in list_of_bags :
            train_lst2 = []
            valid_lst2 = []
            if len(bag) <= 1 :
                train_lst2 = bag
            else:
                valid_lst2.append(bag[0])
                train_lst2 = bag[1:]
            train_lst1.append(train_lst2)
            valid_lst1.append(valid_lst2)
        new_train_data[relation] = train_lst1
        new_valid_data[relation] = valid_lst1

    if edit_type == 'kn_m1' :
        print('======== Model Editing using Knowledge neuron method1 ========')
        model, train_acc, valid_acc = edit_model_kn_method1(model,tokenizer,train_data,valid_data,arguments,display=True)
        train_acc = get_accuracy(model,tokenizer,train_data,arguments)
        val_acc = get_accuracy(model,tokenizer,valid_data,arguments)
        print('Final train accuracy:',train_acc)
        print('Final generality:',val_acc)
        if arguments.model_save_path != None :
                torch.save(model.state_dict(), arguments.model_save_path+'/Final_edited_model_kn_m1.pth')
        return model
    elif edit_type == 'kn_m2' :
        print('======== Model Editing using Knowledge nueron method2 ========')
        model, train_acc, valid_acc = edit_model_kn_method2(model,
                                                            tokenizer,
                                                            train_data,
                                                            valid_data,
                                                            arguments,
                                                            20,
                                                            model_save_path=arguments.model_save_path,
                                                            display=True)
        train_acc = get_accuracy(model,tokenizer,train_data,arguments)
        val_acc = get_accuracy(model,tokenizer,valid_data,arguments)
        print('Final train accuracy:',train_acc)
        print('Final generality:',val_acc)
        if arguments.model_save_path != None :
                torch.save(model.state_dict(), arguments.model_save_path+'/Final_edited_model_kn_m2.pth')
        return model
    elif edit_type == 'BIRD' :
        print('======== Model Editing using BIRD ========')
    
        train_data_bird_format = get_BIRD_data(train_data,tokenizer,arguments.relation_data_path)
        valid_data_bird_format = get_BIRD_data(valid_data,tokenizer,arguments.relation_data_path)
        model, train_acc, val_acc = edit_model_BIRD(model,tokenizer,train_data_bird_format,valid_data_bird_format,arguments,10,display=True)
        train_acc = get_accuracy(model,tokenizer,train_data,arguments)
        val_acc = get_accuracy(model,tokenizer,valid_data,arguments)
        print('Final train accuracy:',train_acc)
        print('Final generality:',val_acc)
        if arguments.model_save_path != None :
                torch.save(model.state_dict(), arguments.model_save_path+'/Final_edited_model_BIRD.pth')
        return model
    elif edit_type == 'T_Patcher' :
        print('======== Model Editing using T-Patcher ========')
        model, train_acc, val_acc = edit_model_T_Patcher(model,tokenizer,train_data,valid_data,arguments,model_save_path = arguments.model_save_path,display=True,num_epochs=30)
        train_acc = get_accuracy(model,tokenizer,train_data,arguments)
        val_acc = get_accuracy(model,tokenizer,valid_data,arguments)
        print('Final train accuracy:',train_acc)
        print('Final generality:',val_acc)
        if arguments.model_save_path != None :
                torch.save(model.state_dict(), arguments.model_save_path+'/Final_edited_model_T_Patcher.pth')
        return model
    
    elif edit_type == 'kn_m1+T_Patcher' :
        print('======== Model Editing using Knowledge neuron method1 and T-Patcher ========')
        model_1, train_acc, valid_acc = edit_model_kn_method1(model,tokenizer,edit_train_data,edit_valid_data,arguments,display=True)
        model, train_acc, val_acc = edit_model_T_Patcher(model_1,tokenizer,new_train_data,new_valid_data,arguments,model_save_path = arguments.model_save_path,display=True,num_epochs=30)
        
        train_acc = get_accuracy(model,tokenizer,train_data,arguments)
        val_acc = get_accuracy(model,tokenizer,valid_data,arguments)
        print('Final train accuracy:',train_acc)
        print('Final generality:',val_acc)
        if arguments.model_save_path != None :
                torch.save(model.state_dict(), arguments.model_save_path+'/Final_edited_model_kn_m1+T_Patcher.pth')
        return model
    elif edit_type == 'kn_m2+T_Patcher' :
        print('======== Model Editing using Knowledge neuron method2 and T-Patcher ========')
        model_1, train_acc, valid_acc = edit_model_kn_method2(model,
                                                            tokenizer,
                                                            edit_train_data,
                                                            edit_valid_data,
                                                            arguments,
                                                            20,
                                                            model_save_path=arguments.model_save_path,
                                                            display=True)
        model, train_acc, val_acc = edit_model_T_Patcher(model_1,tokenizer,new_train_data,new_valid_data,arguments,model_save_path = arguments.model_save_path,display=True,num_epochs=30)
        train_acc = get_accuracy(model,tokenizer,train_data,arguments)
        val_acc = get_accuracy(model,tokenizer,valid_data,arguments)
        print('Final train accuracy:',train_acc)
        print('Final generality:',val_acc)
        if arguments.model_save_path != None :
                torch.save(model.state_dict(), arguments.model_save_path+'/Final_edited_model_kn_m2+T_Patcher.pth')
        return model
    elif  edit_type == 'BIRD+T_Patcher' :
        print('======== Model Editing using BIRD and T-Patcher ========')
        train_data_bird_format = get_BIRD_data(edit_train_data,tokenizer,arguments.relation_data_path)
        valid_data_bird_format = get_BIRD_data(edit_valid_data,tokenizer,arguments.relation_data_path)
        model_1, train_acc, val_acc = edit_model_BIRD(model,tokenizer,train_data_bird_format,valid_data_bird_format,arguments,10,display=True)
        model, train_acc, val_acc = edit_model_T_Patcher(model_1,tokenizer,new_train_data,new_valid_data,arguments,model_save_path = arguments.model_save_path,display=True,num_epochs=30)
        train_acc = get_accuracy(model,tokenizer,train_data,arguments)
        val_acc = get_accuracy(model,tokenizer,valid_data,arguments)
        print('Final train accuracy:',train_acc)
        print('Final generality:',val_acc)
        if arguments.model_save_path != None :
                torch.save(model.state_dict(), arguments.model_save_path+'/Final_edited_model_BIRD+T_Patcher.pth')
        return model
    
    else:
        if edit_type == None :
            edit_type = 'BIRD and Modified T-Patcher'
        print('======== Model Editing using ',edit_type,' ========')
        
        model_1, train_acc, valid_acc = edit_model_kn_method2(model,
                                                            tokenizer,
                                                            edit_train_data,
                                                            edit_valid_data,
                                                            arguments,
                                                            20,
                                                            model_save_path=arguments.model_save_path,
                                                            display=True)
        
        model  = add_patch_modified2(model_1,tokenizer,new_train_data,new_valid_data,None,arguments,num_patches=200,num_epochs=50,model_save_path=arguments.model_save_path, display=True)
        train_acc = get_accuracy(model,tokenizer,train_data,arguments)
        val_acc = get_accuracy(model,tokenizer,valid_data,arguments)
        print('Final train accuracy:',train_acc)
        print('Final generality:',val_acc)
        if arguments.model_save_path != None :
                torch.save(model.state_dict(), arguments.model_save_path+'/Final_edited_model_BIRD+m_T_Patcher.pth')
        return model


def get_kns_statistics(folder_path,display = False):
    relation_nonzero_neuron_dict = dict()
    relation_zero_neuron_dict = dict()
    for file_name in os.listdir(folder_path) :
        if (file_name.split('_')[-1]).split('.')[0] == 'kns' :
            with open(folder_path+'/'+file_name,'r') as fp :
                data = json.load(fp)
                zero_neurons_data = {}
                non_zero_neurons_data = {}
                for relation, list_of_bags in data.items():
                    zero_neuron_lst1 = []
                    non_zero_neuron_lst1 = []
                    number_non_zero_neuron_prompts = 0
                    number_zero_neuron_prompts = 0
                    for bag in list_of_bags :
                        zero_neuron_lst2 = []
                        non_zero_neuron_lst2 = []
                        for prompt_data in bag :
                            if len(prompt_data[-1]) >= 2 :
                                non_zero_neuron_lst2.append(prompt_data)
                                number_non_zero_neuron_prompts+=1
                            else:
                                zero_neuron_lst2.append(prompt_data)
                                number_zero_neuron_prompts+=1
                        zero_neuron_lst1.append(zero_neuron_lst2)
                        non_zero_neuron_lst1.append(non_zero_neuron_lst2)
                    relation_nonzero_neuron_dict[relation] = relation_nonzero_neuron_dict.get(relation,0) + number_non_zero_neuron_prompts
                    relation_zero_neuron_dict[relation] = relation_zero_neuron_dict.get(relation,0) + number_zero_neuron_prompts
                    zero_neurons_data[relation] = zero_neuron_lst1
                    non_zero_neurons_data[relation] = non_zero_neuron_lst1
    relation_list = []
    zero_neuron_list = []
    nonzero_neuron_list = []
    for relation in relation_zero_neuron_dict.keys():
        relation_list.append(relation)
        zero_neuron_list.append(relation_zero_neuron_dict.get(relation,0))
        nonzero_neuron_list.append(relation_nonzero_neuron_dict.get(relation,0))
    
    if display :
        bar_width = 0.35
        x = np.arange(len(relation_list))
        plt.figure(figsize=(20,6))
        plt.bar(x - bar_width/2, nonzero_neuron_list, width=bar_width, label = 'nonzero neuron facts')
        plt.bar(x + bar_width/2, zero_neuron_list, width=bar_width, label = 'zero neuron facts')
        plt.xlabel('relation type')
        plt.ylabel('number of facts')
        plt.title('number of facts in each relation type with zero and non zero neurons')
        plt.xticks(x, relation_list)
        plt.legend()
        plt.show()


if __name__ == '__main__' :



    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    args = Params()
    args.seed = 42
    args.output_dir = './results'
    args.output_prefix = 'TREx-all'
    args.bert_model = 'bert-base-cased'
    args.do_lower_case = False
    args.tmp_data_path = './knowledge_neuron/knowledge_neurons/data/PARAREL/data_all_allbags.json'
    args.data_path = './knowledge_neuron/knowledge_neurons/data/PARAREL/data_all.json'
    args.debug = 100000
    args.pt_relation = 'P101'
    args.max_seq_length = 128
    args.get_pred = True
    args.batch_size = 20
    args.num_batch = 1
    args.get_ig_pred = True
    args.get_ig_gold = True
    args.get_base = True
    args.phaseone_processed_data_path = './data'
    args.relative_attribution_threshold = 0.2
    #args.relations = ['P1376'] #'P463',
    args.kns_results_folder = './kns_results'
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.n_gpu = 1
    args.model_save_path = './temp_model_save'
    args.relation_data_path = './knowledge_neuron/knowledge_neurons/data/LAMA/raw_data/TREx'

    # prepare eval set
    if os.path.exists(args.tmp_data_path):
        with open(args.tmp_data_path, 'r') as f:
            eval_bag_list_perrel = json.load(f)
    else:
        with open(args.data_path, 'r') as f:
            eval_bag_list_all = json.load(f)
        # split bag list into relations
        eval_bag_list_perrel = {}
        for bag_idx, eval_bag in enumerate(eval_bag_list_all):
            bag_rel = eval_bag[0][2].split('(')[0]
            if bag_rel not in eval_bag_list_perrel:
                eval_bag_list_perrel[bag_rel] = []
            if len(eval_bag_list_perrel[bag_rel]) >= args.debug:
                continue
            eval_bag_list_perrel[bag_rel].append(eval_bag)
        with open(args.tmp_data_path, 'w') as fw:
            json.dump(eval_bag_list_perrel, fw, indent=2)



    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
    model = BertForMaskedLM.from_pretrained('bert-base-cased')
    model = model.to(device)
    print(model)

    args.relations = ['P1376'] # change the relation types which you wish to edit
    relation_lst = [i for i in eval_bag_list_perrel.keys()]
    input_relation = str(input('Enter the relation type:'))
    if input_relation in relation_lst :
        args.relations = [input_relation]
    
    args.edit_type = None
    input_edit_type = str(input('Enter the type of edit:'))
    edit_types = ['kn_m1', 'kn_m2', 'BIRD', 'T_Patcher', 'kn_m1+T_Patcher', 'kn_m2+T_Patcher', 'BIRD+T_Patcher', 'BIRD and Modified T-Patcher']
    if input_edit_type in edit_types :
        args.edit_type = input_edit_type

    model = main(model,tokenizer,eval_bag_list_perrel,args,edit_type=args.edit_type,display=True)
