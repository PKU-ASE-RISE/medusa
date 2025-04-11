import torch
import os, evaluate, torch, argparse, sys
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_scheduler
from datasets import load_dataset
from glue_tasks import glue_tasks
from merge import merge_model
from T5train import get_metric, answer_list
from T5_mutual_mask import get_model

if __name__ == '__main__':    

    parser = argparse.ArgumentParser()
    parser.add_argument('--out_file', type=str, default='log/1.txt', required=True,help='will APPEND results to target file.')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--method', type=str, default="_SUM",help='merge methods to use; check details in merge.py.')
    parser.add_argument('--models', nargs='+', required=True,help='folders storing the models; shouldnt contain the base model')
    parser.add_argument('--base_model', type=str, default="google/t5-v1_1-base",help='base model; will be loaded')
    # parser.add_argument('--answer_list', nargs='+', required=True) 
    parser.add_argument('--dataset', type=str, default='mnli')
    parser.add_argument('--no_metric',  action="store_true")
    parser.add_argument('--peft', type=str, default=None)
    parser.add_argument('--tiny_sample', action="store_true")
    
    parser.add_argument('--larger_batch',  type=int, default=1)
    parser.add_argument('--modularized', action="store_true")
    parser.add_argument('--smaller_batch',  type=int, default=1)
    parser.add_argument('--mixed', action="store_true")
    parser.add_argument('--lambda1',  type=int, default=20)
    parser.add_argument('--lambda2',  type=float, default=1)
    args = parser.parse_args()
    
    args.model = args.base_model
    
    sys.stdout = open(args.out_file, 'a+')
    
    train_dataloader, test_dataloader, validation_dataloader = glue_tasks(args)
    
    models = [get_model(args.base_model, "cpu", args.peft)]
    models[-1].eval()
    
    for i in args.models:
        models += [T5ForConditionalGeneration.from_pretrained(i,dropout_rate=0)]
        models[-1].eval()
    for name,model in zip(args.models, models):
        print(name,':')
    # for name,model in zip(args.models, models):
    #     print(name,':',get_metric(model, validation_dataloader, args.answer_list))
    
    if(len(args.models)>=2):
        args.models += ['****magically merged****']
        models += [merge_model(models, args.method, args)]
    
    
    for name,model in zip(args.models[-1:], models[-1:]):
        model.to(args.device)
        print(name,':',get_metric(model, validation_dataloader, answer_list[args.dataset]))
    
    print('#'*20)
    print('#'*20)
    print('#'*20)
    