
import os, evaluate, torch, argparse, sys
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_scheduler
import numpy as np


from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from glue_tasks import glue_tasks, add_glue_tasks
from gradient_masking import magnitude_mutual_mask, warmup_gradient_mask_block, mutual_mask, half_mask
from gradient_masking import get_param_list, keep_topk, get_task_vector, get_sign_conflict, list_sub
from T5train import get_metric, get_training
import logging, warnings

from peft import LoraModel, LoraConfig, get_peft_model, IA3Config


def get_model(type, device, peft):
    
    model = T5ForConditionalGeneration.from_pretrained(type, dropout_rate=0).to(device)
    
    if peft == "lora":
        config = LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
        )
    elif peft == "ia3":
        config = IA3Config(
            peft_type="IA3",
        )
    elif not peft == None:
       raise Exception(f"{peft} is not supported")
        
    if not peft == None:
        model = get_peft_model(model, config)
        model.print_trainable_parameters()

    return model

if __name__ == '__main__':    

    parser = argparse.ArgumentParser()
    parser.add_argument('--mask', type=str, default='mutual_mask',help='mask method to use.')
    parser.add_argument('--out_dir', type=str, default='ckpts/mutual0827/', help='where to put model; will save a best ckpt under ckpts/mask0_best/ folder.')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--model', type=str, default="google/t5-v1_1-base",help='base model to use')
    parser.add_argument('--top_k', type=float, default=40, help='save top-k greatest gradient position')
    parser.add_argument('--datasets', type=str, nargs='+', required=True)
    parser.add_argument('--no_metric',  action="store_true")
    parser.add_argument('--ref_models', type=str, nargs='+')
    parser.add_argument('--larger_batch',  type=int, default=1)
    parser.add_argument('--smaller_batch',  type=int, default=1)
    parser.add_argument('--modularized', action="store_true")
    
    
    parser.add_argument('--tiny_sample', action="store_true")

    
    parser.add_argument('--early_stopping', action="store_true")
    
    parser.add_argument('--mixed', action="store_true")
    parser.add_argument('--test_batch', action="store_true")
    parser.add_argument('--cut_percent', type=int, default=30,help='train to what extent')
    parser.add_argument('--num_warmup_steps', type=int, default=200,help='warm up step in learning rate scheduler')
    
    parser.add_argument('--peft', type=str, default=None)
    
    args = parser.parse_args()
    args.base_model = args.model
    device = args.device
    OUT_DIR = args.out_dir
    
    all_exist = 1
    for dataset in args.datasets:
        if not os.path.exists(OUT_DIR + f'{dataset}_epoch_0/'):
            all_exist = 0
            break
    if all_exist:
        raise Exception(f'{"_".join(args.datasets)}  already trained!')
    
    args.train_dataloaders, args.test_dataloaders, args.validation_dataloaders = [],[],[]
    args.epochs, args.batchs= [],[]
    add_glue_tasks(args)
    
    
    os.makedirs(OUT_DIR, exist_ok=True)
    if 'toy' not in OUT_DIR:
        sys.stdout = open(OUT_DIR+'output_log.txt', 'w+')
    
    # model = T5ForConditionalGeneration.from_pretrained(args.model, dropout_rate=0).to(device)
    model = get_model(args.model, args.device, args.peft)
    
    print (f'masking with {args.mask} on top-{args.top_k}')
    
    model_masks = []
    if args.mask=='no_mask':
        
        for data in args.datasets:
            model_masks += [None]

    elif args.mask=='magnitude_mutual_mask':
       
        
        
        delta_params = []
        base_param = None
        
        for train_dataloader, epoch, batch, dataset in \
            zip(args.train_dataloaders, args.epochs, args.batchs, args.datasets):
            
            model = get_model(args.model, args.device, args.peft)
            # model = T5ForConditionalGeneration.from_pretrained(args.model, dropout_rate=0).to(device)
            
            if base_param is None:
                base_param = get_param_list(model, clone=True)
            model = get_training(args, model, epoch, batch, dataset, train_dataloader, \
                cut_epoch=int(epoch*args.cut_percent/100),no_metric=args.no_metric)
            delta_params += [list(map(lambda x:keep_topk(x,args).abs(),\
                get_task_vector(get_param_list(model, clone=True),base_param)))]
            del model
        
        delta_sum = [torch.max(torch.stack([delta_params[j][i].abs() for j in range(len(delta_params))],axis=0),axis=0).values \
            for i in range(len(delta_params[0]))]
        
        for delta_param in delta_params:
            model_mask = magnitude_mutual_mask(None)
            model_mask.generate_mask(delta_param, delta_param, delta_sum, args, fill_up=float('inf'))
            model_masks += [model_mask]
  
    elif args.mask == 'half_mask':
        model = T5ForConditionalGeneration.from_pretrained(args.model, dropout_rate=0)
        for id,dataset in enumerate(args.datasets):
            model_mask = half_mask(model)
            model_mask.generate_mask(id)
            model_masks += [model_mask]
            
    elif args.mask=='soft_magnitude_mutual_mask':
      
        
        delta_params = []
        base_param = None
        
        for train_dataloader, epoch, batch, dataset in \
            zip(args.train_dataloaders, args.epochs, args.batchs, args.datasets):
                
            model = get_model(args.model, args.device, args.peft)
            # model = T5ForConditionalGeneration.from_pretrained(args.model, dropout_rate=0).to(device)
            
            if base_param is None:
                base_param = get_param_list(model, clone=True)
            model = get_training(args, model, epoch, batch, dataset, train_dataloader, \
                cut_epoch=int(epoch*args.cut_percent/100),no_metric=args.no_metric)
            delta_params += [list(map(lambda x:keep_topk(x,args).abs(),\
                get_task_vector(get_param_list(model, clone=True),base_param)))]
            del model
        
        delta_sum = [sum(delta_params[j][i] for j in range(len(delta_params))) \
            for i in range(len(delta_params[0]))]
        
        torch.manual_seed(42)
        delta_random = [ p * torch.rand_like(p) for p in delta_sum ]
        
        for delta_param in delta_params:
            model_mask = magnitude_mutual_mask(None)
            delta_sum = list_sub(delta_sum, delta_param)
            
            model_mask.generate_mask(delta_param, delta_random, delta_sum, args, fill_up=float('inf'))
            model_masks += [model_mask]

    elif args.mask=='reference_mask':
        delta_params = []
        base_param = None
        
        for train_dataloader, epoch, batch, dataset in \
            zip(args.train_dataloaders, args.epochs, args.batchs, args.dataset):
            
            model = get_model(args.model, args.device, args.peft)
            # model = T5ForConditionalGeneration.from_pretrained(args.model, dropout_rate=0).to(device)
            
            if base_param is None:
                base_param = get_param_list(model, clone=True)
            model = get_training(args,model,epoch,batch,dataset,train_dataloader,\
                cut_epoch=int(epoch*args.cut_percent/100),no_metric=args.no_metric)
            delta_params += [list(map(lambda x:keep_topk(x,args),\
                get_task_vector(get_param_list(model, clone=True),base_param)))]
            del model
        
        delta_param = delta_params[0]
        
        
        best_conflict, best_tv, best_ref = 2, None, None
        for ref_model in args.ref_models:
            
            model = get_model(ref_model, args.device, args.peft)
            # model = T5ForConditionalGeneration.from_pretrained(ref_model, dropout_rate=0)
            
            task_vector = list(map(lambda x:keep_topk(x,args),\
                get_task_vector(get_param_list(model, clone=True),base_param)))
            sign_conflict = get_sign_conflict(delta_param, task_vector)
            
            if sign_conflict < best_conflict:
                best_conflict = sign_conflict
                best_tv = task_vector
                best_ref = ref_model
                
        print('best merging objec is',best_ref,'with',best_conflict)
        model_mask = mutual_mask(None)
        model_mask.generate_mask(delta_param,best_tv, args)
        model_masks += [model_mask]
        print('mask counts:',model_mask.mask_count())
    else:
        raise Exception('Masking method is not provided!!!')
    
    for train_dataloader, test_dataloaders, validation_dataloader, epoch, batch, dataset, model_mask in \
        zip(args.train_dataloaders, args.test_dataloaders, args.validation_dataloaders, \
            args.epochs, args.batchs, args.datasets, model_masks):
        if 'toy' not in OUT_DIR:
            sys.stdout = open(OUT_DIR+f'{dataset}_output_log.txt', 'w+')
        
        model = get_model(args.model, args.device, args.peft)
        # model = T5ForConditionalGeneration.from_pretrained(args.model, dropout_rate=0).to(device)
        
        args.out_dir = OUT_DIR + f'{dataset}/'
        model = get_training(args, model, epoch, batch, dataset, train_dataloader, \
            model_mask = model_mask, evaluate_savemodel=True, validation_dataloader=validation_dataloader,\
                no_metric=args.no_metric)

        del model


    