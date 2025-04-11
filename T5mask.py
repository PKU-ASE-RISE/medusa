
import os, evaluate, torch, argparse, sys
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_scheduler
from datasets import load_dataset
import numpy as np
from torch import nn
from torch.optim import AdamW,Adam
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from glue_tasks import glue_tasks
from gradient_masking import no_mask, gradient_mask, gradient_mask_block, random_mask, warmup_gradient_mask_block
from torch.cuda.amp import autocast
from gradient_masking import get_param_list
from T5train import get_metric
from T5_mutual_mask import get_model

from T5train import answer_list, get_training
if __name__ == '__main__':    

    parser = argparse.ArgumentParser()
    parser.add_argument('--mask', type=str, default='normal',help='mask method to use.')
    parser.add_argument('--out_dir', type=str, default='ckpts/mask0/', required=True,help='where to put model; will save a best ckpt under ckpts/mask0_best/ folder.')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--no_metric',  action="store_true")
    parser.add_argument('--larger_batch',  type=int, default=1)
    parser.add_argument('--smaller_batch',  type=int, default=1)
    parser.add_argument('--model', type=str, default="google/t5-v1_1-base",help='base model to use')
    parser.add_argument('--top_k', type=float, default=30, help='save top-k greatest gradient position')
    parser.add_argument('--dataset', type=str, default='mnli',help='dataset to train on')
    parser.add_argument('--early_stopping', action="store_true")
    parser.add_argument('--test_batch', action="store_true")
    parser.add_argument('--mixed', action="store_true")
    parser.add_argument('--modularized', action="store_true")

    parser.add_argument('--tiny_sample', action="store_true")
    parser.add_argument('--peft', type=str, default=None)
    
    args = parser.parse_args()
    args.base_model = args.model
    device = args.device
    
    train_dataloader, test_dataloader, validation_dataloader = glue_tasks(args)

    EPOCHS = args.epoch
    BATCH_SIZE = args.batch
    
    print(args.out_dir)
    OUT_DIR = args.out_dir
    os.makedirs(OUT_DIR, exist_ok=True)

    if os.path.exists(OUT_DIR[:-1]+f'_epoch_{0}/'):
        if 'toy' not in OUT_DIR:
            raise Exception(f'{OUT_DIR}  already trained!')
    
    if 'toy' not in OUT_DIR:
        sys.stdout = open(OUT_DIR+'output_log.txt', 'w+')
    
    
    model = T5ForConditionalGeneration.from_pretrained(args.model, dropout_rate=0).to(device)
    
    if args.peft is not None:
        model = get_model(args.model, args.device, args.peft)
    
    print (f'masking with {args.mask} on top-{args.top_k}')
    
    if args.mask!='' and args.mask!='normal':
        raise Exception('No such masking stategy.')
    else: 
        model_mask = no_mask(model)
        model_mask.generate_mask(train_dataloader, args)

    
    get_training(args, model, args.epoch, args.batch, args.dataset, train_dataloader, \
            model_mask = model_mask, evaluate_savemodel=True, validation_dataloader=validation_dataloader,\
                no_metric=args.no_metric)
