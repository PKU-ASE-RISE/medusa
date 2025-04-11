import torch, time
from torch.optim import AdamW,Adam
import numpy as np
from tqdm.auto import tqdm
from transformers import get_scheduler
from torch.cuda.amp import autocast, GradScaler
answer_list = {
    'rte':[[3,35,5756,297,1],[59,834,35,5756,297,1]],
    'cola':[[9961,1],[29452,1]],
    'mrpc':[[7072,1],[59,834,15,1169,15592,1]],
    'mnli':[[3, 35, 5756, 297, 1],[7163, 1],[27252, 1]],
    'sst2':[[2841,1],[1465,1]],
    'qnli':[[3,35,5756,297,1],[59,834,35,5756,297,1]],
    'qqp':[[19197, 1],[59, 834, 26, 413, 26221, 1]],
    'wnli':[[3,35,5756,297,1],[59,834,35,5756,297,1]],
}
debug=False

REAL_BATCH = {
    'rte':16,
    'wnli':32,
    'cola':256,
    'mrpc':256,
    'mnli':256,
    'sst2':256,
    'qnli':256,
    'qqp':256,
}


def get_metric(model, dataloader, answer_list, no_metric=False):
    if no_metric:
        return 0.0
    model.eval()
    ml = max(list(map(len, answer_list)))
    # metric = evaluate.load("accuracy")
    acc,tot = 0,0
    for id,batch in enumerate(dataloader):
        # _batch = {k: torch.stack(batch[k],axis=-1).to(model.device) for k in ['input_ids', 'labels']}
        with torch.no_grad():
            source_ids, source_mask, lm_labels, target_mask = batch
            lm_labels[lm_labels[:, :] == 0] = -100

            outputs = model.generate(
                input_ids=source_ids.to(model.device),
                attention_mask=source_mask.to(model.device),
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                min_length=ml+1
            )
        scores = torch.stack(outputs.scores, dim=1).to("cpu").numpy()
        # predictions = np.argmax(np.asarray(logits)[:,0,:], axis=-1)
        # references = np.asarray(batch['labels'])[0,:]
        
        predictions = np.argmax(scores, axis=-1)
        references = np.asarray(lm_labels)[:,:]

        
        orignial_length = scores[0].shape[0]
        scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        scores = scores / np.sum(scores, axis=-1, keepdims=True)
        for i in range(scores.shape[0]):
            t = references[i]
            good = 1
            # lofits is not softmaxed
            prob = 1
            for k in range(len(t)):
                if t[k]>1:
                    prob *= scores[i,k,t[k]]
                else:
                    break
            if debug:
                print(t,prob)
            for t in answer_list:
                p = 1
                for k in range(len(t)):
                    if t[k]>1:
                        p *= scores[i,k,t[k]]
                    else:
                        break
                    
                if debug:
                    print(t,p)
                if p == prob:
                    same_one = 1
                    for k in range(len(t)):
                        if t[k]>1:
                            if(t[k]!=references[i][k]):
                                same_one = 0
                                break
                    if not same_one:
                        good = 0
                        break
                if p > prob :
                    good = 0
                    break
            if debug:
                print(good)
                print('-'*50)
            acc += good
            tot += 1

    return acc / tot

def get_training(args, model, epoch, batch, dataset_name, train_dataloader, 
    model_mask = None, cut_epoch=20000, evaluate_savemodel=False, validation_dataloader=None,
    no_metric=False):
    # exit(0)
    if cut_epoch <= 0:
        cut_epoch = 1
        
    print(dataset_name, batch, epoch)
    
    
    device = args.device
    OUT_DIR = args.out_dir
    
    optimizer = Adam(model.parameters(), lr=args.lr)
    num_training_steps = epoch * len(train_dataloader)
    
    if model_mask is not None:
        model_mask.apply_mask(model)
    
    progress_bar = tqdm(range(num_training_steps))
    ce = min(epoch, cut_epoch)
    best_metric = None

    accum_iter = int(REAL_BATCH[dataset_name] / batch)
    if not args.modularized:
        accum_iter = 1

    scaler = GradScaler()
    
    early_stopping_count = 0
    start = time.time()
    for epoch in range(ce):
        model.train()
        acc, tot = 0,0
        for id,_batch in enumerate(train_dataloader):
            source_ids, source_mask, lm_labels, target_mask = _batch
            lm_labels[lm_labels[:, :] == 0] = -100

            if not args.mixed:
                outputs = model(
                    input_ids=source_ids.to(model.device),
                    attention_mask=source_mask.to(model.device),
                    labels = lm_labels.to(model.device),
                    decoder_attention_mask=target_mask.to(model.device)
                )
                
                loss = outputs.loss / accum_iter 
                loss.backward()
                acc += loss.item()
                tot += len(_batch)
            else:
                with autocast(dtype=torch.bfloat16):
                    outputs = model(
                        input_ids=source_ids.to(model.device),
                        attention_mask=source_mask.to(model.device),
                        labels = lm_labels.to(model.device),
                        decoder_attention_mask=target_mask.to(model.device)
                    )
                loss = outputs.loss
                scaler.scale(loss).backward()
                acc += loss.item()
                tot += len(_batch)
                
            progress_bar.update(1)
            if  (id + 1) % accum_iter == 0 or id + 1 == len(train_dataloader):
                if model_mask is not None:
                    model_mask()
                if args.mixed:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
            if args.test_batch and time.time()-start > 60*60:
                break
                
        if evaluate_savemodel:
            print(epoch,'loss:',acc / tot)
            now2_metric = get_metric(model, validation_dataloader, answer_list[dataset_name], no_metric)
            print(epoch,'validation_metric:',now2_metric)
            if best_metric is None or now2_metric > best_metric:
                best_metric = now2_metric
                model.save_pretrained(OUT_DIR[:-1]+'_best/',safe_serialization=False)
                early_stopping_count = 0
            else:
                early_stopping_count += 1
            model.save_pretrained(OUT_DIR,safe_serialization=False)
            
            
            
            if epoch % 5 == 0:
                model.save_pretrained(OUT_DIR[:-1]+f'_epoch_{epoch}/',safe_serialization=False)
            if args.test_batch:
                with open('es_config.txt','a+') as f:
                    f.write(f'{args.dataset} in {args.model}\n')
                    f.write(f'early_stopping:{epoch}\n')
                    f.write(f'batch_size:{batch}\n')
                    f.write('#'*20+'\n')
                break
            if early_stopping_count >= 5 and args.early_stopping:
                with open('cas_config.txt','a+') as f:
                    f.write(f'{args.dataset} in {args.model}\n')
                    f.write(f'early_stopping:{epoch}\n')
                    f.write(f'batch_size:{batch}\n')
                    f.write(f'best_metric:{best_metric}\n')
                    f.write(f'time:{best_metric}\n')
                    f.write('#'*20+'\n')
                break
    if evaluate_savemodel: 
        print('best validation_metric:',best_metric)
    return model


