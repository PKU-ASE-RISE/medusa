import os
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_scheduler
from datasets import load_dataset


from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, SequentialSampler
ava = ['cola','rte','sst2','qnli','qqp','wnli','mnli',]

def getDataLoader_cola(args,input,tokenizer):
    inputs = ["cola sentence: " + doc1 for doc1 in input['sentence']]
    tokenized_inputs = tokenizer(inputs, padding = True, truncation = True, return_tensors="pt")
    source_ids = tokenized_inputs['input_ids']
    source_mask = tokenized_inputs['attention_mask']
    
    targets = [('acceptable' if tag else 'unacceptable') for tag in input['label']]
    tokenized_outputs = tokenizer(targets, padding = True, return_tensors="pt")
    target_ids = tokenized_outputs['input_ids']
    target_mask = tokenized_outputs['attention_mask']

    batch_size = args.batch
    data = TensorDataset(source_ids, source_mask, target_ids, target_mask)
    sampler = SequentialSampler(data)
    return DataLoader(data, sampler = sampler, batch_size=batch_size)

def getDataLoader_rte(args,input, tokenizer):
    inputs = ["rte sentence1: " + doc1 + " sentence2: "+ doc2 for doc1, doc2 in zip(input['sentence1'], input['sentence2'])]
    tokenized_inputs = tokenizer(inputs, padding = True, truncation = True, return_tensors="pt")
    source_ids = tokenized_inputs['input_ids']
    source_mask = tokenized_inputs['attention_mask']

    targets = ['entailment' if label == 0 else 'not_entailment' for label in input['label']]
    tokenized_outputs = tokenizer(targets, padding = True, return_tensors="pt")
    target_ids = tokenized_outputs['input_ids']
    target_mask = tokenized_outputs['attention_mask']
    
    batch_size = args.batch
    data = TensorDataset(source_ids, source_mask, target_ids, target_mask)
    sampler = SequentialSampler(data)
    return DataLoader(data, sampler = sampler, batch_size=batch_size)

def getDataLoader_mrpc(args,input, tokenizer):
    inputs = ["mrpc sentence1: " + doc1 + " sentence2: "+ doc2 for doc1, doc2 in zip(input['sentence1'], input['sentence2'])]
    tokenized_inputs = tokenizer(inputs, padding = True, truncation = True, return_tensors="pt")
    source_ids = tokenized_inputs['input_ids']
    source_mask = tokenized_inputs['attention_mask']
    
    targets = [('equivalent' if label==1 else 'not_equivalent') for label in input['label']]
    tokenized_outputs = tokenizer(targets, padding = True, return_tensors="pt")
    target_ids = tokenized_outputs['input_ids']
    target_mask = tokenized_outputs['attention_mask']
    
    batch_size = args.batch
    data = TensorDataset(source_ids, source_mask, target_ids, target_mask)
    sampler = SequentialSampler(data)
    return DataLoader(data, sampler = sampler, batch_size=batch_size)

def getDataLoader_mnli(args,input, tokenizer):
    inputs = ["mnli premise: " + doc1 + " hypothesis: "+ doc2 for doc1, doc2 in zip(input['premise'], input['hypothesis'])]
    tokenized_inputs = tokenizer(inputs, padding = True, truncation = True, return_tensors="pt")
    source_ids = tokenized_inputs['input_ids']
    source_mask = tokenized_inputs['attention_mask']
    def modify_mnli(tag):
        if tag==0:
            return 'entailment'
        elif tag==1:
            return 'neutral'
        else:
            return 'contradiction'
    targets = [modify_mnli(label) for label in input['label']]
    tokenized_outputs = tokenizer(targets, padding = True, return_tensors="pt")
    target_ids = tokenized_outputs['input_ids']
    target_mask = tokenized_outputs['attention_mask']
    
    batch_size = args.batch
    data = TensorDataset(source_ids, source_mask, target_ids, target_mask)
    sampler = SequentialSampler(data)
    return DataLoader(data, sampler = sampler, batch_size=batch_size)
    
def getDataLoader_sst2(args,input, tokenizer):
    inputs = ["sst2 sentence: " + doc1 for doc1 in input['sentence']]
    tokenized_inputs = tokenizer(inputs, padding = True, truncation = True, return_tensors="pt")
    source_ids = tokenized_inputs['input_ids']
    source_mask = tokenized_inputs['attention_mask']

    targets = ["negative" if label == 0 else "positive" for label in input['label']]
    tokenized_outputs = tokenizer(targets, padding = True, return_tensors="pt")
    target_ids = tokenized_outputs['input_ids']
    target_mask = tokenized_outputs['attention_mask']
    
    batch_size = args.batch
    data = TensorDataset(source_ids, source_mask, target_ids, target_mask)
    sampler = SequentialSampler(data)
    return DataLoader(data, sampler = sampler, batch_size=batch_size)

def getDataLoader_qnli(args,input, tokenizer):
    inputs = ["qnli question: "+doc1+" sentence: "+doc2 for doc1, doc2 in zip(input['question'], input['sentence'])]
    tokenized_inputs = tokenizer(inputs, padding = True, truncation = True, return_tensors="pt")
    source_ids = tokenized_inputs['input_ids']
    source_mask = tokenized_inputs['attention_mask']

    targets = ['entailment' if label == 0 else 'not_entailment' for label in input['label']]
    tokenized_outputs = tokenizer(targets, padding = True, return_tensors="pt")
    target_ids = tokenized_outputs['input_ids']
    target_mask = tokenized_outputs['attention_mask']
    
    batch_size = args.batch
    data = TensorDataset(source_ids, source_mask, target_ids, target_mask)
    sampler = SequentialSampler(data)
    return DataLoader(data, sampler = sampler, batch_size=batch_size)

def getDataLoader_qqp(args,input, tokenizer):
    inputs = ["qqp question1: "+doc1+" question2: "+doc2 for doc1, doc2 in zip(input['question1'], input['question2'])]
    tokenized_inputs = tokenizer(inputs, padding = True, truncation = True, return_tensors="pt")
    source_ids = tokenized_inputs['input_ids']
    source_mask = tokenized_inputs['attention_mask']

    targets = ["duplicate" if label == 1 else "not_duplicate" for label in input['label']]
    tokenized_outputs = tokenizer(targets, padding = True, return_tensors="pt")
    target_ids = tokenized_outputs['input_ids']
    target_mask = tokenized_outputs['attention_mask']
    
    batch_size = args.batch
    data = TensorDataset(source_ids, source_mask, target_ids, target_mask)
    sampler = SequentialSampler(data)
    return DataLoader(data, sampler = sampler, batch_size=batch_size)


def getDataLoader_wnli(args,input, tokenizer):
    inputs = ["wnli sentence1: " + doc1 + " sentence2: "+ doc2 for doc1, doc2 in zip(input['sentence1'], input['sentence2'])]
    tokenized_inputs = tokenizer(inputs, padding = True, truncation = True, return_tensors="pt")
    source_ids = tokenized_inputs['input_ids']
    source_mask = tokenized_inputs['attention_mask']

    targets = ['entailment' if label == 1 else 'not_entailment' for label in input['label']]
    tokenized_outputs = tokenizer(targets, padding = True, return_tensors="pt")
    target_ids = tokenized_outputs['input_ids']
    target_mask = tokenized_outputs['attention_mask']
    
    batch_size = args.batch
    data = TensorDataset(source_ids, source_mask, target_ids, target_mask)
    sampler = SequentialSampler(data)
    return DataLoader(data, sampler = sampler, batch_size=batch_size)


def glue_tasks(args, fixed_old=False):
    
    max_batches = {
        'cola':[[128,32],[128,64]],
        'mnli':[[16,4],[32,8]],
        'rte':[[16,4],[16,8]],
        'sst2':[[64,16],[128,32]],
        'qnli':[[16,4],[32,8]],
        'qqp':[[32,8],[64,16]],
        'wnli':[[32,8],[32,8]],
    }
    
    max_epochs = {
        'cola':[[10,10]],
        'mnli':[[1,1]],
        'rte':[[20,20]],
        'sst2':[[10,10]],
        'qnli':[[3,3]],
        'qqp':[[3,3]],
        'wnli':[[20,20]],
    }
        
    
    max_peft_batches = {
        'cola':16,
        'mnli':2,
        'rte':2,
        'sst2':8,
        'qnli':2,
        'qqp':2,
        'wnli':2,
        
    }
    max_peft_epochs = {
        'cola':16,
        'mnli':6,
        'rte':16,
        'sst2':8,
        'qnli':2,
        'qqp':8,
        'wnli':32,
    }
    
    if args.dataset == 'cola':
        getDataLoader = getDataLoader_cola
        splits = ['test','train','validation']

    elif args.dataset == 'mnli':
        getDataLoader = getDataLoader_mnli
        splits = ['test_matched','train','validation_matched']

    elif args.dataset == 'rte':
        getDataLoader = getDataLoader_rte
        splits = ['test','train','validation']

    elif args.dataset == 'sst2':
        getDataLoader = getDataLoader_sst2
        splits = ['test','train','validation']

    elif args.dataset == 'qnli':
        getDataLoader = getDataLoader_qnli
        splits = ['test','train','validation']

    elif args.dataset == 'qqp':
        getDataLoader = getDataLoader_qqp
        splits = ['test','train','validation']

    elif args.dataset == 'wnli':
        getDataLoader = getDataLoader_wnli
        splits = ['test','train','validation']

    else:
        raise Exception(f'{args.dataset} dataset is not defined!')
    
            
    if 'large' in args.base_model:
        args.batch = int(args.batch / 4)
        


    try:
        if args.test_batch:
            args.batch = 8
    except:
        None
        
    try:
        if args.modularized:
            if args.peft is not None:
                args.batch = max_peft_batches[args.dataset]
                args.epoch = max_peft_epochs[args.dataset]
                args.lr = 1e-4
            else:
                args.batch = max_batches[args.dataset][int(args.mixed)][int('large' in args.model)]
                args.epoch = max_epochs[args.dataset][0][int('large' in args.model)]
                args.lr = 3e-4
        if args.tiny_sample:
            args.batch = max_batches[args.dataset][int(args.mixed)][int('large' in args.model)]
            args.epoch = int (max_epochs[args.dataset][0][int('large' in args.model)] / 8)
            args.lr = 6e-4
    except Exception as e:
        print(e)
        raise e
        None    
    try:
        if args.early_stopping:
            args.epoch = 32
    except:
        None

    try:
        args.batch *= args.larger_batch
    except:
        print('No larger batch activated.')
        
    try:
        args.batch = int(args.batch / args.smaller_batch)
    except:
        print('No smaller batch activated.')

    
    dataset_test = load_dataset("nyu-mll/glue", args.dataset, split=splits[0])
    dataset_train = load_dataset("nyu-mll/glue", args.dataset, split=splits[1])
    dataset_validation = load_dataset("nyu-mll/glue", args.dataset, split=splits[2])
    tokenizer = T5Tokenizer.from_pretrained(args.model)
    return getDataLoader(args,dataset_train,tokenizer), getDataLoader(args,dataset_test,tokenizer),\
        getDataLoader(args,dataset_validation,tokenizer)


def add_glue_tasks(args):
    for dataset in args.datasets:
        args.dataset = dataset
        train_dataloader, test_dataloader, validation_dataloader = glue_tasks(args)
        args.train_dataloaders += [train_dataloader]
        args.test_dataloaders += [test_dataloader]
        args.validation_dataloaders += [validation_dataloader]
        args.epochs += [args.epoch]
        args.batchs += [args.batch]
