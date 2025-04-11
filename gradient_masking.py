
import torch
from torch.optim import AdamW,Adam
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_scheduler
EXIST_SP = False
def list_add(a,b):
    return [i+j for i,j in zip(a,b)]
def list_sub(a,b):
    return [i-j for i,j in zip(a,b)]

def get_param_list(model, clone=False, get_all=0):
    params = []
    for name, param in model.named_parameters():
        if get_all == 0 and param.requires_grad is False:
            continue
        if clone:
            params += [param.cpu().clone()]  
        else:
            params += [param] 
    return params

def get_name_list(model):
    params = []
    for name, param in model.named_parameters():
        params += [name]  
    return params

def get_task_vector(params, base_params):
    ret = []
    for i,j in zip(params,base_params):
        ret += [i-j]
    return ret

def get_sign_conflict(a_list, b_list):
    acc,tot=0,0
    for a,b in zip(a_list,b_list):
        acc+=((a>0)*(b<0)).sum()+((a<0)*(b>0)).sum()
        tot+=a.reshape(-1).shape[0]
    return acc/tot

def keep_topk(param, args):
    p=(1-args.top_k/100)
    
    all_params = param.reshape(-1).abs()
    if args.top_k>=99:
        try:
            threshold = -torch.inf
        except:
            threshold = -float('inf')
    else:
        threshold = all_params.kthvalue(int(all_params.shape[0]*p)).values
    mask = param.abs()>threshold
    return param * mask

def keep_topk_params(params, args):
    return [keep_topk(p,args)for p in params]

class no_mask:
    def __init__(self, model):
        self.params, self.masks = [],[]
        if model is not None:
            self.model = model
            self.device = model.device
            self.params = get_param_list(self.model)
    
    def apply_mask(self, model):
        try:
            self.model 
            raise Exception('Old model not released!')
        except:
            self.model = model
            self.device = model.device
            self.params = get_param_list(self.model)

            for name, mask in zip(get_name_list(self.model), self.masks):
                if EXIST_SP and 'relative_attention_bias.weight' in name:
                    mask.fill_(False)
        
        
    def mask_count(self):
        acc,tot = 0,0
        for i in self.masks:
            acc+=i.sum()
            tot+=i.reshape(-1).shape[0]
        return acc/tot
        
    def generate_mask(self, dataset, args):
        self.masks = []
        for name, param in self.model.named_parameters():
            mask = torch.ones_like(param)
            if EXIST_SP and 'relative_attention_bias.weight' in name:
                mask = torch.zeros_like(param)
            self.masks += [mask>0.5]
            # print(mask.sum(),'///',mask.reshape(-1).shape[0])
    def __call__(self):
        for param,mask in zip(self.params,self.masks):
            param.grad = param.grad.masked_fill(~mask.to(self.device), 0)
            # param.grad *= mask.to(self.device)
            # param.grad[~mask] = 0
            
class mutual_mask(no_mask):
    def generate_mask(self, delta_param, delta_sum, args):
        for param, sum in zip(delta_param, delta_sum):
            self.masks += \
                [(param>0) * (sum>0) + (param<0) * (sum<0)]
                
class half_mask(no_mask):
    def generate_mask(self, id):
        self.masks = []
        for name, param in self.model.named_parameters():
            mask = torch.ones_like(param)
            border = int(param.shape[0]/2)
            if id == 1:
                mask[border:,...] *= 0
            else:
                mask[:border,...] *= 0
            self.masks += [mask>0.5]
                
class magnitude_mutual_mask(no_mask):
    def generate_mask(self, reference, delta_param, delta_sum, args, fill_up=None):
        for ref, param, sum in zip(reference, delta_param, delta_sum):
            mask = ((param >= sum) & (ref>0))
            self.masks += \
                [mask]
            if fill_up is not None:
                sum += -sum + sum.masked_fill(mask, fill_up)

        
            
class gradient_mask(no_mask):
    def generate_mask(self, dataset, args):
        self.masks = []
        
        self.model.train()
        
        for id,batch in enumerate(dataset):
            _batch = {k: torch.stack(batch[k],axis=-1).to(self.device) 
                      for k in ['input_ids', 'labels']}
            outputs = self.model(**_batch)
            loss = outputs.loss
            loss.backward()
        
        all_params = torch.concat([param.grad.reshape(-1).abs() for param in self.params], axis=-1)
        if args.top_k>=99:
            try:
                threshold = -torch.inf
            except:
                threshold = -float('inf')
        else:
            threshold = all_params.kthvalue(int(all_params.shape[0]*(1-args.top_k/100))).values
        print(f'generating mask by threshold {threshold}')
        for name, param in self.model.named_parameters():
            mask = param.grad.abs()>threshold
            if EXIST_SP and 'relative_attention_bias.weight' in name:
                mask = torch.zeros_like(param)
            self.masks += [mask.to(self.device)]
            # print(param.grad)
            param.grad *= 0
        
class gradient_mask_block(no_mask):
    def generate_mask(self, dataset, args):
        self.masks = []
        
        self.model.train()
        
        for id,batch in enumerate(dataset):
            _batch = {k: torch.stack(batch[k],axis=-1).to(self.device) 
                      for k in ['input_ids', 'labels']}
            outputs = self.model(**_batch)
            loss = outputs.loss
            loss.backward()
     
        
        for name, param in self.model.named_parameters():
            all_params = param.grad.reshape(-1).abs()
            if args.top_k>=99:
                try:
                    threshold = -torch.inf
                except:
                    threshold = -float('inf')
            else:
                threshold = all_params.kthvalue(int(all_params.shape[0]*(1-args.top_k/100))).values
            print(f'generating mask by threshold {threshold}')
            # print(param.grad)
            
            mask = param.grad.abs()>threshold
            if EXIST_SP and 'relative_attention_bias.weight' in name:
                mask = torch.zeros_like(param)
            self.masks += [mask.to(self.device)]
            param.grad *= 0

class warmup_gradient_mask_block(gradient_mask_block):
    def generate_mask(self, dataset, args):
        self.model.train()
        
        optimizer = Adam(self.model.parameters(), lr=args.lr)
        num_training_steps = args.epochs * len(dataset)
        lr_scheduler = get_scheduler(
            name="linear", 
            optimizer=optimizer, 
            num_warmup_steps = args.num_warmup_steps,
            num_training_steps=num_training_steps
        )
        for epoch in range(args.warmup_epochs):
            for id,batch in enumerate(dataset):
                _batch = {k: torch.stack(batch[k],axis=-1).to(self.device) \
                    for k in ['input_ids', 'labels']}
                outputs = self.model(**_batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
            
        super().generate_mask(dataset, args)
        
     
            
class random_mask(no_mask):
    def generate_mask(self, dataset, args):
        self.masks = []

        for name, param in self.model.named_parameters():
            mask = torch.randint(0,100,param.shape)
            mask = mask<args.top_k
            if EXIST_SP and 'relative_attention_bias.weight' in name:
                mask = torch.zeros_like(param)
            self.masks += [mask.to(self.device)]