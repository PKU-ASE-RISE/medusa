import torch
from gradient_masking import get_task_vector

eps = 1e-8


def merge_W (w_list, method, args):
    # merge param matrixes with method
    # w_list should contain base model as w_list[0]
    # methods can use now:
        # _SUM     : lINEAR AVERAGE
        # _TA     : TASK ARITHMETIC
        # TIES_TIES: TIES-MERGING
        # DARE_SUM : DARE + LINEAR AVERAGE
        # DARE_TIES: DARE + TIES-MERGING
        
    w0 = torch.zeros_like(w_list[0]).to(w_list[0].device)
    m1,m2 = method.split('_')[0],method.split('_')[1]
    
    if m1 == 'DARE':
        P = 90
        for w in w_list[1:]:
            mask = (torch.randint(0,100,w.shape)>=P).to(w_list[0].device)
            w += -w + mask * (w-w_list[0]) / (1 - P/100) + w_list[0]
    elif m1 == 'TIES':
        top_k = 1 - args.lambda1 / 100
        w_l2 = w_list[1:]
        for j in range(len(w_l2)):
            w_l2[j] -= w_list[0]
        for param in w_l2:
            all_params = param.reshape(-1).abs()
            threshold = all_params.kthvalue(int(all_params.shape[0]*top_k)).values
            # print('threshold',threshold)
            mask = param.abs()>threshold
            param *= mask
        s = sum(w_l2)
        return (w_list[0] + args.lambda2 * (s > 0) * sum([w * (w>0) for w in w_l2]) / (sum([(w>0) for w in w_l2])+eps) \
                + args.lambda2 *  (s < 0) * sum([w * (w<0) for w in w_l2]) / (sum([(w<0) for w in w_l2])+eps)  )   
                   
    if m2 == 'SUM':
        for w in w_list[1:]:
            w0 += w
        return w0 / len(w_list[1:])
    elif m2 == 'TA':
        for w in w_list[1:]:
            w0 += (w-w_list[0]) * 0.4
        return w0 + w_list[0]
    elif m2 == 'TIES':
        w_l2 = w_list[1:]
        for j in range(len(w_l2)):
            w_l2[j] -= w_list[0]
        s = sum(w_l2)
        return (w_list[0] + (s > 0) * sum([w * (w>0) for w in w_l2]) / (sum([(w>0) for w in w_l2])+eps) \
                + (s < 0) * sum([w * (w<0) for w in w_l2]) / (sum([(w<0) for w in w_l2])+eps)  ) 
    else:
        raise Exception('Method Invalid')


        
                    

def merge_params(param_list, method, args):
    # take param_list[0] as the base model;
    ret = []
    for i in range(len(param_list[0])):
        ret += [merge_W([param[i].to(args.device) for param in param_list], method, args).to('cpu')]
    return ret

def merge_model(model_list, method, args):
    # model_list should contain base model as model_list[0]
    merged_param = merge_params([[param.clone() for name, param in model.named_parameters()] for model in model_list],\
        method, args)
    
    model = model_list[0]
    pretrained_dict = model.state_dict()
    id = 0
    for name, param in model.named_parameters():
        pretrained_dict[name] = merged_param[id]
        # param = merged_param[id]
        id += 1
    model.load_state_dict(pretrained_dict)
    return model
    
        
