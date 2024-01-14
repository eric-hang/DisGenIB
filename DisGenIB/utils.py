import os
import torch
import numpy as np
import torch.nn.functional as F
def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)

def check_dir(path):
    '''
    Create directory if it does not exist.
        path:           Path of directory.
    '''
    if not os.path.exists(path):
        os.mkdir(path)


def label2onehot(opt,label):
    if opt.dataset == 'miniImageNet' or opt.dataset == 'CIFAR-FS':
        z = torch.zeros(len(label),64).cuda()
    elif opt.dataset == 'tieredImageNet':
        z = torch.zeros(len(label),351).cuda()
    elif opt.dataset == 'FC100':
        z = torch.zeros(len(label),60).cuda()
    else:
        assert False
    z = z.scatter_(1,label.unsqueeze(1),1)
    return z


def one_hot(indices, depth):
    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    index = indices.view(indices.size()+torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1,index,1)
    
    return encoded_indicies

def MGD_estimate(query,prototypes,support_labels_one_hot,support,n_way,logits=None):
    scale = 10
    if logits is None:
        logits = torch.nn.functional.cosine_similarity(query.unsqueeze(2).expand(-1, -1, prototypes.shape[1], -1),
                                                   prototypes.unsqueeze(1).expand(-1, query.shape[1], -1, -1), dim=-1)
    assign = F.softmax(logits * scale, dim=-1)
    assign = torch.cat([support_labels_one_hot, assign], dim=1)
    assign_transposed = assign.transpose(1, 2)
    emb = torch.cat([support, query], dim=1)
    mean = torch.bmm(assign_transposed, emb)
    mean = mean.div(
        assign_transposed.sum(dim=2, keepdim=True).expand_as(mean)
    )
    diff = torch.pow(emb.unsqueeze(1).expand(-1, n_way, -1, -1) - mean.unsqueeze(2).expand(-1, -1, emb.shape[1], -1), 2)
    std = (assign_transposed.unsqueeze(-1).expand_as(diff) * diff).sum(dim=2) / assign_transposed.unsqueeze(-1).expand_as(diff).sum(dim=2)
    return logits,mean,std

def FuseCosineNetHead(query, support, support_labels, n_way, n_shot,boost_prototypes,x_gen_Y = None,normalize_=True,is_prototype=False):
    tasks_per_batch = query.size(0)
    n_support = support.size(1)
    n_query = query.size(1)
    d = query.size(2)

    assert (query.dim() == 3)
    assert (support.dim() == 3)
    assert (query.size(0) == support.size(0) and query.size(2) == support.size(2))
    assert (n_support == n_way * n_shot)

    support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * n_support), n_way)
    support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, n_support, n_way)


    labels_train_transposed = support_labels_one_hot.transpose(1, 2)

    prototypes = torch.bmm(labels_train_transposed, support)
    
    prototypes = prototypes.div(
        labels_train_transposed.sum(dim=2, keepdim=True).expand_as(prototypes)
    )

    boost_prototypes = boost_prototypes.reshape(tasks_per_batch,n_way,n_support,-1)[:,:,:,:].mean(dim=2)

    x_gen_Y = x_gen_Y.reshape(tasks_per_batch,n_way,n_support,-1)[:,:,:,:].mean(dim=2)

    if normalize_:
        boost_prototypes = F.normalize(boost_prototypes)
        query = F.normalize(query)

    _,mean_1,std_1 = MGD_estimate(query,prototypes,support_labels_one_hot,support,n_way)
    _,mean_2,std_2 = MGD_estimate(query,boost_prototypes,support_labels_one_hot,support,n_way)
    _,mean_3,std_3 = MGD_estimate(query,x_gen_Y,support_labels_one_hot,support,n_way)

    prototypes_gen = (mean_1 * std_2 + mean_2 * std_1) / (std_2 + std_1)
    prototypes_counterfactual = (mean_1 * std_3 + mean_3 * std_1) / (std_3 + std_1)


    _,mean_1,std_1 = MGD_estimate(query,prototypes_gen,support_labels_one_hot,support,n_way)
    _,mean_2,std_2 = MGD_estimate(query,prototypes_counterfactual,support_labels_one_hot,support,n_way)

    prototypes = (mean_1 * std_2 + mean_2 * std_1) / (std_2 + std_1)

    logits = torch.nn.functional.cosine_similarity(query.unsqueeze(2).expand(-1, -1, prototypes.shape[1], -1),
                                                   prototypes.unsqueeze(1).expand(-1, query.shape[1], -1, -1), dim=-1)
    return logits

def count_accuracy(logits, label):
    pred = torch.argmax(logits, dim=1).view(-1)
    label = label.view(-1)
    accuracy = 100 * pred.eq(label).float().mean()
    return accuracy

def get_attr(opt):
    if opt.dataset == 'miniImageNet':
        matcontent = np.load('path to prior',allow_pickle=True)
        attribute_dict = torch.from_numpy(matcontent['attribute']).float().squeeze(1)
        attribute_dict /= attribute_dict.pow(2).sum(1).sqrt().unsqueeze(1).expand(attribute_dict.size(0),attribute_dict.size(1))

    return attribute_dict