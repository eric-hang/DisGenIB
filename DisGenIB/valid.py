import torch
from tqdm import tqdm
from torch.autograd import Variable
import numpy as np
from utils import count_accuracy,FuseCosineNetHead,log
import os
import torch.nn.functional as F
from datetime import *
def valid(epoch,embedding_net, netA,netZ,netG_A,netG_Y,classifier,max_val_acc,opt,dloader_val,file_path,attribute_dict,scale = None):
    x_entropy = torch.nn.CrossEntropyLoss()
    _, _, _,_,_= [x.eval() for x in (embedding_net, netA,netZ,netG_A,netG_Y)]
    val_accuracies = []
    val_losses = []

    gen_label = [_ for _ in range(opt.test_way)]
    gen_label = torch.repeat_interleave(torch.tensor(gen_label),opt.test_way * opt.val_shot).long().cuda()

    for i, batch in enumerate(tqdm(dloader_val(opt.seed)), 1):
        data_support, labels_support, \
        data_query, labels_query, k_all, _ = [
            x.cuda() for x in batch]

        test_n_support = opt.test_way * opt.val_shot
        test_n_query = opt.test_way * opt.val_query

        with torch.no_grad():
            if opt.use_feature:
                emb_support = data_support.reshape(-1,opt.feature_size)
                emb_query = data_query.reshape(-1,opt.feature_size)

            if opt.general:
                means_A, log_var_A = netA(F.normalize(emb_support))
                std = torch.exp(0.5 * log_var_A)
                eps = Variable(torch.randn(means_A.shape,device='cuda:0'))
                A = eps * std + means_A

                A = torch.repeat_interleave(A,opt.test_way,0)
            else:
                A = attribute_dict[torch.repeat_interleave(k_all.squeeze(),test_n_support).detach().cpu().tolist()].float().cuda()

            means_Z, log_var_Z = netZ(F.normalize(emb_support))
            std = torch.exp(0.5 * log_var_Z)
            eps = Variable(torch.randn(means_Z.shape,device='cuda:0'))
            Z = eps * std + means_Z
            Z = Z.repeat((opt.test_way,1))

                 
            x_gen_A = netG_A(Z,c=A)


            if opt.head == 'FuseCosNet':
                emb_support = emb_support.reshape(1,test_n_support, -1)

                emb_query = emb_query.reshape(1, test_n_query, -1)
                logits = FuseCosineNetHead(emb_query, emb_support, labels_support, opt.test_way, opt.val_shot,x_gen_A,x_gen_A,normalize_=False)
            
            if opt.head == 'FuseCosNet':
                loss = x_entropy(logits.reshape(-1, opt.test_way), labels_query.reshape(-1))
                acc = count_accuracy(logits.reshape(-1, opt.test_way), labels_query.reshape(-1))

        val_accuracies.append(acc.item())
        val_losses.append(loss.item())

    val_acc_avg = np.mean(np.array(val_accuracies))
    val_acc_ci95 = 1.96 * np.std(np.array(val_accuracies)) / np.sqrt(opt.val_episode)

    val_loss_avg = np.mean(np.array(val_losses))

    if val_acc_avg > max_val_acc:
        max_val_acc = val_acc_avg
        torch.save({"netA":netA.state_dict(),"netZ":netZ.state_dict(),'embedding_net':embedding_net.state_dict(),'netG_A':netG_A.state_dict(),'netG_Y':netG_Y.state_dict(),'classifier':classifier.state_dict(),'scale':scale}, \
                   os.path.join(opt.save_path, 'best_pretrain_model_meta_infer_val_{}w_{}s_{}_{}.pth'.format(opt.test_way, opt.val_shot, opt.head,opt.phase)))
        log(file_path,'Validation Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} ± {:.2f} % (Best)' \
            .format(epoch, val_loss_avg, val_acc_avg, val_acc_ci95))
        return max_val_acc
    else:
        torch.save({"netA":netA.state_dict(),"netZ":netZ.state_dict(),'embedding_net':embedding_net.state_dict(),'netG_A':netG_A.state_dict(),'netG_Y':netG_Y.state_dict(),'classifier':classifier.state_dict(),'scale':scale}, \
                   os.path.join(opt.save_path, 'latest_pretrain_model_meta_infer_val_{}w_{}s_{}_{}.pth'.format(opt.test_way, opt.val_shot, opt.head,opt.phase)))
        log(file_path,'Validation Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} ± {:.2f} %' \
            .format(epoch, val_loss_avg, val_acc_avg, val_acc_ci95))
        return None