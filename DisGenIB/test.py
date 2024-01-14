import torch
from tqdm import tqdm
from torch.autograd import Variable
import numpy as np
from utils import count_accuracy,FuseCosineNetHead,get_attr
import torch.nn.functional as F
from get_model_data import get_model
def test(opt,dataset_test,data_loader):
    dloader_test = data_loader(
        dataset=dataset_test,
        nKnovel=opt.test_way,
        nKbase=0,
        nExemplars=opt.val_shot,  # num training examples per novel category
        nTestNovel=opt.val_query * opt.test_way,  # num test examples for all the novel categories
        nTestBase=0,  # num test examples for all the base categories
        batch_size=1,
        num_workers=0,
        epoch_size=1 * opt.val_episode,  # num of batches per epoch
    )

    (embedding_net,netA,netZ,netG_A,netG_Y,classifier) = get_model(opt)

    saved_models = torch.load('path to pretrained model')


    embedding_net.load_state_dict(saved_models['embedding_net'])
    embedding_net.eval()
    netA.load_state_dict(saved_models['netA'])
    netA.eval()
    netZ.load_state_dict(saved_models['netZ'])
    netZ.eval()
    netG_A.load_state_dict(saved_models['netG_A'])
    netG_A.eval()


    attribute_dict = get_attr(opt).cuda()

    _, _, _,_ = [x.eval() for x in (embedding_net, netA,netZ,netG_A)]

    test_accuracies = []
    test_losses = []

    for i, batch in enumerate(tqdm(dloader_test(opt.seed)), 1):
        data_support, labels_support, \
        data_query, labels_query, k_all, _ = [
            x.cuda() for x in batch]

        test_n_support = opt.test_way * opt.val_shot
        test_n_query = opt.test_way * opt.val_query
        with torch.no_grad():
            if opt.feature:
                emb_support = data_support.reshape(-1,opt.feature_size)
                emb_query = data_query.reshape(-1,opt.feature_size)
            

            means_Z, log_var_Z = netZ(F.normalize(emb_support))
            std = torch.exp(0.5 * log_var_Z)
            eps = Variable(torch.randn(means_Z.shape,device='cuda:0'))
            z = eps * std + means_Z
            z = z.repeat((opt.test_way,1))

            if opt.general:
                means_A, log_var_A = netA(F.normalize(emb_support))
                std = torch.exp(0.5 * log_var_A)
                eps = Variable(torch.randn(means_A.shape,device='cuda:0'))
                a = eps * std + means_A
                a = torch.repeat_interleave(a,opt.test_way,0)
            else:
                a = attribute_dict[torch.repeat_interleave(k_all.squeeze(),test_n_support).detach().cpu().tolist()].float().cuda()

            x_gen = netG_A(z,c=a)
            
            if opt.head == 'FuseCosNet':
                emb_support = emb_support.reshape(1,test_n_support, -1)
                emb_query = emb_query.reshape(1, test_n_query, -1)

                logits = FuseCosineNetHead(emb_query, emb_support, labels_support, opt.test_way, opt.val_shot,x_gen,x_gen,normalize_=False,is_prototype = False)
            


        if opt.head == 'FuseCosNet':
            acc = count_accuracy(logits.reshape(-1, opt.test_way), labels_query.reshape(-1)).item()

        test_accuracies.append(acc)
    
    test_acc_avg = np.mean(np.array(test_accuracies))
    test_acc_ci95 = 1.96 * np.std(np.array(test_accuracies)) / np.sqrt(opt.val_episode)

    test_loss_avg = np.mean(np.array(test_losses))

    print('Test Loss: {:.4f}\tAccuracy: {:.2f} ± {:.2f} % (Best)' \
        .format(test_loss_avg, test_acc_avg, test_acc_ci95))