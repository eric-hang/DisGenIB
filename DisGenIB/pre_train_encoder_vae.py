import torch
import os
from utils import check_dir, log, label2onehot,get_attr
from get_model_data import get_model
from tqdm import tqdm
from tcvae.tcvae import anneal_kl
import numpy as np
import torch.nn.functional as F
from valid import valid
def pre_train_encoder_vae(opt, dataset_train, dataset_val, dataset_test, data_loader):
    data_loader_pre = torch.utils.data.DataLoader
    dloader_train = data_loader_pre(
            dataset=dataset_train,
            batch_size=512,
            shuffle=True,
            num_workers=0
        )
    dloader_val = data_loader(
        dataset=dataset_val,
        nKnovel=opt.test_way,
        nKbase=0,
        nExemplars=opt.val_shot,  # num training examples per novel category
        nTestNovel=opt.val_query * opt.test_way,  # num test examples for all the novel categories
        nTestBase=0,  # num test examples for all the base categories
        batch_size=1,
        num_workers=0,
        epoch_size=1 * opt.val_episode,  # num of batches per epoch
    )

    check_dir('./experiments/')
    check_dir(opt.save_path)

    log_file_path = os.path.join(opt.save_path, "pretrain_encoder_log.txt")
    log(log_file_path, str(vars(opt)))

    (embedding_net,netA,netZ,netG_A,netG_Y,classifier) = get_model(opt)

    if opt.general:
        optimizer = torch.optim.Adam([{'params': netA.parameters()},
                                    {'params': netG_A.parameters()},
                                    {'params': netG_Y.parameters()},
                                    {'params': netZ.parameters()},
                                    {'params': classifier.parameters()}],lr=opt.lr,betas=(opt.beta1, 0.999),weight_decay=5e-4)
    else:
        optimizer = torch.optim.Adam([{'params': netG_A.parameters()},
                                    {'params': netZ.parameters()},],lr=opt.lr,betas=(opt.beta1, 0.999),weight_decay=5e-4)
    max_val_acc = 0.0

    attribute_dict = get_attr(opt).cuda()

    for epoch in range(1, opt.num_epoch + 1):
        epoch_learning_rate = 0.1
        for param_group in optimizer.param_groups:
            epoch_learning_rate = param_group['lr']

        log(log_file_path, 'Train Epoch: {}\tLearning Rate: {}'.format(
            epoch, epoch_learning_rate))

        if opt.general:
            embedding_net.eval()
            _, _,_,_,_= [x.train() for x in (netA,netZ,netG_A,netG_Y,classifier)]
        else:
            _,_,_,_= [x.eval() for x in (netA,embedding_net,netG_Y,classifier)]
            _,_= [x.train() for x in (netZ,netG_A)]

        train_losses = []
        beta = anneal_kl(opt, epoch)

        for i, batch in enumerate(tqdm(dloader_train)):
            # break
            # time1 = time.time()
            data, labels = [x.cuda() for x in batch]
            if opt.pre_head == 'LinearNet' or opt.pre_head == 'CosineNet':
                if opt.use_nohub:
                    emb = data.reshape(-1,opt.feature_size)
                else:
                    nb, ns, nc, nw, nh = data.shape
                    data = data.reshape(nb*ns, nc, nw, nh)
                    with torch.no_grad():
                        emb = embedding_net(data)
                if opt.general:
                    labels = torch.repeat_interleave(labels,opt.train_shot)
                    means_A, log_var_A = netA(F.normalize(emb))
                    std = torch.exp(0.5 * log_var_A)
                    eps = torch.randn(std.shape,device='cuda:0')
                    A = eps * std + means_A
                    labels_ = label2onehot(opt,labels)
                else:
                    A = attribute_dict[torch.repeat_interleave(labels,opt.train_shot).detach().cpu().tolist()].float()


                means_Z, log_var_Z = netZ(F.normalize(emb))
                std = torch.exp(0.5 * log_var_Z)
                eps = torch.randn(std.shape,device='cuda:0')
                Z = eps * std + means_Z
                
                
                recon_A = netG_A(Z,c=A)

                if opt.general:
                    recon_Y = netG_Y(Z,c=labels_)


                input_eps = torch.randn(emb.shape,device='cuda:0')
                hard_input = emb + opt.add_noise * input_eps


                recon_loss_A = torch.sum(torch.pow(F.normalize(recon_A) - F.normalize(hard_input).detach(), 2), 1).mean()
                KL_XZ = -0.5 * torch.sum(1 + log_var_Z - means_Z.pow(2) - log_var_Z.exp()) / recon_A.size(0)


                if opt.general:
                    recon_loss_Y = torch.sum(torch.pow(F.normalize(recon_Y) - F.normalize(hard_input).detach(), 2), 1).mean()
                    logits = classifier(A)
                    cls_loss = torch.nn.CrossEntropyLoss()(logits,labels)
                    KL_XA = -0.5 * torch.sum(1 + log_var_A - means_A.pow(2) - log_var_A.exp()) / recon_A.size(0)
                    pass

            if opt.general:
                loss = recon_loss_A + 2*recon_loss_Y + beta * (2*KL_XZ + KL_XA + cls_loss) #
            else:
                loss = recon_loss_A + beta * KL_XZ #

            train_losses.append(loss.item())


            if (i % (len(dloader_train)//5) == 0):
                train_loss_mean = np.mean(np.array(train_losses))
                log(log_file_path, 'Train Epoch: {}\tBatch: [{}]\tLoss: {:.3f}\tavg_loss: {:.3f}\t'.format(
                    epoch, i, loss.item(),train_loss_mean))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return_acc = valid(epoch, embedding_net,netA,netZ,netG_A,netG_Y,classifier,max_val_acc,opt,dloader_val,log_file_path,attribute_dict)
        if return_acc is not None:
            max_val_acc = return_acc