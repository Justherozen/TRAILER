import argparse
import os
import random
import time
import pickle
import math
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm
from datetime import datetime
from models.build_model import build_model_pu
from datasets.datasets import get_dataset_trans
from utils.evaluate_utils import hungarian_evaluate
from utils.utils import *
from utils.losses import *
from utils.sinkhorn_knopp import SinkhornKnopp

def main():
    parser = argparse.ArgumentParser(description='Base Training')
    parser.add_argument('--data-root', default=f'data', help='directory to store data')
    parser.add_argument('--split-root', default=f'random_splits', help='directory to store datasets')
    parser.add_argument('--out', default=f'outputs', help='directory to output the result')
    parser.add_argument('--num-workers', type=int, default=4, help='number of workers')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100', 'svhn', 'tinyimagenet', 'oxfordpets', 'aircraft', 'stanfordcars', 'imagenet100', 'herbarium'], help='dataset name')
    parser.add_argument('--lbl-percent', type=int, default=50, help='percent of labeled data')
    parser.add_argument('--novel-percent', default=50, type=int, help='percentage of novel classes, default 50')
    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run, deafult 50')
    parser.add_argument('--batch-size', default=200, type=int, help='train batchsize')
    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate, default 1e-3')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', type=int, default=-1, help="random seed (-1: don't use random seed)")
    parser.add_argument('--no-progress', action='store_true', help="don't use progress bar")
    parser.add_argument('--no-class', default=10, type=int, help='total classes')
    parser.add_argument('--threshold', default=0.5, type=float, help='pseudo-label threshold, default 0.50')
    parser.add_argument('--threshold2', default=0.5, type=float, help='pseudo-label threshold, default 0.50')
    parser.add_argument('--split-id', default='split_0', type=str, help='random data split number')
    parser.add_argument('--ssl-indexes', default='', type=str, help='path to random data split')
    parser.add_argument('--rho', default='0.3,0.9', type=str, help='pseudo-label threshold, default 0.50')
    parser.add_argument('--warmup', default=15, type=int, help='total classes')
    parser.add_argument('--chosen_neighbors', default=100, type=int, help='total classes')
    parser.add_argument('--entropy_q', default=0.3, type=float, help='total classes')
    parser.add_argument('--temparature', default=0.3, type=float, help='total classes')
    parser.add_argument('--knn_weight', default=0.2, type=float, help='total classes')
    args = parser.parse_args()
    run_started = datetime.today().strftime('%d-%m-%y_%H%M')
    split_id = f'split_{random.randint(1, 100000)}'
    args.split_id = split_id
    args.ssl_indexes = f'{args.split_root}/{args.dataset}_{args.lbl_percent}_{args.novel_percent}_{args.split_id}.pkl'
    args.exp_name = f'dataset_{args.dataset}_lbl_percent_{args.lbl_percent}_novel_percent_{args.novel_percent}_split_id_{args.split_id}_{run_started}'
    
    args.out = os.path.join(args.out, args.exp_name)
    os.makedirs(args.out, exist_ok=True)

    best_acc = 0    
    best_acc_trans = 0
    best_acc_novel_trans = 0
    

    writer = SummaryWriter(logdir=args.out)

    with open(f'{args.out}/score_logger_base.txt', 'a+') as ofile:
        ofile.write('************************************************************************\n\n')
        ofile.write(' | '.join(f'{k}={v}' for k, v in vars(args).items()))
        ofile.write('\n\n************************************************************************\n')
    
    args.n_gpu = torch.cuda.device_count()
    args.dtype = torch.float32
    if args.seed != -1:
        set_seed(args)

    # set dataset specific parameters
    if args.dataset == 'cifar10':
        args.no_class = 10
    elif args.dataset == 'cifar100':
        args.no_class = 100

    args.data_root = os.path.join(args.data_root, args.dataset)
    os.makedirs(args.data_root, exist_ok=True)
    os.makedirs(args.split_root, exist_ok=True)

    # Load dataset
    args.no_known = args.no_class - int((args.novel_percent*args.no_class)/100)
    lbl_dataset, unlbl_dataset, pl_dataset, test_dataset_known, test_dataset_novel, test_dataset_all, test_dataset_known_trans, test_dataset_novel_trans, test_dataset_all_trans = get_dataset_trans(args)

    # Create dataloaders
    unlbl_batchsize = int((float(args.batch_size) * len(unlbl_dataset))/(len(lbl_dataset) + len(unlbl_dataset)))
    lbl_batchsize = args.batch_size - unlbl_batchsize
    args.iteration = (len(lbl_dataset) + len(unlbl_dataset)) // args.batch_size

    train_sampler = RandomSampler
    lbl_loader = DataLoader(lbl_dataset, sampler=train_sampler(lbl_dataset), batch_size=lbl_batchsize, num_workers=args.num_workers, drop_last=True)
    unlbl_loader = DataLoader(unlbl_dataset, sampler=train_sampler(unlbl_dataset), batch_size=unlbl_batchsize, num_workers=args.num_workers, drop_last=True)
    pl_loader = DataLoader(pl_dataset, sampler=SequentialSampler(pl_dataset), batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False)
    # Transductive setting
    test_loader_known_trans = DataLoader(test_dataset_known_trans, sampler=SequentialSampler(test_dataset_known_trans), batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False)
    test_loader_novel_trans = DataLoader(test_dataset_novel_trans, sampler=SequentialSampler(test_dataset_novel_trans), batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False)
    test_loader_all_trans = DataLoader(test_dataset_all_trans, sampler=SequentialSampler(test_dataset_all_trans), batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False)


    # Create model, start from SimCLR pretraining checkpoint
    if args.dataset == 'cifar10':
        state_dict = torch.load('/home/xrx/openworld_cvpr/base/pretrained/simclr_cifar_10.pth.tar')
        args.chosen_neighbors = 50
        args.knn_weight = 0.2
        args.entropy_q = 0.3
        args.temparature = 0.3
        args.rho = '0.3,0.9'
    elif args.dataset == 'cifar100':
        state_dict = torch.load('/home/xrx/openworld_cvpr/base/pretrained/simclr_cifar_100.pth.tar')
        args.chosen_neighbors = 10
        args.knn_weight = 0.5
        args.entropy_q = 0.1
        args.temparature = 0.5
        args.rho = '0.5,1'

    [args.rho_start, args.rho_end] = [float(item) for item in args.rho.split(',')]
    model = build_model_pu(args)
    ema_model = build_model_pu(args,ema=True)
    model.load_state_dict(state_dict, strict=False)

    print(' | '.join(f'{k}={v}' for k, v in vars(args).items()))
    ema_model = ema_model.cuda()
    model = model.cuda()
    ema_optimizer= WeightEMA(0.95, model, ema_model)
    #EMA update for prediction of PU classifier
    sinkhorn = SinkhornKnopp(num_iters_sk=3,epsilon_sk=0.05,imb_factor=1)

    # optimizer
    if torch.cuda.device_count() > 1:
        optimizer = torch.optim.Adam(model.module.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=0.01*args.lr)
    start_epoch = 0
    if args.resume:
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    model.zero_grad()
    cur_M = model.classifier_new.ori_M.T.cuda()
    train_stat = {
        'all_prototype': torch.zeros(args.no_class,512).cuda(),
        'feature_con_bank': torch.zeros(len(unlbl_loader.dataset),128).cuda(),
        'feature_all': torch.zeros(len(unlbl_loader.dataset),512).cuda(),
        'target_pu_max': -math.inf,
        'pseudo_list_all': torch.zeros((len(unlbl_loader.dataset),args.no_class)),
        'prob': np.zeros(len(unlbl_loader.dataset)),
    }
    train_stat['all_prototype'][:args.no_known,:] = cur_M[:args.no_known,:]
    for epoch in range(start_epoch, args.epochs):
        #training
        train_stat = train(args, lbl_loader, unlbl_loader, model, optimizer,ema_optimizer,scheduler, epoch,sinkhorn,train_stat)
        #test
        test_acc_known_trans = test_known(args, test_loader_known_trans, model, epoch)
        novel_cluster_results_trans = test_cluster(args, test_loader_novel_trans, model, epoch, offset=args.no_known)
        all_cluster_results_trans = test_cluster(args, test_loader_all_trans, model, epoch)
        test_acc_trans = all_cluster_results_trans["acc"]
        test_acc_novel_trans = novel_cluster_results_trans["acc"]

        is_best_trans = test_acc_novel_trans > best_acc_novel_trans
        best_acc_trans = max(test_acc_trans, best_acc_trans)
        best_acc_novel_trans = max(test_acc_novel_trans,best_acc_novel_trans)

        print(f'epoch: {epoch}, acc-known-trans: {test_acc_known_trans}')
        print(f'epoch: {epoch}, acc-novel-trans: {novel_cluster_results_trans["acc"]}, nmi-novel: {novel_cluster_results_trans["nmi"]}')
        print(f'epoch: {epoch}, acc-all-trans: {all_cluster_results_trans["acc"]}, nmi-all: {all_cluster_results_trans["nmi"]}, best-acc: {best_acc_trans}, best-acc-novel: {best_acc_novel_trans}')


        model_to_save = model.module if hasattr(model, "module") else model    
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model_to_save,
            'cur_m': model.module.classifier_new.ori_M if hasattr(model, "module") else model.classifier_new.ori_M,
            'acc': test_acc_trans,
            'best_acc': best_acc_trans,
            'optimizer': optimizer.state_dict()
        }, is_best_trans, args.out, tag='base')

    writer.close()


def train(args, lbl_loader, unlbl_loader, model, optimizer, ema_optimizer,scheduler,epoch,sinkhorn,train_stat):
    batch_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    rho = args.rho_start + (args.rho_end - args.rho_start) * linear_rampup(epoch, args.warmup ,30)
    w = linear_rampup(epoch,args.warmup,30)
    if not args.no_progress:
        p_bar = tqdm(range(args.iteration))

    train_loader = zip(lbl_loader, unlbl_loader)

    all_prototype = train_stat['all_prototype']
    #For normalization of PU classifier 
    target_pu_max_old = train_stat['target_pu_max']
    target_pu_max = -math.inf 
    
    for batch_idx, (data_lbl, data_unlbl) in enumerate(train_loader):
        (inputs_l_w, inputs_l_s), targets_l, _, index_l = data_lbl 
        (inputs_u_w, inputs_u_s), targets_u, _, index_u = data_unlbl 
        batch_prob = torch.Tensor(train_stat['prob'][index_u]).cuda()
        inputs = interleave(
            torch.cat((inputs_l_w,inputs_l_s, inputs_u_w, inputs_u_s)), 4).cuda()
        targets_l = targets_l.cuda()
        targets_u = targets_u.cuda()
        batch_l = inputs_l_w.shape
        model.train()
        feat, logits, feat_norm, feat_con, outputs_pu = model(inputs)
        target_pu_max = max(target_pu_max,outputs_pu[:, 1].max().exp())
        logits = de_interleave(logits, 4)
        logits_l_w, logits_l_s = logits[:2*batch_l[0]].chunk(2)
        logits_u_w, logits_u_s = logits[2*batch_l[0]:].chunk(2)        
        feat_norm = de_interleave(feat_norm, 4)
        feat_norm_l_w, feat_norm_l_s = feat_norm[:2*batch_l[0]].chunk(2)
        feat_norm_u_w, feat_norm_u_s = feat_norm[2*batch_l[0]:].chunk(2)
        feat_con = de_interleave(feat_con, 4)
        feat_con_l_w, feat_con_l_s = feat_con[:2*batch_l[0]].chunk(2)
        feat_con_u_w, feat_con_u_s = feat_con[2*batch_l[0]:].chunk(2)

        #Rough assignment of pseudo-labels via optimal transport (Sinkhorn)       
        class_logit = torch.cat((logits_l_w, logits_u_w), 0)
        pseudo_label_all = sinkhorn(class_logit.detach().clone())
        pseudo_label = pseudo_label_all[logits_l_w.shape[0]:]

        #KNN-augmented Contrastive learning
        batch_u = inputs_u_w.shape[0]
        bank_length = train_stat['feature_con_bank'].shape[0]
        train_stat['feature_con_bank'][index_u] = feat_con_u_w.detach().clone()
        train_stat['feature_all'][index_u] = feat_norm_u_w.detach().clone()
        with torch.no_grad():
            cosine_corr = torch.matmul(feat_con_u_w,train_stat['feature_con_bank'].T)
            _, knn_index = torch.topk(cosine_corr, k = args.chosen_neighbors, dim=-1, largest=True)
            mask_knn = torch.scatter(torch.zeros([batch_u,bank_length]).cuda(), 1, knn_index[:,1:], 1).detach().clone()
        loss_con_knn = supcon_knn(features=feat_con_u_w, features_all=train_stat['feature_con_bank'].detach().clone(), mask=mask_knn)
        
        #Training objectives of PU classifier
        outputs_pu = de_interleave(outputs_pu, 4)
        logits_pu_all = outputs_pu[:, 1]
        log_pu_l = logits_pu_all[:2*batch_l[0]].chunk(2)[0]
        log_pu_u = logits_pu_all[2*batch_l[0]:].chunk(2)[0]
        output_pu_u = outputs_pu[2*batch_l[0]:].chunk(2)[0]
        var_loss = torch.logsumexp(log_pu_u, dim=0) - math.log(len(log_pu_u)) - 1 * torch.mean(log_pu_l)
        target_u_pu = output_pu_u[:, 1].exp()
        target_l_pu = torch.ones(len(inputs_l_w), dtype=torch.float32)
        target_l_pu = target_l_pu.cuda() if torch.cuda.is_available() else target_l_pu
        rand_perm = torch.randperm(inputs_l_w.size(0))
        data_l_perm, target_l_perm = inputs_l_w[rand_perm], target_l_pu[rand_perm]
        m = torch.distributions.beta.Beta(0.3, 0.3)
        lam = m.sample()
        data = lam * inputs_u_w + (1 - lam) * data_l_perm
        target = lam * target_u_pu + (1 - lam) * target_l_perm
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        _,_,_,_,out_log_pu_all = model(data)
        reg_mix_log = ((torch.log(target) - out_log_pu_all[:, 1]) ** 2).mean()
        loss_pu = var_loss + 0.03 * reg_mix_log
        
        #Label refinery via PU prediction
        with torch.no_grad():
            pu_labels_u = (targets_u < args.no_known).int()
            target_u_pu_t = target_u_pu/target_pu_max_old
            target_pos = target_u_pu_t > 0.5
            target_neg = target_u_pu_t < 0.5
            mask_P=torch.ones(args.no_class)
            mask_N=torch.ones(args.no_class)
            for i in range(args.no_class) :
                if(i>=args.no_known):
                    mask_P[i]=0
                else :
                    mask_N[i]=0
            mask_P=mask_P.cuda()
            mask_N=mask_N.cuda()
            if epoch > args.warmup:
                for i in range(pseudo_label.size()[0]) :
                    if(target_neg[i]):
                        pseudo_label[i] = pseudo_label[i]*mask_N
                    if(target_pos[i]):
                        pseudo_label[i] = pseudo_label[i]*mask_P    
        max_probs_pl, targets_u_pl = torch.max(pseudo_label, dim=-1)
        with torch.no_grad():
            train_stat['pseudo_list_all'][index_u] = (pseudo_label.detach().cpu())
        #Selected samples for known/novel classes
        tmp_prob_novel = (batch_prob) * (targets_u_pl>=args.no_known)
        index_chosen_novel =  torch.where(tmp_prob_novel == 1)[0]
        tmp_prob_known = (batch_prob) * (targets_u_pl<args.no_known)
        index_chosen_known =  torch.where(tmp_prob_known == 1)[0]
        
        
        
        #Entropy regularization
        q = torch.Tensor([1] * args.no_known + [1] * (args.no_class-args.no_known)).cuda()
        q = q / q.sum()
        loss_reg = -1 * entropy(torch.mean(F.softmax(class_logit/args.entropy_q, dim=1), 0), input_as_probabilities = True,q=q)
            
        #Unsupervised contrastive loss
        feat_unsupcon = torch.cat((feat_con_l_w,feat_con_u_w,feat_con_l_s,feat_con_u_s),0)
        unsupcon_logits, unsup_labels = info_nce_logits(features=feat_unsupcon)
        loss_con_unsup = torch.nn.CrossEntropyLoss()(unsupcon_logits, unsup_labels)
        

        #Classification loss
        loss_ce_sup = F.cross_entropy(class_logit[:batch_l[0]]/args.temparature, targets_l)
        loss_ce_pseudo_novel = (F.cross_entropy(logits_u_s/args.temparature, targets_u_pl, reduction='none')[index_chosen_novel]).mean() if len(index_chosen_novel) else 0
        loss_ce_pseudo_known = (F.cross_entropy(logits_u_s/args.temparature, targets_u_pl, reduction='none')[index_chosen_known]).mean() if len(index_chosen_known) else 0
        loss_ce_unsup =  loss_ce_pseudo_novel + loss_ce_pseudo_known

        #Prototypical loss
        logits_u_proto = torch.matmul(feat_norm_u_w, all_prototype.T.detach().clone())
        loss_proto = (F.cross_entropy(logits_u_proto/args.temparature, targets_u_pl, reduction='none')[index_chosen_novel]).mean() if len(index_chosen_novel) else 0
        
        if (epoch < args.warmup):
            #warm-up phase
            final_loss = loss_ce_sup + loss_con_unsup + loss_pu + 5 * loss_reg 
        else:
            #training phase
            final_loss = loss_ce_sup + loss_con_unsup + loss_pu + loss_reg + (1-w) * loss_proto  + w * (loss_ce_unsup)  + args.knn_weight * w * loss_con_knn
        
       

        losses.update(final_loss.item(), inputs_l_w.size(0))
        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()
        ema_optimizer.step()
        scheduler.step()
        batch_time.update(time.time() - end)
        end = time.time()

        if not args.no_progress and batch_idx%1000==1:
            p_bar.set_description("train epoch: {epoch}/{epochs:4}. itr: {batch:4}/{iter:4}. btime: {bt:.3f}s. loss: {loss:.4f}.".format(
                epoch=epoch + 1,
                epochs=args.epochs,
                batch=batch_idx + 1,
                iter=args.iteration,
                bt=batch_time.avg,
                loss=losses.avg
                ))
            p_bar.update()

    
    
    
    #Class-wise sample selection with high-confidence
    print("rho",rho)
    max_score_all,target_list_all  = torch.max(train_stat['pseudo_list_all'], dim=-1)
    prob = np.zeros(target_list_all.shape[0])
    idx_chosen_known = []
    idx_chosen_novel = []
    for j in range(args.no_known):
        index_j =  np.where(target_list_all.numpy()==j)[0]
        max_score_j = max_score_all[index_j]
        sort_index_j = (-max_score_j).sort()[1].cpu().numpy()
        partition_j = int(len(index_j)*rho)
        if len(index_j) == 0:
            continue
        idx_chosen_known.append(index_j[sort_index_j[:partition_j]])

    for j in range(args.no_known,args.no_class):
        index_j =  np.where(target_list_all.numpy()==j)[0]
        max_score_j = max_score_all[index_j]
        sort_index_j = (-max_score_j).sort()[1].cpu().numpy()
        partition_j = int(len(index_j)*rho)
        if(len(index_j)):
            #prototype calculation
            features_j = train_stat['feature_all'][index_j]
            prototype_j = features_j.mean(0)
            all_prototype[j] = prototype_j
        else:
            continue
        idx_chosen_novel.append(index_j[sort_index_j[:partition_j]])
    all_prototype = F.normalize(all_prototype, dim=1)
    train_stat['all_prototype'] = all_prototype.detach().clone()
    if(len(idx_chosen_known)):
        idx_chosen_known = np.concatenate(idx_chosen_known)
    if(len(idx_chosen_novel)):
        idx_chosen_novel = np.concatenate(idx_chosen_novel)
    prob[idx_chosen_known] = 1
    prob[idx_chosen_novel] = 1
    train_stat['prob'] = prob
    if not args.no_progress:
        p_bar.close()
    train_stat['target_pu_max'] = target_pu_max
    return train_stat


def test_known(args, test_loader, model, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    model.eval()

    if not args.no_progress:
        test_loader = tqdm(test_loader)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.cuda()
            targets = targets.cuda()
            _, outputs,_,_,_ = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress and batch_idx%1000==1:
                test_loader.set_description("test cluster epoch: {epoch}/{epochs:4}. itr: {batch:4}/{iter:4}. btime: {bt:.3f}s.".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    bt=batch_time.avg,
                ))
        if not args.no_progress:
            test_loader.close()
    return top1.avg


def test_cluster(args, test_loader, model, epoch, offset=0):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    gt_targets =[]
    predictions = []
    model.eval()
    if not args.no_progress:
        test_loader = tqdm(test_loader)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            inputs = inputs.cuda()
            targets = targets.cuda()
            _, outputs,_,_,output_pu = model(inputs)
            _, max_idx = torch.max(outputs, dim=1)
            predictions.extend(max_idx.cpu().numpy().tolist())
            gt_targets.extend(targets.cpu().numpy().tolist())
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress and batch_idx%1000==1:
                test_loader.set_description("test cluster epoch: {epoch}/{epochs:4}. itr: {batch:4}/{iter:4}. btime: {bt:.3f}s.".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    bt=batch_time.avg,
                ))
        if not args.no_progress:
            test_loader.close()

    predictions = np.array(predictions)
    gt_targets = np.array(gt_targets)

    predictions = torch.from_numpy(predictions)
    gt_targets = torch.from_numpy(gt_targets)
    eval_output = hungarian_evaluate(predictions, gt_targets, offset)
    return eval_output


if __name__ == '__main__':
    cudnn.benchmark = True
    main()
