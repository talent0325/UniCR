import os
import time
import random
import json
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR

import numpy as np
from configs.opts import parser
from model.cs_model import CS_Model as main_model
from utils import AverageMeter, Prepare_logger, get_and_save_args
from utils.Recorder import Recorder
from utils.helper import *

import torch.nn.functional as F
# from helper import *
# =================================  seed config ============================
SEED = 43
# SEED = args.seed
random.seed(SEED)
np.random.seed(seed=SEED)
torch.manual_seed(seed=SEED)
torch.cuda.manual_seed(seed=SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

config_path = './configs/main.json'
with open(config_path) as fp:
    config = json.load(fp)
print(config)
# =============================================================================

def AVPSLoss(av_simm, soft_label):
    """audio-visual pair similarity loss for fully supervised setting,
    please refer to Eq.(8, 9) in our paper.
    """
    # av_simm: [bs, 10]
    relu_av_simm = F.relu(av_simm)
    sum_av_simm = torch.sum(relu_av_simm, dim=-1, keepdim=True)
    avg_av_simm = relu_av_simm / (sum_av_simm + 1e-8)
    loss = nn.MSELoss()(avg_av_simm, soft_label)
    return loss
 

def main():
    # utils variable
    global args, logger, writer, dataset_configs
    # statistics variable
    global best_accuracy, best_accuracy_epoch
    best_accuracy, best_accuracy_epoch = 0, 0
    # configs
    dataset_configs = get_and_save_args(parser)
    parser.set_defaults(**dataset_configs)
    args = parser.parse_args()
    args.seed = SEED
    import datetime
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if args.time == '':
        args.time = now
    # select GPUs
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    '''Create snapshot_pred dir for copying code and saving models '''
    if not args.evaluate:
        if not os.path.exists(args.snapshot_pref):
            os.makedirs(args.snapshot_pref, exist_ok=True)

        if not os.path.exists(os.path.join(args.snapshot_pref, args.time)):
            os.makedirs(os.path.join(args.snapshot_pref, args.time), exist_ok=True)

        if not os.path.exists(args.logs_dir):
            os.makedirs(args.logs_dir, exist_ok=True)
    if os.path.isfile(args.resume):
        args.snapshot_pref = os.path.dirname(args.resume)

    logger = Prepare_logger(args, eval=args.evaluate)
    logger.info(args.postscript)
    logger.info(config)
    if not args.evaluate:
        logger.info(f'\nCreating folder: {os.path.join(args.snapshot_pref, args.time)}')
        logger.info('\nRuntime args\n\n{}\n'.format(json.dumps(vars(args), indent=4)))
    else:
        logger.info(f'\nLog file will be save in a {args.logs_dir}/Eval-{args.resume}.log.')

    '''Dataset'''
    from dataset.AVE_dataset import AVEDataset as AVEDataset
    # from dataset.AVE_dataset import BGMDataset as AVEDataset
    is_select = args.is_select
    audio_process_mode = args.audio_preprocess_mode
    data_root = args.data_root
    meta_root = args.meta_root
    ave = args.ave
    avepm = args.avepm
    if ave == False and avepm == False:
        raise ValueError("Please choose one of the two datasets: AVE or AVEPM")
    if ave == True and avepm == True:
        raise ValueError("Please choose only one of the two datasets: AVE or AVEPM")
    preprocess_mode = args.preprocess
    v_feature_root = args.v_feature_root
    a_feature_root = args.a_feature_root

    # MARK: Dataset
    train_dataloader = DataLoader(
        AVEDataset(
            split='train',
            data_root=data_root,
            meta_root=meta_root,
            ave=ave,
            avepm=avepm,
            preprocess_mode=preprocess_mode,
            audio_process_mode=audio_process_mode,
            is_select=is_select,
            a_feature_root=a_feature_root,
            v_feature_root=v_feature_root,
            bgm=args.bgm),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        # pin_memory=True
    )

    val_dataloader = DataLoader(
        AVEDataset(
            split='val',
            data_root=data_root,
            meta_root=meta_root,
            ave=ave,
            avepm=avepm,
            preprocess_mode=preprocess_mode,
            audio_process_mode=audio_process_mode,
            is_select=is_select,
            a_feature_root=a_feature_root,
            v_feature_root=v_feature_root,
            bgm=args.bgm),
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=0,
        # pin_memory=True
    )

    test_dataloader = DataLoader(
        AVEDataset(
            split='test',
            data_root=data_root,
            meta_root=meta_root,
            ave=ave,
            avepm=avepm,
            preprocess_mode=preprocess_mode,
            audio_process_mode=audio_process_mode,
            is_select=is_select,
            a_feature_root=a_feature_root,
            v_feature_root=v_feature_root,
            bgm=args.bgm),
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=0,
        # pin_memory=True
    )

    '''model setting'''
    if is_select:
        config['model']['category_num'] = 10
    else:
        if ave:
            config['model']['category_num'] = 28
        elif avepm:
            config['model']['category_num'] = 86
    config['model']['guide'] = args.guide
    mainModel = main_model(config['model'], args)
    mainModel = nn.DataParallel(mainModel).cuda()
    mainModel = mainModel.float()
    # MARK: param
    param_group = []
    param_group_cln = []
    for name, param in mainModel.named_parameters():  
        if 'cln' in name:
            param_group_cln.append({"params": param, "lr":args.lr_cln})
        else:
            param_group.append({"params": param, "lr":args.lr})

    logger.info(f"Number of CLN params: {sum(len(group['params']) for group in param_group_cln)}")
    logger.info(f"Number of Main params: {sum(len(group['params']) for group in param_group)}")

    optimizer = torch.optim.Adam(param_group)
    optimizer_cln = torch.optim.Adam(param_group_cln)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.5)
    criterion = nn.BCEWithLogitsLoss().cuda()
    criterion_event = nn.CrossEntropyLoss().cuda()

    '''Resume from a checkpoint'''
    if os.path.isfile(args.resume):
        logger.info(f"\nLoading Checkpoint: {args.resume}\n")
        mainModel.load_state_dict(torch.load(args.resume))
    elif args.resume != "" and (not os.path.isfile(args.resume)):
        raise FileNotFoundError

    '''Only Evaluate'''
    if args.evaluate:
        logger.info(f"\nStart Evaluation..")
        validate_epoch(mainModel, test_dataloader, criterion, criterion_event, epoch=0, eval_only=True)
        return
    
    '''Training and Testing'''
    for epoch in range(args.n_epoch):
        if args.use_cln and args.use_saliency:
            loss = train_epoch_cln_saliency(mainModel, train_dataloader, criterion, criterion_event, optimizer, optimizer_cln, epoch)
        elif args.use_cln and not args.use_saliency:
            loss = train_epoch_cln(mainModel, train_dataloader, criterion, criterion_event, optimizer, optimizer_cln, epoch)
        elif not args.use_cln and args.use_saliency:
            loss = train_epoch_saliency(mainModel, train_dataloader, criterion, criterion_event, optimizer, optimizer_cln, epoch)
        else:
            loss = train_epoch(mainModel, train_dataloader, criterion, criterion_event, optimizer, optimizer_cln, epoch)

        if ((epoch + 1) % args.eval_freq == 0) or (epoch == args.n_epoch - 1):
            acc = validate_epoch(mainModel, val_dataloader, criterion, criterion_event, epoch)
            if acc > best_accuracy:
                best_accuracy = acc
                best_accuracy_epoch = epoch
                save_checkpoint(
                    mainModel.state_dict(),
                    top1=best_accuracy,
                    task='Supervised',
                    epoch=epoch + 1,
                )
            logger.info("-----------------------------")
            logger.info(f"best acc: {best_accuracy} at Epoch-{best_accuracy_epoch}")
            logger.info("-----------------------------")
        scheduler.step()


def train_epoch_cln_saliency(model, train_dataloader, criterion, criterion_event, optimizer, optimizer_cln, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    train_acc = AverageMeter()
    end_time = time.time()
    if epoch == 0:
        logger.info("====== Train using cln and saliency module ======")
    model.train()
    optimizer.zero_grad()
    optimizer_cln.zero_grad()
    if epoch == args.warm_up + 5:
        logger.info('================================== Saliency Module required_grad set to False =========')
        for name, param in model.named_parameters():
            if 'saliency' in name:
                param.requires_grad = False
        optimizer_params = [p for n, p in model.named_parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(optimizer_params, lr=args.lr)

    for n_iter, batch_data in enumerate(train_dataloader):

        data_time.update(time.time() - end_time)
        '''Feed input to model'''
        visual_feature, audio_feature, labels = batch_data
        # labels = labels.double().cuda()
        labels = labels.float().cuda()
        is_event_score_a, p_a, is_event_score_v, p_v, is_event_score_av, p_av, c_a, c_v, c_av, sal_map = model(audio_feature, visual_feature)
        
        if epoch < args.warm_up:
            c_a_hat = 1 / 3
            c_v_hat = 1 / 3
            c_av_hat = 1 / 3
        else:
            c_a_hat, c_v_hat, c_av_hat = normalize_score_v2([c_a, c_v, c_av], lambda_param=1)
        
        event_scores = p_a * c_a_hat + p_v * c_v_hat + p_av * c_av_hat
        is_event_scores = is_event_score_a * c_a_hat + is_event_score_v * c_v_hat + is_event_score_av * c_av_hat
        is_event_scores = is_event_scores.squeeze()  # 帧级别
        labels_foreground = labels  # [32, 10, 28]
        labels_BCE, labels_evn = labels_foreground.max(-1)
        labels_event, _ = labels_evn.max(-1)
        loss_is_event = criterion(is_event_scores, labels_BCE.cuda())
        event_scores_clip, _ = event_scores.max(1)
        loss_event_class = criterion_event(event_scores_clip, labels_event.cuda())
        loss = loss_is_event + loss_event_class

        """compute loss and backward"""
        l_a = criterion(p_a, labels)
        l_v = criterion(p_v, labels)
        l_av = criterion(p_av, labels)
        l_event_scores = criterion(event_scores, labels)


        # # Saliency Loss
        sal_map = sal_map.squeeze()
        lambda_sparse = 0.0001
        lambda_tv = 0.01
        lambda_temporal = 0.001
        lambda_entropy = 0.001

        loss_sparse = sal_map.mean()
        loss_tv = total_variation(sal_map)
        loss_temporal = F.l1_loss(sal_map[:, 1:], sal_map[:, :-1]) # 鼓励相邻时间帧的 saliency map 保持连续（自监督）
        entropy = - (sal_map * (sal_map + 1e-6).log()).mean() # 鼓励显著性图分布不是平均的，即使稀疏，也要明确突出某些区域
        loss_entropy_weighted = (is_event_scores.detach() * entropy).mean()

        # loss_saliency = lambda_sparse * loss_sparse + lambda_tv * loss_tv + lambda_temporal * loss_temporal + lambda_entropy * entropy + 0.01 * loss_entropy_weighted
        loss_saliency = lambda_sparse * loss_sparse + lambda_tv * loss_tv + lambda_temporal * loss_temporal + lambda_entropy * entropy 
        lambda_saliency = 1

        loss_total = loss + l_a + l_v + l_av + loss_saliency * lambda_saliency + l_event_scores

        loss_total.backward(retain_graph=True)

        if epoch > args.warm_up:
            # loss_cln, _, _ = MMLoss_fn([p_a, p_v, p_av], [c_a, c_v, c_av], labels)
            loss_cln, _, _, _ = MMLoss_fn_v2(
                [p_a, p_v, p_av], 
                [c_a, c_v, c_av], 
                labels,
                [is_event_score_a, is_event_score_v, is_event_score_av],
                labels_BCE
            )
            loss_cln.backward()
            optimizer_cln.step()
            optimizer_cln.zero_grad()  # 清空梯度，避免影响下次计算
        optimizer.step()
        optimizer.zero_grad()


        '''Clip Gradient'''
        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)

        losses.update(loss.item(), visual_feature.size(0) * 10)
        batch_time.update(time.time() - end_time)
        end_time = time.time()


        '''Print logs in Terminal'''
        if n_iter % args.print_freq == 0:
            if epoch > args.warm_up:
                logger.info(
                    f'Train Epoch: [{epoch}][{n_iter}/{len(train_dataloader)}]\t'
                    f'Loss_total {losses.val:.4f} ({losses.avg:.4f})\t'
                    f'Loss_cln {loss_cln.item():.4f}\t'
                )
            else:
                logger.info(
                    f'Train Epoch: [{epoch}][{n_iter}/{len(train_dataloader)}]\t'
                    f'Loss_total {losses.val:.4f} ({losses.avg:.4f})\t'
                )
    return losses.avg



def train_epoch_cln(model, train_dataloader, criterion, criterion_event, optimizer, optimizer_cln, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    train_acc = AverageMeter()
    end_time = time.time()

    model.train()
    optimizer.zero_grad()
    optimizer_cln.zero_grad()
    if epoch == 0:
        logger.info("====== Train only using cln ======")

    for n_iter, batch_data in enumerate(train_dataloader):
        data_time.update(time.time() - end_time)
        '''Feed input to model'''
        visual_feature, audio_feature, labels = batch_data
        labels = labels.float().cuda()
        is_event_score_a, p_a, is_event_score_v, p_v, is_event_score_av, p_av, c_a, c_v, c_av = model(audio_feature, visual_feature)
        
        if epoch < args.warm_up:
            c_a_hat = 1 / 3
            c_v_hat = 1 / 3
            c_av_hat = 1 / 3
        else:
            c_a_hat, c_v_hat, c_av_hat = normalize_score_v2([c_a, c_v, c_av], lambda_param=1)
            
        event_scores = p_a * c_a_hat + p_v * c_v_hat + p_av * c_av_hat

        is_event_scores = is_event_score_a * c_a_hat + is_event_score_v * c_v_hat + is_event_score_av * c_av_hat
        is_event_scores = is_event_scores.squeeze()  # 帧级别

        labels_foreground = labels  # [32, 10, 28]
        labels_BCE, labels_evn = labels_foreground.max(-1)
        labels_event, _ = labels_evn.max(-1)
        loss_is_event = criterion(is_event_scores, labels_BCE.cuda())
        event_scores_clip, _ = event_scores.max(1)
        loss_event_class = criterion_event(event_scores_clip, labels_event.cuda())
        loss = loss_is_event + loss_event_class

        """compute loss and backward"""
        l_a = criterion(p_a, labels)
        l_v = criterion(p_v, labels)
        l_av = criterion(p_av, labels)
        l_event_scores = criterion(event_scores, labels)

        loss_total = loss + l_a + l_v + l_av + l_event_scores
        # loss_total = loss
        loss_total.backward(retain_graph=True)

        if epoch > args.warm_up:
            loss_cln, _, _ = MMLoss_fn([p_a, p_v, p_av], [c_a, c_v, c_av], labels)
            loss_cln.backward()
            optimizer_cln.step()
            optimizer_cln.zero_grad()  # 清空梯度，避免影响下次计算
        optimizer.step()
        optimizer.zero_grad()


        '''Clip Gradient'''
        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)

        losses.update(loss.item(), visual_feature.size(0) * 10)
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        '''Print logs in Terminal'''
        if n_iter % args.print_freq == 0:
            if epoch > args.warm_up:
                logger.info(
                    f'Train Epoch: [{epoch}][{n_iter}/{len(train_dataloader)}]\t'
                    f'Loss_total {losses.val:.4f} ({losses.avg:.4f})\t'
                    f'Loss_cln {loss_cln.item():.4f}\t'
                )
            else:
                logger.info(
                    f'Train Epoch: [{epoch}][{n_iter}/{len(train_dataloader)}]\t'
                    f'Loss_total {losses.val:.4f} ({losses.avg:.4f})\t'
                )
    return losses.avg

def train_epoch_saliency(model, train_dataloader, criterion, criterion_event,optimizer, optimizer_cln, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    train_acc = AverageMeter()
    end_time = time.time()

    model.train()
    optimizer.zero_grad()
    optimizer_cln.zero_grad()
    if epoch == 0:
        logger.info("====== Train only using saliency module ======")

    # MARK: freeze saliency module
    if epoch == args.warm_up + 2:
        logger.info('================================== Saliency Module required_grad set to False =========')
        for name, param in model.named_parameters():
            if 'saliency' in name:
                param.requires_grad = False
        optimizer_params = [p for n, p in model.named_parameters() if'saliency' not in n and p.requires_grad]
        optimizer = torch.optim.Adam(optimizer_params, lr=args.lr)
    
    for n_iter, batch_data in enumerate(train_dataloader):
        data_time.update(time.time() - end_time)
        '''Feed input to model'''
        visual_feature, audio_feature, labels = batch_data
        # labels = labels.double().cuda()
        labels = labels.float().cuda()
        is_event_score_a, p_a, is_event_score_v, p_v, is_event_score_av, p_av, c_a, c_v, c_av, sal_map = model(audio_feature, visual_feature)
        
        event_scores = p_av
        is_event_scores = is_event_score_av
        is_event_scores = is_event_scores.squeeze()  # 帧级别
        labels_foreground = labels  # [32, 10, 28]
        labels_BCE, labels_evn = labels_foreground.max(-1)
        labels_event, _ = labels_evn.max(-1)
        loss_is_event = criterion(is_event_scores, labels_BCE.cuda())
        event_scores_clip, _ = event_scores.max(1)
        loss_event_class = criterion_event(event_scores_clip, labels_event.cuda())
        loss = loss_is_event + loss_event_class

        l_event_scores = criterion(event_scores, labels)

        # Saliency Loss
        sal_map = sal_map.squeeze()
        lambda_sparse = 0.001
        lambda_tv = 0.01
        lambda_temporal = 0.05
        lambda_entropy = 0.001

        loss_sparse = sal_map.mean()
        loss_tv = total_variation(sal_map)
        loss_temporal = F.l1_loss(sal_map[:, 1:], sal_map[:, :-1]) # 鼓励相邻时间帧的 saliency map 保持连续（自监督）
        entropy = - (sal_map * (sal_map + 1e-6).log()).mean() # 鼓励显著性图分布不是平均的，即使稀疏，也要明确突出某些区域

        loss_saliency = lambda_sparse * loss_sparse + lambda_tv * loss_tv + lambda_temporal * loss_temporal + lambda_entropy * entropy
        lambda_saliency = 1

        """compute loss and backward"""
        loss_total = loss + loss_saliency * lambda_saliency + l_event_scores
        loss_total.backward()
        optimizer.step()
        optimizer.zero_grad()

        '''Clip Gradient'''
        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)

        losses.update(loss.item(), visual_feature.size(0) * 10)
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        '''Print logs in Terminal'''
        if n_iter % args.print_freq == 0:
            if epoch > args.warm_up:
                logger.info(
                    f'Train Epoch: [{epoch}][{n_iter}/{len(train_dataloader)}]\t'
                    f'Loss_total {losses.val:.4f} ({losses.avg:.4f})\t'
                )
            else:
                logger.info(
                    f'Train Epoch: [{epoch}][{n_iter}/{len(train_dataloader)}]\t'
                    f'Loss_total {losses.val:.4f} ({losses.avg:.4f})\t'
                )
    return losses.avg

def train_epoch(model, train_dataloader, criterion, criterion_event, optimizer, optimizer_cln, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    train_acc = AverageMeter()
    end_time = time.time()

    model.train()
    optimizer.zero_grad()
    optimizer_cln.zero_grad()
    if epoch == 0:
        logger.info("====== Train without cln and saliency module ======")

    for n_iter, batch_data in enumerate(train_dataloader):

        data_time.update(time.time() - end_time)
        '''Feed input to model'''
        visual_feature, audio_feature, labels = batch_data
        # labels = labels.double().cuda()
        labels = labels.float().cuda()
        is_event_score_a, p_a, is_event_score_v, p_v, is_event_score_av, p_av, c_a, c_v, c_av = model(audio_feature, visual_feature)

        event_scores = p_av
        is_event_scores = is_event_score_av
        is_event_scores = is_event_scores.squeeze()  # 帧级别
        labels_foreground = labels  # [32, 10, 28]
        labels_BCE, labels_evn = labels_foreground.max(-1)
        labels_event, _ = labels_evn.max(-1)
        loss_is_event = criterion(is_event_scores, labels_BCE.cuda())
        event_scores_clip, _ = event_scores.max(1)
        loss_event_class = criterion_event(event_scores_clip, labels_event.cuda())
        loss = loss_is_event + loss_event_class

        """compute loss and backward"""
        l_event_scores = criterion(event_scores, labels)
        loss_total = loss + l_event_scores
        loss_total.backward()
        optimizer.step()
        optimizer.zero_grad()

        '''Clip Gradient'''
        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)

        losses.update(loss.item(), visual_feature.size(0) * 10)
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        '''Print logs in Terminal'''
        if n_iter % args.print_freq == 0:
            if epoch > args.warm_up:
                logger.info(
                    f'Train Epoch: [{epoch}][{n_iter}/{len(train_dataloader)}]\t'
                    f'Loss_total {losses.val:.4f} ({losses.avg:.4f})\t'
                )
            else:
                logger.info(
                    f'Train Epoch: [{epoch}][{n_iter}/{len(train_dataloader)}]\t'
                    f'Loss_total {losses.val:.4f} ({losses.avg:.4f})\t'
                )
    return losses.avg



@torch.no_grad()
def validate_epoch(model, test_dataloader, criterion, criterion_event, epoch, eval_only=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    end_time = time.time()

    model.eval()
    # model.double()

    total_acc = 0
    total_num = 0
    for n_iter, batch_data in enumerate(test_dataloader):
        data_time.update(time.time() - end_time)

        '''Feed input to model'''
        visual_feature, audio_feature, labels = batch_data
        # For a model in a float type
        visual_feature = visual_feature.float()
        audio_feature = audio_feature.float()
        labels = labels.float().cuda()

        bs = visual_feature.size(0)
        with torch.no_grad():
            if args.use_saliency:
                is_event_score_a, p_a, is_event_score_v, p_v, is_event_score_av, p_av, c_a, c_v, c_av, _ = model(audio_feature, visual_feature)
            else:
                is_event_score_a, p_a, is_event_score_v, p_v, is_event_score_av, p_av, c_a, c_v, c_av = model(audio_feature, visual_feature)
            
            if args.use_cln:
                c_a_hat, c_v_hat, c_av_hat = normalize_score_v2([c_a, c_v, c_av], lambda_param=1)
                c_a_neg_hat, c_neg_v_hat, c_av_neg_hat = normalize_score_v2([1-c_a_hat, 1-c_v_hat, 1-c_av_hat], lambda_param=1)
                is_event_scores = is_event_score_a * c_a_hat + is_event_score_v * c_v_hat + is_event_score_av * c_av_hat
                is_event_scores = is_event_scores.squeeze()  # 帧级别
                event_scores = p_a * c_a_hat + p_v * c_v_hat + p_av * c_av_hat
            else:
                is_event_scores = is_event_score_av
                is_event_scores = is_event_scores.squeeze()  # 帧级别
                event_scores = p_av

        acc = compute_accuracy_supervised(is_event_scores, event_scores, labels, bg_class=args.category_num)
        accuracy.update(acc.item(), bs * 10)
    logger.info(f"\tEvaluation results (acc): {accuracy.avg:.4f}%.")
    return accuracy.avg


def compute_accuracy_supervised(is_event_scores, event_scores, labels, bg_class=28):
    """
    is_event_scores: [B, T]，每一帧是否是事件的预测（sigmoid 输出）
    event_scores: [B, T, C]，每一帧的分类 logits，C=28
    labels: [B, T, C]，one-hot 标签（无背景类别）
    bg_class: 背景类编号（默认为 28）

    返回：
        acc: 分类准确率（前景分类 + 背景判断）
    """
    B, T, C = labels.shape
    full_C = C + 1  # 添加背景类

    # Step 1: 构建新标签，扩展一个背景维度
    labels_with_bg = torch.zeros(B, T, full_C, device=labels.device)  # [B, T, 29]
    labels_with_bg[:, :, :C] = labels  # 保留原来的 one-hot 标签

    # Step 2: is_event_scores → 判断是否前景
    is_event_scores = is_event_scores.sigmoid()         # [B, T]
    scores_pos_ind = is_event_scores > 0.5              # [B, T], 前景 True
    labels_with_bg[:, :, bg_class] = (~scores_pos_ind).float()  # 背景帧 one-hot 第 28 位为 1

    # Step 3: 提取标签索引
    _, targets = labels_with_bg.max(-1)  # [B, T]，每一帧的类别编号（含背景）

    # Step 4: event_scores 最大类别
    _, event_class = event_scores.max(-1)  # [B, T]
    pred = scores_pos_ind.long() * event_class  # 前景帧预测为 event_class，背景设为 0

    # Step 5: 背景帧设为 bg_class
    pred[~scores_pos_ind] = bg_class

    # Step 6: 计算准确率
    correct = pred.eq(targets)
    acc = correct.sum().float() * 100. / correct.numel()

    return acc

def save_checkpoint(state_dict, top1, task, epoch):
    model_name = f'{args.snapshot_pref}{args.time}/epoch_{epoch}_{top1:.3f}.pth.tar'
    torch.save(state_dict, model_name)

import torch.multiprocessing as mp
if __name__ == '__main__':
    main()
