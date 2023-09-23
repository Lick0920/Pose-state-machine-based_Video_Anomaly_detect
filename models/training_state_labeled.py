"""
Train\Test helper, based on awesome previous work by https://github.com/amirmk89/gepc
"""

import os
import time
import shutil
import torch
import torch.optim as optim
from tqdm import tqdm

def adjust_lr(optimizer, epoch, lr=None, lr_decay=None, scheduler=None):
    if scheduler is not None:
        scheduler.step()
        new_lr = scheduler.get_lr()[0]
    elif (lr is not None) and (lr_decay is not None):
        new_lr = lr * (lr_decay ** epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    else:
        raise ValueError('Missing parameters for LR adjustment')
    return new_lr


class Trainer:
    def __init__(self, args, model, train_loader, test_loader,input_frame=20,seg_len=24,
                 optimizer_f=None, scheduler_f=None,model_dis=None,linear=None):
        ##### 参数 
        self.input_frame =input_frame
        self.seg_len = seg_len
        self.model = model
        self.args = args
        self.train_loader = train_loader
        self.test_loader = test_loader
        #### 2 stage #####################
        pretrained_dic = torch.load("data\\exp_dir_state\\ShanghaiTech\\Aug09_2327_s4_20_4_5e5\\11_ep_Aug09_2329__checkpoint.pth.tar")
        self.model_unsupervised = model_dis
        self.model_unsupervised.load_state_dict(pretrained_dic['state_dict'], strict=False)

        #### 3 stage
        self.linear = linear
        self.losse = torch.nn.CrossEntropyLoss()

        # Loss, Optimizer and Scheduler
        self.lossf = torch.nn.MSELoss()
        self.lossr = torch.nn.MSELoss(reduce=False)
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        if optimizer_f is None:
            self.optimizer = self.get_optimizer()
        else:
            self.optimizer = optimizer_f(self.model.parameters())
        if scheduler_f is None:
            self.scheduler = None
        else:
            self.scheduler = scheduler_f(self.optimizer)

    def get_optimizer(self):
        if self.args.optimizer == 'adam':
            if self.args.lr:
                return optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            else:
                return optim.Adam(self.model.parameters())
        elif self.args.optimizer == 'adamx':
            if self.args.lr:
                return optim.Adamax(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            else:
                return optim.Adamax(self.model.parameters())
        return optim.SGD(self.model.parameters(), lr=self.args.lr)

    def adjust_lr(self, epoch):
        return adjust_lr(self.optimizer, epoch, self.args.model_lr, self.args.model_lr_decay, self.scheduler)

    def save_checkpoint(self, epoch, is_best=False, filename=None):
        """
        state: {'epoch': cur_epoch + 1, 'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict()}
        """
        state = self.gen_checkpoint_state(epoch)
        if filename is None:
            filename = 'checkpoint.pth.tar'

        state['args'] = self.args

        path_join = os.path.join(self.args.ckpt_dir, filename)
        torch.save(state, path_join)
        if is_best:
            shutil.copy(path_join, os.path.join(self.args.ckpt_dir, 'checkpoint_best.pth.tar'))

    def load_checkpoint(self, filename):
        filename = filename
        try:
            checkpoint = torch.load(filename)
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Checkpoint loaded successfully from '{}' at (epoch {})\n"
                  .format(filename, checkpoint['epoch']))
        except FileNotFoundError:
            print("No checkpoint exists from '{}'. Skipping...\n".format(self.args.ckpt_dir))

    def train(self, log_writer=None, clip=100):
        time_str = time.strftime("%b%d_%H%M_")
        checkpoint_filename = time_str + '_checkpoint.pth.tar'
        start_epoch = 0
        num_epochs = self.args.epochs
        self.model.train()
        self.model = self.model.to(self.args.device)
        key_break = False
        self.model_unsupervised.eval()
        self.model_unsupervised.to(self.args.device)
        self.linear.train()
        self.linear = self.linear.to(self.args.device)
        for epoch in range(start_epoch, num_epochs):
            if key_break:
                break
            print("Starting Epoch {} / {}".format(epoch + 1, num_epochs))
            pbar = tqdm(self.train_loader)
            for itern, data_arr in enumerate(pbar):
                try:
                    data = [data.to(self.args.device, non_blocking=True) for data in data_arr]
                    score = data[-3].amin(dim=-1)
                    label = data[-1] # 256 的 float 64 1 ---> 正常数据 都是1  ### 有label 了
                    if self.args.model_confidence:
                        samp = data[0]
                    else:
                        samp = data[0][:, :2] ## sample 256 2 24 18
                    # ## transposed ck0706 ####
                    # samp = samp.permute(0,2,1,3) # sample 256 24 18 2
                    # ######################################
                    samp = samp.permute(0,2,1,3).reshape(-1, self.seg_len, 36).float()

                    input =  samp[:,0:self.input_frame,:] # 预测
                    # input =  samp[:,self.seg_len-self.input_frame:self.seg_len,:] # 反预测
                    pred = self.model(input) # sample 256 2 24 18  --》256 144
                    if pred is None:
                        continue
                    if self.args.model_confidence: # 不同置信度
                        pred = pred * score
                    
                    p_truth = samp[:,self.input_frame:self.seg_len,:].reshape(-1, (self.seg_len-self.input_frame )* 36)
                    # p_truth = samp[:,0:self.seg_len-self.input_frame,:].reshape(-1, (self.seg_len-self.input_frame )* 36) # 反预测
                    
                    # #### 无监督的训练法：
                    # losses = self.lossf(pred, p_truth)
                    # losses.backward()
                    # orlu_dis = self.lossr(pred, p_truth).reshape(-1, (self.seg_len-self.input_frame), 36)
                    # orlu_dis = torch.mean(orlu_dis, dim=-1) # 256 x 4
                    label_4 = label[::,-4:] # 
                    ##### 调控负样本数量  正负样本1：1 ###
                    # #### distill ############### 用dis 改变label ##########################
                    # dis_pred = self.model_unsupervised(input)
                    # orlu_dis_dis = self.lossr(dis_pred, p_truth).reshape(-1, (self.seg_len-self.input_frame), 36)
                    # orlu_dis_dis = torch.mean(orlu_dis_dis, dim=-1) # (0到1 之间)
                    # print(torch.min(orlu_dis_dis), torch.max(orlu_dis_dis))
                    # label_4_dis = label[::,-4:].float()*100 + orlu_dis_dis # 0 + dis_anomaly
                    # label_4_dis = orlu_dis_dis    # 1
                    # ###### 比例可调  stage 4 比例可调
                    out_cls = self.linear(pred)
                    losses = self.losse(out_cls,label_4)
                    losses.backward()
                    # label_4_dis = label[::,-4:].float()*12 * cls +label[::,-4:].float()* 12 *orlu_dis_dis* cls + orlu_dis_dis

                    ############### 第一个系数 拉高正常和异常的整体差距， 第二个系数 拉高异常样本之间的差距
                    # label_4_dis = label[::,-4:].float()*10 +label[::,-4:].float() * 10 * orlu_dis_dis + orlu_dis_dis  # 2 目前最好 lr 5e5
                    # label_4_dis = label[::,-4:].float()*5 + label[::,-4:].float()* 10*orlu_dis_dis + orlu_dis_dis  # 2 目前最好 lr 5e5
                    # del orlu_dis_dis
                    # del dis_pred
                    # # del label_4
                    # #############################################################

                    # losses = self.lossf(orlu_dis, label_4_dis)
                    # losses.backward()
                    # # ##### stage 3  94.+  系数可学习  96.21 a=0.5
                    # out_cls = self.linear(pred)
                    # cls_loss = self.losse(out_cls,label_4)
                    # a = 0.9 # 0.001
                    # if losses > 1 or cls_loss> 1:
                    #     print(losses, cls_loss)
                    # total_loss = a*losses + (1-a)*cls_loss
                    # total_loss.backward()
                    # # #####
                    
                    
                    #############################################################
                    # losses = self.lossf(orlu_dis, label_4.float())
                    # losses.backward()
                    ###############################################
                
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    pbar.set_description("Loss: {}".format(losses.item()))
                    # log_writer.add_scalar('NLL Loss', losses.item(), epoch * len(self.train_loader) + itern)

                except KeyboardInterrupt:
                    print('Keyboard Interrupted. Save results? [yes/no]')
                    choice = input().lower()
                    if choice == "yes":
                        key_break = True
                        break
                    else:
                        exit(1)

            self.save_checkpoint(epoch, filename=str(epoch)+ "_ep_" + checkpoint_filename)
            new_lr = self.adjust_lr(epoch)
            print('Checkpoint Saved. New LR: {0:.3e}'.format(new_lr))

    def test(self):
        self.model.eval()
        self.model.to(self.args.device)
        self.model_unsupervised.eval()
        self.model_unsupervised.to(self.args.device)
        pbar = tqdm(self.test_loader)
        probs = torch.empty(0).to(self.args.device)
        print("Starting Test Eval")
        self.lossf = torch.nn.MSELoss(reduce=False)
        self.linear.eval()
        self.linear = self.linear.to(self.args.device)
        for itern, data_arr in enumerate(pbar):
            data = [data.to(self.args.device, non_blocking=True) for data in data_arr]
            score = data[-2].amin(dim=-1)
            label = data[-1] # 256 的 float 64 1 ---> 正常数据 都是1  ### 有label 了
            if self.args.model_confidence:
                samp = data[0]
            else:
                samp = data[0][:, :2]
            samp = samp.permute(0,2,1,3).reshape(-1, self.seg_len, 36).float()
            with torch.no_grad():
                input = samp[:,0:self.input_frame,:]
                # input =  samp[:,self.seg_len-self.input_frame:self.seg_len,:] # 反预测
                pred = self.model(input) # sample 256 2 24 18  --》256 144
                if pred is None:
                    continue
                if self.args.model_confidence: # 不同置信度
                    pred = pred * score
            
                p_truth = samp[:,self.input_frame:self.seg_len,:].reshape(-1, (self.seg_len-self.input_frame )* 36)
                # p_truth = samp[:,0:self.seg_len-self.input_frame,:].reshape(-1, (self.seg_len-self.input_frame )* 36) # 反预测

                orlu_dis = self.lossr(pred, p_truth).reshape(-1, (self.seg_len-self.input_frame), 36)
                orlu_dis = torch.mean(orlu_dis, dim=-1) # 256 x 4
                
                # #### distill ############### 用dis 改变label  ### test的时候不用加起来了
                # dis_pred = self.model_unsupervised(input)
                # orlu_dis_dis = self.lossr(dis_pred, p_truth).reshape(-1, (self.seg_len-self.input_frame), 36)
                # orlu_dis_dis = torch.mean(orlu_dis_dis, dim=-1) # (0到1 之间)
                # # label_4_dis = orlu_dis_dis   # 少一个label的1 分不开
                losses = orlu_dis
                
                # label_4_dis = label[::,-4:].float()*1 + orlu_dis_dis # 1 + dis_anomaly 
                # label_4_dis = orlu_dis_dis    # 0826 1  不用label 了/
                # ###### 比例可调  stage 4 比例可调
                
                # cls = self.losse(out_cls,label_4)
                # label_4_dis = label[::,-4:].float()*12 * cls +label[::,-4:].float()* 12 * orlu_dis_dis* cls + orlu_dis_dis

                # label_4_dis = label[::,-4:].float()*10 +label[::,-4:].float()* 10 *orlu_dis_dis + orlu_dis_dis   # 0826 10 2
                # ##### 0827 不对，预测不能用 label

                # losses = self.lossf(orlu_dis, label_4_dis)
                # total_loss = losses
                ###### 这样才对
                # total_loss = orlu_dis + orlu_dis_dis
                # total_loss =  self.lossf(pred,p_truth) # 试一下只无监督

                # ###############
                # #### stage 3
                # out_cls = self.linear(pred)
                # cls_loss = self.losse(out_cls,label_4)
                # a = 0.9
                # total_loss = a*losses + (1-a)*cls_loss
                #####
                # del orlu_dis_dis
                # del orlu_dis
                # del dis_pred
                # del losses
            # ###################################################
            # losses = self.lossf(orlu_dis, label_4.float())
            # losses = self.lossf(pred, p_truth)
            # losses = orlu_dis_dis  # 只用无监督 不训练

            probs = torch.cat((probs, -1 * losses.mean(dim=1)), dim=0)
        prob_mat_np = probs.cpu().detach().numpy().squeeze().copy(order='C')
        return prob_mat_np

    def gen_checkpoint_state(self, epoch):
        checkpoint_state = {'epoch': epoch + 1,
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(), }
        return checkpoint_state
