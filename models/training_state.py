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
                 optimizer_f=None, scheduler_f=None):
        ##### 参数 
        self.input_frame =input_frame
        self.seg_len = seg_len
        self.model = model
        self.args = args
        self.train_loader = train_loader
        self.test_loader = test_loader
        # Loss, Optimizer and Scheduler
        self.lossf = torch.nn.MSELoss()
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
        for epoch in range(start_epoch, num_epochs):
            if key_break:
                break
            print("Starting Epoch {} / {}".format(epoch + 1, num_epochs))
            pbar = tqdm(self.train_loader)
            for itern, data_arr in enumerate(pbar):
                try:
                    data = [data.to(self.args.device, non_blocking=True) for data in data_arr]
                    
                    samp = data[0]
                    samp = samp.permute(0,2,1,3).reshape(-1, self.seg_len, 72).float()

                    input =  samp[:,0:self.input_frame,:] # pred
                    # input =  samp[:,self.seg_len-self.input_frame:self.seg_len,:] # reverse pred
                    pred = self.model(input) # sample 256 2 24 18  ->256 144
                    if pred is None:
                        continue
                    # if self.args.model_confidence: # not confi
                    #     pred = pred * score
                    
                    p_truth = samp[:,self.input_frame:self.seg_len,:].reshape(-1, (self.seg_len-self.input_frame )* 72)
                    # p_truth = samp[:,0:self.seg_len-self.input_frame,:].reshape(-1, (self.seg_len-self.input_frame )* 36) #
                    losses = self.lossf(pred, p_truth)
                    losses.backward()
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

            self.save_checkpoint(epoch, filename=str(epoch)+ "_ep_" +checkpoint_filename)
            new_lr = self.adjust_lr(epoch)
            print('Checkpoint Saved. New LR: {0:.3e}'.format(new_lr))

    def test(self):
        self.model.eval()
        self.model.to(self.args.device)
        pbar = tqdm(self.test_loader)
        probs = torch.empty(0).to(self.args.device)
        print("Starting Test Eval")
        self.lossf = torch.nn.MSELoss(reduce=False)
        for itern, data_arr in enumerate(pbar):
            data = [data.to(self.args.device, non_blocking=True) for data in data_arr]
            samp = data[0]
            samp = samp.permute(0,2,1,3).reshape(-1, self.seg_len, 72).float()
           
            with torch.no_grad():
                input =  samp[:,0:self.input_frame,:] # pred
             
                pred = self.model(input) # sample 256 2 24 18  ->256 144
                if pred is None:
                    continue
            
            p_truth = samp[:,self.input_frame:self.seg_len,:].reshape(-1, (self.seg_len-self.input_frame )* 72)
            # p_truth = samp[:,0:self.seg_len-self.input_frame,:].reshape(-1, (self.seg_len-self.input_frame )* 36) # reverse pred

            losses = self.lossf(pred, p_truth)

            probs = torch.cat((probs, losses.mean(dim=1)), dim=0) # -1? 
        prob_mat_np = probs.cpu().detach().numpy().squeeze().copy(order='C')
        return prob_mat_np

    def gen_checkpoint_state(self, epoch):
        checkpoint_state = {'epoch': epoch + 1,
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(), }
        return checkpoint_state
