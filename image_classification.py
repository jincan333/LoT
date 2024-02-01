import hydra
from upcycle.cuda import try_cuda
import logging
import traceback
import torch
import sys
import os
import time
from torch import nn
import torch.nn.functional as F
import wandb

current_script_directory = os.path.dirname(os.path.abspath(__file__))
gnosis_directory = os.path.abspath(os.path.join(current_script_directory, '..'))
sys.path.append(gnosis_directory)
from utils.data import get_loaders
from upcycle.scripting import startup
from hydra.utils import instantiate

def kl_div_logits(p, q, T):
    loss_func = nn.KLDivLoss(reduction = 'batchmean', log_target=True)
    loss = loss_func(F.log_softmax(p/T, dim=-1), F.log_softmax(q/T, dim=-1)) * T * T
    return loss


def get_batch(data_loader, batch_index):
    start_index = batch_index * data_loader.batch_size
    end_index = start_index + data_loader.batch_size
    batch_data = []
    batch_targets = []
    
    for i in range(start_index, end_index):
        if i >= len(data_loader.dataset):
            break
        data, target = data_loader.dataset[i]
        batch_data.append(data)
        batch_targets.append(target)
    
    return torch.stack(batch_data), torch.tensor(batch_targets)


def evaluate(teacher, student, loader, epoch, config):
    teacher.eval()
    student.eval()
    teacher_loss, student_loss = 0, 0
    teacher_correct, student_correct = 0, 0
    total = 0
    start = time.time()
    for batch in loader:
        with torch.no_grad():
            inputs, targets = try_cuda(*batch[:2])
            teacher_pred=F.log_softmax(teacher(inputs), dim=-1)
            student_pred=F.log_softmax(student(inputs), dim=-1)
            teacher_loss+=F.cross_entropy(teacher_pred, targets)
            student_loss+=F.cross_entropy(student_pred, targets)
            total += targets.size(0)
            teacher_correct+=teacher_pred.max(1)[1].eq(targets).sum().item()
            student_correct+=student_pred.max(1)[1].eq(targets).sum().item()
    end = time.time()
    step=epoch
    print('[eval] Epoch: %d | Teacher Test Loss: %.3f | Teacher Test Acc: %.3f | Student Test Loss: %.3f | Student Test Acc: %.3f | Time: %.3f |'
            % (step, teacher_loss / len(loader), 100. * teacher_correct / total, student_loss / len(loader), 100. * student_correct / total, end-start))
    wandb.log({'teacher test acc': 100. * teacher_correct / total, 'student test acc': 100. * student_correct / total}, step=step)


def train(teacher, student, loader, epoch, config, teacher_optimizer, student_optimizer, teacher_scheduler, student_scheduler):
    teacher.train()
    student.train()
    loss = 0
    student_correct, teacher_correct = 0, 0
    total = 0
    start = time.time()
    for _ in range(len(loader) // 10):
        for idx, (inputs, targets) in enumerate(loader):
            if idx > 10:
                break
            inputs, targets = try_cuda(inputs, targets)
            # update teacher and student
            teacher_optimizer.zero_grad()
            student_optimizer.zero_grad()
            teacher_pred=F.log_softmax(teacher(inputs), dim=-1)
            student_pred=F.log_softmax(student(inputs), dim=-1)
            if config.student_loss=='kl_ce':
                teacher_loss=F.cross_entropy(teacher_pred, targets) + config.loss.alpha*kl_div_logits(teacher_pred, student_pred.detach(), config.T)
                student_loss=F.cross_entropy(student_pred, targets) + config.student_alpha * kl_div_logits(student_pred, teacher_pred.detach(), config.T)
            elif config.student_loss=='kl':
                teacher_loss=F.cross_entropy(teacher_pred, targets) + config.loss.alpha*kl_div_logits(teacher_pred, student_pred.detach(), config.T)
                student_loss=kl_div_logits(student_pred, teacher_pred.detach(), config.T)
            elif config.student_loss=='symmetric_kl':
                teacher_loss=F.cross_entropy(teacher_pred, targets) + config.loss.alpha*(kl_div_logits(teacher_pred, student_pred.detach(), config.T)+kl_div_logits(student_pred.detach(), teacher_pred, config.T))
                student_loss=kl_div_logits(student_pred, teacher_pred.detach(), config.T) + kl_div_logits(teacher_pred.detach(), student_pred, config.T)
            elif config.student_loss=='symmetric_kl_ce':
                teacher_loss=F.cross_entropy(teacher_pred, targets) + config.loss.alpha*(kl_div_logits(teacher_pred, student_pred.detach(), config.T)+kl_div_logits(student_pred.detach(), teacher_pred, config.T))
                student_loss=F.cross_entropy(student_pred, targets) + config.student_alpha * (kl_div_logits(student_pred, teacher_pred.detach(), config.T) + kl_div_logits(teacher_pred.detach(), student_pred, config.T))           
            teacher_loss.backward()
            student_loss.backward()
            teacher_optimizer.step()
            student_optimizer.step()
            teacher_correct+=teacher_pred.max(1)[1].eq(targets).sum().item()
            student_correct+=student_pred.max(1)[1].eq(targets).sum().item()
            loss += teacher_loss
            total += targets.size(0)
            #  student additional train
            for _ in range(config.student_ratio):
                s_inputs, s_targets = get_batch(loader, config.student_index)
                s_inputs, s_targets = s_inputs.cuda(), s_targets.cuda()
                config.student_index = (config.student_index+1) % len(loader)
                teacher_pred=F.log_softmax(teacher(s_inputs), dim=-1)
                student_pred=F.log_softmax(student(s_inputs), dim=-1)
                if config.student_loss=='kl_ce':
                    student_loss=F.cross_entropy(student_pred, s_targets) + config.student_alpha * kl_div_logits(student_pred, teacher_pred.detach(), config.T)
                elif config.student_loss=='kl':
                    student_loss=kl_div_logits(student_pred, teacher_pred.detach(), config.T)   
                elif config.student_loss=='symmetric_kl':
                    student_loss=kl_div_logits(student_pred, teacher_pred.detach(), config.T) + kl_div_logits(teacher_pred.detach(), student_pred, config.T)
                elif config.student_loss=='symmetric_kl_ce':
                    student_loss=F.cross_entropy(student_pred, s_targets) + config.student_alpha * (kl_div_logits(student_pred, teacher_pred.detach(), config.T) + kl_div_logits(teacher_pred.detach(), student_pred, config.T))
                student_optimizer.zero_grad()
                student_loss.backward()
                student_optimizer.step()
    end = time.time()
    step=epoch
    print('[Train] Epoch: %d | Teacher lr=%.4f | Teacher Loss: %.3f | Teacher Train Acc: %.3f | Student lr=%.4f | Student Train Acc: %.3f | Time: %.3f |'
            % (step, teacher_scheduler.get_last_lr()[0], loss / len(loader), 100. * teacher_correct / total, student_scheduler.get_last_lr()[0], 100. * student_correct / total, end-start))
    wandb.log({'teacher lr': teacher_scheduler.get_last_lr()[0], 'teacher train acc': 100. * teacher_correct / total, 'student lr': student_scheduler.get_last_lr()[0], 'student train acc': 100. * student_correct / total}, step=step)
    teacher_scheduler.step()
    student_scheduler.step()


@hydra.main(config_path='../config', config_name='image_classification')
def main(config):
    try:
        # construct logger, model, dataloaders
        config, logger = startup(config)
        wandb.login(key='ca2f2a2ae6e84e31bbc09a8f35f9b9a534dfbe9b')
        wandb.init(project='ensemble_distill_unequal_steps_cifar', entity='jincan333', name=config.exp_name)
        print(config.depth_list)
        config.depth_list=''.join(char for char in str(config.depth_list) if char.isdigit())
        config.depth_list=[int(config.depth_list[2*i:2*i+2]) for i in range(len(config.depth_list)//2)]
        device = torch.device(f"cuda:{config.gpu}")
        torch.cuda.set_device(int(config.gpu))
        train_loader, test_loader, train_splits = get_loaders(config)

        # init teacher
        torch.manual_seed(config.seed)
        config.classifier.depth = config.depth_list[0]
        print('teacher depth:', config.classifier.depth)
        teacher=hydra.utils.instantiate(config.classifier)
        teacher=try_cuda(teacher)

        # init student
        torch.manual_seed(config.seed+1)
        config.classifier.depth = config.depth_list[1]
        print('student depth:', config.classifier.depth)
        student=hydra.utils.instantiate(config.classifier)
        student=try_cuda(student)
        config.student_index=0

        print(f"==== train and evaluate unequal restart ====")
        teacher_optimizer = instantiate(config.trainer.optimizer, params=teacher.parameters())
        if config.student_weight_decay >= 0:
            config.trainer.optimizer.weight_decay=config.student_weight_decay
        student_optimizer = instantiate(config.trainer.optimizer, params=student.parameters())
        teacher_scheduler = instantiate(config.trainer.lr_scheduler, optimizer=teacher_optimizer)
        student_scheduler = instantiate(config.trainer.lr_scheduler, optimizer=student_optimizer)
        evaluate(teacher, student, test_loader, 0, config)
        for epoch in range(1, config.trainer.num_epochs+1):
            train(teacher, student, train_loader, epoch, config, teacher_optimizer, student_optimizer, teacher_scheduler, student_scheduler)
            evaluate(teacher, student, test_loader, epoch, config)
        torch.save(teacher.state_dict(), 'teacher.pt')
        torch.save(student.state_dict(), 'student.pt')
        print('ckpt location:', os.path.join(config.data_dir, config.exp_name))
        wandb.finish()

    except Exception:
        logging.error(traceback.format_exc())
        return float('NaN')


if __name__ == '__main__':
    main()
