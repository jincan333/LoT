import logging
import traceback
import torch
import sys
import os
import time
from torch import nn
import torch.nn.functional as F
import wandb
import configparser
import argparse
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data import get_torch_dataset
from model.preresnet import PreResNet


def try_cuda(*wargs):
    cuda_args = []
    for arg in wargs:
        if hasattr(arg, 'cuda'):
            cuda_args.append(arg.cuda())
        else:
            print(f"{arg} does not have a .cuda() method.")
            cuda_args.append(arg)
    return tuple(cuda_args)


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


def evaluate(teacher, student, loader, epoch):
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


def train(teacher, student, loader, epoch, args, teacher_optimizer, student_optimizer, teacher_scheduler, student_scheduler):
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
            if args.loss=='kl_ce':
                teacher_loss=F.cross_entropy(teacher_pred, targets) + args.alpha*kl_div_logits(teacher_pred, student_pred.detach(), args.T)
                student_loss=F.cross_entropy(student_pred, targets) + args.alpha * kl_div_logits(student_pred, teacher_pred.detach(), args.T)
            elif args.loss=='kl':
                teacher_loss=F.cross_entropy(teacher_pred, targets) + args.alpha*kl_div_logits(teacher_pred, student_pred.detach(), args.T)
                student_loss=kl_div_logits(student_pred, teacher_pred.detach(), args.T)
            elif args.loss=='symmetric_kl':
                teacher_loss=F.cross_entropy(teacher_pred, targets) + args.alpha*(kl_div_logits(teacher_pred, student_pred.detach(), args.T)+kl_div_logits(student_pred.detach(), teacher_pred, args.T))
                student_loss=kl_div_logits(student_pred, teacher_pred.detach(), args.T) + kl_div_logits(teacher_pred.detach(), student_pred, args.T)
            elif args.loss=='symmetric_kl_ce':
                teacher_loss=F.cross_entropy(teacher_pred, targets) + args.alpha*(kl_div_logits(teacher_pred, student_pred.detach(), args.T)+kl_div_logits(student_pred.detach(), teacher_pred, args.T))
                student_loss=F.cross_entropy(student_pred, targets) + args.alpha * (kl_div_logits(student_pred, teacher_pred.detach(), args.T) + kl_div_logits(teacher_pred.detach(), student_pred, args.T))           
            teacher_loss.backward()
            student_loss.backward()
            teacher_optimizer.step()
            student_optimizer.step()
            teacher_correct+=teacher_pred.max(1)[1].eq(targets).sum().item()
            student_correct+=student_pred.max(1)[1].eq(targets).sum().item()
            loss += teacher_loss
            total += targets.size(0)
            #  student additional train
            for _ in range(args.student_steps_ratio - 1):
                s_inputs, s_targets = get_batch(loader, args.student_index)
                s_inputs, s_targets = s_inputs.cuda(), s_targets.cuda()
                args.student_index = (args.student_index+1) % len(loader)
                teacher_pred=F.log_softmax(teacher(s_inputs), dim=-1)
                student_pred=F.log_softmax(student(s_inputs), dim=-1)
                if args.loss=='kl_ce':
                    student_loss=F.cross_entropy(student_pred, s_targets) + args.alpha * kl_div_logits(student_pred, teacher_pred.detach(), args.T)
                elif args.loss=='kl':
                    student_loss=kl_div_logits(student_pred, teacher_pred.detach(), args.T)   
                elif args.loss=='symmetric_kl':
                    student_loss=kl_div_logits(student_pred, teacher_pred.detach(), args.T) + kl_div_logits(teacher_pred.detach(), student_pred, args.T)
                elif args.loss=='symmetric_kl_ce':
                    student_loss=F.cross_entropy(student_pred, s_targets) + args.alpha * (kl_div_logits(student_pred, teacher_pred.detach(), args.T) + kl_div_logits(teacher_pred.detach(), student_pred, args.T))
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


parser = argparse.ArgumentParser(description='PyTorch Image Classification')
parser.add_argument('--exp_name', type=str, default='LoT_ResNet')
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--models_num', type=int, default=2)
parser.add_argument('--detach', type=int, default=1)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=0,help='random seed')
parser.add_argument('--T', type=float, default=1.5)
parser.add_argument('--student_index', type=int, default=0, help='an independent index for student updating')
parser.add_argument('--student_steps_ratio', type=int, default=4)
parser.add_argument('--loss', type=str, default='kl_ce', choice=['kl', 'kl_ce', 'symmetric_kl', 'symmetric_kl_ce'])
# original
parser.add_argument('--dataset', type=str, default='cifar100', choices = ['cifar10', 'cifar100'])
parser.add_argument('--datadir', type=str, default='data', help='data directory')
parser.add_argument('--input_size', type=int, default=32, help='image input size')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_worker', type=int, default=4)
parser.add_argument('--depth_list', type=str, default='20_20', help='resnet model depth list', choice=['20_20', '20_56', '56_56', '56_20'])
parser.add_argument('--optimizer', type=str, default='sgd')
parser.add_argument('--lr', type=float, default=1.0)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--scheduler', type=str, default='cosine')
parser.add_argument('--epochs', type=int, default=180)
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str,  default='ckpt/LoT_ResNet'+randomhash+'CIFAR.pt', help='path to save the final model')
args = parser.parse_args()
print(json.dumps(vars(args), indent=4))


def main():
    try:
        # construct model, dataloaders
        config=configparser.ConfigParser()
        config.read('key.args')
        wandb_username=config.get('WANDB', 'USER_NAME')
        wandb_key=config.get('WANDB', 'API_KEY')
        wandb.login(key=wandb_key)
        wandb.init(project='LoT_ResNet', entity=wandb_username, name=args.exp_name)
        print(args.depth_list)
        depth_list=''.join(char for char in str(args.depth_list) if char.isdigit())
        depth_list=[int(depth_list[2*i:2*i+2]) for i in range(len(depth_list)//2)]
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(int(args.gpu))
        train_loader, test_loader = get_torch_dataset(args)

        # init teacher
        torch.manual_seed(args.seed)
        print('teacher depth:', depth_list[0])
        teacher=PreResNet(num_classes=args.num_classes, depth=depth_list[0], input_size=args.input_size)
        teacher=try_cuda(teacher)

        # init student
        torch.manual_seed(args.seed+1)
        print('student depth:', depth_list[1])
        student=PreResNet(num_classes=args.num_classes, depth=depth_list[0], input_size=args.input_size)
        student=try_cuda(student)
        args.student_index=0
        print(f"==== train and evaluate unequal restart ====")
        if args.optimizer=='sgd':
            teacher_optimizer = torch.optim.SGD(lr=args.lr, weight_decay=args.weight_decay, momemtum=0.9, nesterov=True, params=teacher.parameters())
            student_optimizer = torch.optim.SGD(lr=args.lr, weight_decay=args.weight_decay, momemtum=0.9, nesterov=True, params=student.parameters())
        if args.scheduler=='cosine':
            teacher_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(T_max=args.epochs, eta_min=0, optimizer=teacher_optimizer)
            student_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(T_max=args.epochs, eta_min=0, optimizer=teacher_optimizer)
        evaluate(teacher, student, test_loader, 0, args)
        for epoch in range(1, args.trainer.num_epochs+1):
            train(teacher, student, train_loader, epoch, args, teacher_optimizer, student_optimizer, teacher_scheduler, student_scheduler)
            evaluate(teacher, student, test_loader, epoch, args)
        torch.save(teacher.state_dict(), args.save+'_teacher.pt')
        torch.save(student.state_dict(), args.save+'_student.pt')
        print('ckpt location:', args.save)
        wandb.finish()

    except Exception:
        logging.error(traceback.format_exc())
        return float('NaN')


if __name__ == '__main__':
    main()
