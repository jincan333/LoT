import argparse
import time
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import json
import wandb
import os
import sys
import configparser
from itertools import islice


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_utils import get_lm_corpus
import model.rnn as rnn


parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM Language Modeling')
parser.add_argument('--exp_name', type=str, default='LoT_LSTM')
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--models_num', type=int, default=2)
parser.add_argument('--detach', type=int, default=1)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=0,help='random seed')
parser.add_argument('--T', type=float, default=1.5)
parser.add_argument('--momentum', type=float, default=0.3)
parser.add_argument('--lr_gamma', type=float, default=0.25, help='best:0.25, ')
parser.add_argument('--current_index', type=int, default=0, help='an independent index for student updating')
parser.add_argument('--student_steps_ratio', type=int, default=4)
# original
parser.add_argument('--data', type=str, default='ptb', choices = ['ptb', 'wt103'])
parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--emsize', type=int, default=650)
parser.add_argument('--nhid', type=int, default=650)
parser.add_argument('--nlayers', type=int, default=2)
parser.add_argument('--lr', type=float, default=40)
parser.add_argument('--clip', type=float, default=0.20)
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=30)
parser.add_argument('--batch_chunk', type=int, default=1, help='split batch into chunks to save memory')
parser.add_argument('--bptt', type=int, default=100)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--decreasing_step', type=list, default=[0.4, 0.65, 0.75, 0.83])
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str,  default='ckpt/LoT_LSTM'+randomhash+'WT103.pt',
                    help='path to save the final model')
parser.add_argument('--opt', type=str,  default='SGD',
                    help='SGD, Adam, RMSprop, Momentum')
args = parser.parse_args()

config=configparser.ConfigParser()
config.read('key.config')
wandb_username=config.get('WANDB', 'USER_NAME')
wandb_key=config.get('WANDB', 'API_KEY')

wandb.login(key=wandb_key)
wandb.init(project='LoT_LSTM_WT103', entity=wandb_username, name=args.exp_name)
torch.cuda.set_device(int(args.gpu))
device=torch.device(f'cuda:{args.gpu}')
def set_random_seed(s):
    np.random.seed(args.seed+s)
    torch.manual_seed(args.seed+s)
    torch.cuda.manual_seed(args.seed+s)


def kl_div_logits(p, q, T):
    loss_func = nn.KLDivLoss(reduction = 'batchmean', log_target=True)
    loss = loss_func(F.log_softmax(p/T, dim=-1), F.log_softmax(q/T, dim=-1)) * T * T
    return loss


set_random_seed(0)
eval_batch_size = 10
assert args.batch_size % args.batch_chunk == 0

if args.data=='ptb':
    datadir='data/ptb'
elif args.data=='wt103':
    datadir='data/wikitext-103'
corpus = get_lm_corpus(datadir, args.data)
ntokens = len(corpus.vocab)
train_data = corpus.get_iterator('train', args.batch_size, args.bptt, device=device, ext_len=0)
val_data = corpus.get_iterator('valid', eval_batch_size, args.bptt, device=device, ext_len=0)
test_data = corpus.get_iterator('test', eval_batch_size, args.bptt, device=device, ext_len=0)

args.eval_interval = math.ceil(train_data.data.size(0) / args.bptt)
args.max_step = args.epochs * args.eval_interval
args.eval_interval = 1000
print(json.dumps(vars(args), indent=4))

models = [0 for _ in range(args.models_num)]
total_params = 0
for k in range(args.models_num):
    set_random_seed(k)
    models[k] = rnn.RNNModel(ntokens, args.emsize, args.nhid, args.nlayers, args.dropout).to(device)
    print(models[k])
    total_params += sum(p.numel() for p in models[k].parameters())
    # print(f'Current parameters: {total_params}')
print(f"Total parameters: {total_params}")

criterion = nn.CrossEntropyLoss().cuda()
# Training code

def repackage_hidden(h):
    # detach
    return tuple(v.clone().detach() for v in h)


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    losses = [0 for _ in range(args.models_num)]
    for k in range(args.models_num):
        with torch.no_grad():
            models[k].eval()
            total_loss = 0
            hidden = models[k].init_hidden(eval_batch_size) #hidden size(nlayers, bsz, hdsize)
            for i in range(0, data_source.data.size(0) - 1, args.bptt):# iterate over every timestep
                data, targets, seq_len = data_source.get_batch(i)
                targets = targets.clone().detach().view(-1)
                output, hidden = models[k](data, hidden)
                total_loss += len(data) * criterion(output, targets).data
                hidden = repackage_hidden(hidden)
        losses[k] = total_loss / data_source.data.size(0)
    models[0].train()
    models[1].train()
    return losses


def train():
    global train_step
    # choose a optimizer
    for k in range(args.models_num):
        models[k].train()
    t_total_loss = 0
    s_total_loss = 0
    start_time = time.time()
    teacher_hiddenes = [models[k].init_hidden(args.batch_size) for k in range(args.models_num)]
    student_hiddenes = [models[k].init_hidden(args.batch_size) for k in range(args.models_num)]
    # train_data size(batchcnt, bsz)
    for batch, i in enumerate(range(0, train_data.data.size(0) - 1, args.bptt)):
        data, targets, seq_len = train_data.get_batch(i)
        targets = targets.clone().detach().view(-1)
        teacher_hiddenes = [repackage_hidden(teacher_hiddenes[l]) for l in range(args.models_num)]
        logits_list = [0 for _ in range(args.models_num)]
        # if args.alpha > 0:
        for k in range(args.models_num):
            logits_list[k], teacher_hiddenes[k] = models[k](data, teacher_hiddenes[k])
        if args.detach:
            teacher_loss=F.cross_entropy(logits_list[0], targets)+ \
                        args.alpha*(kl_div_logits(logits_list[0], logits_list[1].detach(), args.T))
            student_loss=F.cross_entropy(logits_list[1], targets) + args.alpha * kl_div_logits(logits_list[1], logits_list[0].detach(), args.T)
        else:
            teacher_loss=F.cross_entropy(logits_list[0], targets)+ \
                        args.alpha*(kl_div_logits(logits_list[0], logits_list[1], args.T))
            student_loss=F.cross_entropy(logits_list[1], targets) + args.alpha * kl_div_logits(logits_list[1], logits_list[0], args.T)
        opt.zero_grad()
        student_opt.zero_grad()
        teacher_loss.backward()
        student_loss.backward()
        torch.nn.utils.clip_grad_norm_(models[0].parameters(), args.clip)
        torch.nn.utils.clip_grad_norm_(models[1].parameters(), args.clip)
        opt.step()
        student_opt.step()
        t_total_loss += teacher_loss
        s_total_loss += student_loss
        train_step += 1

        # update student
        for _ in range(args.student_steps_ratio - 1):
            data, targets, seq_len = train_data.get_batch(args.current_index)
            targets = targets.clone().detach().view(-1)
            args.current_index = (args.bptt+args.current_index)%(train_data.data.size(0) - 1)
            student_hiddenes = [repackage_hidden(student_hiddenes[l]) for l in range(args.models_num)]
            for k in range(args.models_num):
                logits_list[k], student_hiddenes[k] = models[k](data, student_hiddenes[k])
            if args.detach:
                student_loss=F.cross_entropy(logits_list[1], targets) + args.alpha * kl_div_logits(logits_list[1], logits_list[0].detach(), args.T)
            else:
                student_loss=F.cross_entropy(logits_list[1], targets) + args.alpha * kl_div_logits(logits_list[1], logits_list[0], args.T)
            student_opt.zero_grad()
            student_loss.backward()
            torch.nn.utils.clip_grad_norm_(models[1].parameters(), args.clip)
            student_opt.step()
        scheduler.step(train_step)
        student_scheduler.step(train_step)

        if train_step % 200 == 0:
            t_cur_loss = t_total_loss / 200
            s_cur_loss = s_total_loss / 200
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | teacher lr {:02.2f} | student lr {:02.2f} | ms/batch {:5.2f} | '
                    'teacher loss {:5.2f} | student loss {:5.2f} | teacher ppl {:5.2f} | student ppl {:5.2f} |'.format(
                epoch, train_step, args.max_step, opt.param_groups[0]['lr'], student_opt.param_groups[0]['lr'],
                elapsed * 1000 / 200, t_cur_loss, s_cur_loss, math.exp(t_cur_loss), math.exp(s_cur_loss)))
            wandb.log({'teacher lr': opt.param_groups[0]['lr'], 'student lr': student_opt.param_groups[0]['lr'], 
            'teacher train ppl': math.exp(t_cur_loss), 'student train ppl': math.exp(s_cur_loss)}, step=train_step)
            t_total_loss = 0
            s_total_loss = 0
            start_time = time.time()

        if train_step % args.eval_interval == 0:
            val_losses = evaluate(val_data)
            print('-' * 89)
            print('| epoch {:3d} | step {:5d} | time: {:5.2f}s | teacher valid loss {:5.2f} | teacher valid loss {:5.2f} | ' 
                    'teacher valid ppl {:5.2f} | student valid ppl {:5.2f} |'.format(epoch, train_step, (time.time() - epoch_start_time),
                    val_losses[0], val_losses[1], math.exp(val_losses[0]), math.exp(val_losses[1])))
            print('-' * 89)
            wandb.log({'teacher valid ppl': math.exp(val_losses[0]), 'student valid ppl': math.exp(val_losses[1])}, step=train_step)
            for k in range(args.models_num):
                if not best_val_losses[k] or val_losses[k] < best_val_losses[k]:
                    with open(args.save+'_'+str(k), 'wb') as f:
                        torch.save(models[k], f)
                    best_val_losses[k] = val_losses[k]
        

# Loop over epochs.
best_val_losses = [None for _ in range(args.models_num)]

opt = torch.optim.SGD(models[0].parameters(), lr=args.lr, momentum=args.momentum)
student_opt= torch.optim.SGD(models[1].parameters(), lr=args.lr, momentum=args.momentum)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_step)
student_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(student_opt, T_max=args.max_step)
train_step = 0

try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model
for k in range(args.models_num):
    with open(args.save+'_'+str(k), 'rb') as f:
        models[k] = torch.load(f)

# Run on test data.
test_losses = evaluate(test_data)
print('=' * 89)
print('| End of training | teacher test loss {:5.2f} | teacher test ppl {:8.2f} | student test loss {:5.2f} | student test ppl {:8.2f} |'.format(
    test_losses[0], math.exp(test_losses[0]), test_losses[1], math.exp(test_losses[1])))
print('=' * 89)
wandb.log({'teacher test ppl': math.exp(test_losses[0]), 'student test ppl': math.exp(test_losses[1])}, step=train_step)
wandb.finish()
