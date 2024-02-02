# coding: utf-8
import argparse
import time
import math
import os, sys
import itertools

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from mem_transformer import MemTransformerLM
from utils.data_utils import get_lm_corpus
from utils.exp_utils import create_exp_dir
from utils.data_parallel import BalancedDataParallel
import torch.nn.functional as F
import configparser
import wandb


parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')
parser.add_argument('--data', type=str, default='data/wikitext-103',
                    help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='wt103',
                    choices=['wt103', 'lm1b', 'enwik8', 'text8', 'ptb'],
                    help='dataset name')
parser.add_argument('--n_layer', type=int, default=12,
                    help='number of total layers')
parser.add_argument('--n_head', type=int, default=10,
                    help='number of heads')
parser.add_argument('--d_head', type=int, default=50,
                    help='head dimension')
parser.add_argument('--d_embed', type=int, default=-1,
                    help='embedding dimension')
parser.add_argument('--d_model', type=int, default=500,
                    help='model dimension')
parser.add_argument('--d_inner', type=int, default=1000,
                    help='inner dimension in FF')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='global dropout rate')
parser.add_argument('--dropatt', type=float, default=0.0,
                    help='attention probability dropout rate')
parser.add_argument('--init', default='normal', type=str,
                    help='parameter initializer to use.')
parser.add_argument('--emb_init', default='normal', type=str,
                    help='parameter initializer to use.')
parser.add_argument('--init_range', type=float, default=0.1,
                    help='parameters initialized by U(-init_range, init_range)')
parser.add_argument('--emb_init_range', type=float, default=0.01,
                    help='parameters initialized by U(-init_range, init_range)')
parser.add_argument('--init_std', type=float, default=0.02,
                    help='parameters initialized by N(0, init_std)')
parser.add_argument('--proj_init_std', type=float, default=0.01,
                    help='parameters initialized by N(0, init_std)')
parser.add_argument('--optim', default='adam', type=str,
                    choices=['adam', 'sgd', 'adagrad'],
                    help='optimizer to use.')
parser.add_argument('--lr', type=float, default=0.00025,
                    help='initial learning rate (0.00025|5 for adam|sgd)')
parser.add_argument('--mom', type=float, default=0.0,
                    help='momentum for sgd')
parser.add_argument('--scheduler', default='cosine', type=str,
                    choices=['cosine', 'inv_sqrt', 'dev_perf', 'constant'],
                    help='lr scheduler to use.')
parser.add_argument('--warmup_step', type=int, default=0,
                    help='upper epoch limit')
parser.add_argument('--decay_rate', type=float, default=0.5,
                    help='decay factor when ReduceLROnPlateau is used')
parser.add_argument('--lr_min', type=float, default=0.0,
                    help='minimum learning rate during annealing')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--clip_nonemb', action='store_true',
                    help='only clip the gradient of non-embedding params')
parser.add_argument('--max_step', type=int, default=100000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=60,
                    help='batch size')
parser.add_argument('--batch_chunk', type=int, default=1,
                    help='split batch into chunks to save memory')
parser.add_argument('--tgt_len', type=int, default=70,
                    help='number of tokens to predict')
parser.add_argument('--eval_tgt_len', type=int, default=50,
                    help='number of tokens to predict for evaluation')
parser.add_argument('--ext_len', type=int, default=0,
                    help='length of the extended context')
parser.add_argument('--mem_len', type=int, default=0,
                    help='length of the retained previous heads')
parser.add_argument('--not_tied', action='store_true',
                    help='do not tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adaptive', action='store_true',
                    help='use adaptive softmax')
parser.add_argument('--div_val', type=int, default=1,
                    help='divident value for adapative input and softmax')
parser.add_argument('--pre_lnorm', action='store_true',
                    help='apply LayerNorm to the input instead of the output')
parser.add_argument('--varlen', action='store_true',
                    help='use variable length')
parser.add_argument('--multi_gpu', action='store_true',
                    help='use multiple GPU')
parser.add_argument('--log-interval', type=int, default=200,
                    help='report interval')
parser.add_argument('--eval-interval', type=int, default=1000,
                    help='evaluation interval')
parser.add_argument('--work_dir', default='LM-TFM', type=str,
                    help='experiment directory.')
parser.add_argument('--restart', action='store_true',
                    help='restart training from the saved checkpoint')
parser.add_argument('--restart_dir', type=str, default='',
                    help='restart dir')
parser.add_argument('--debug', action='store_true',
                    help='run in debug mode (do not create exp dir)')
parser.add_argument('--same_length', action='store_true',
                    help='use the same attn length for all tokens')
parser.add_argument('--attn_type', type=int, default=0,
                    help='attention type. 0 for ours, 1 for Shaw et al,'
                    '2 for Vaswani et al, 3 for Al Rfou et al.')
parser.add_argument('--clamp_len', type=int, default=-1,
                    help='use the same pos embeddings after clamp_len')
parser.add_argument('--eta_min', type=float, default=0.0,
                    help='min learning rate for cosine scheduler')
parser.add_argument('--gpu0_bsz', type=int, default=-1,
                    help='batch size on gpu 0')
parser.add_argument('--max_eval_steps', type=int, default=-1,
                    help='max eval steps')
parser.add_argument('--sample_softmax', type=int, default=-1,
                    help='number of samples in sampled softmax')
parser.add_argument('--patience', type=int, default=0,
                    help='patience')
parser.add_argument('--finetune_v2', action='store_true',
                    help='finetune v2')
parser.add_argument('--finetune_v3', action='store_true',
                    help='finetune v3')
parser.add_argument('--fp16', action='store_true',
                    help='Run in pseudo-fp16 mode (fp16 storage fp32 math).')
parser.add_argument('--static-loss-scale', type=float, default=1,
                    help='Static loss scale, positive power of 2 values can '
                    'improve fp16 convergence.')
parser.add_argument('--dynamic-loss-scale', action='store_true',
                    help='Use dynamic loss scaling.  If supplied, this argument'
                    ' supersedes --static-loss-scale.')
parser.add_argument('--alpha', type=float, default= 0)
parser.add_argument('--student_steps_ratio', type=int, default= 5)
parser.add_argument('--T', type=float, default=1.5)
parser.add_argument('--exp_name', type=str, default='Transformer_LoT')
parser.add_argument('--max_epoch', type=int, default=4)
parser.add_argument('--start_epoch', type=int, default=-1)
parser.add_argument('--auto_step', type=int, default=1,
                    help='automatically set up max_steps according to max_epoch')



args = parser.parse_args()
args.tied = not args.not_tied

if args.d_embed < 0:
    args.d_embed = args.d_model

assert args.ext_len >= 0, 'extended context length must be non-negative'
assert args.batch_size % args.batch_chunk == 0

args.work_dir = '{}-{}'.format(args.work_dir, args.dataset)
args.work_dir = os.path.join(args.work_dir, time.strftime('%Y%m%d-%H%M%S'))
logging = create_exp_dir(args.work_dir,
    scripts_to_save=['transformer_xl_lm.py', 'mem_transformer.py'], debug=args.debug)

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print('WARNING: You have a CUDA device, so you should probably run with --cuda')
    else:
        torch.cuda.manual_seed_all(args.seed)

# Validate `--fp16` option
if args.fp16:
    if not args.cuda:
        print('WARNING: --fp16 requires --cuda, ignoring --fp16 option')
        args.fp16 = False
    else:
        try:
            from apex.fp16_utils import FP16_Optimizer
        except:
            print('WARNING: apex not installed, ignoring --fp16 option')
            args.fp16 = False

device = torch.device(f'cuda:{args.gpu}' if args.cuda else 'cpu')
config=configparser.ConfigParser()
config.read('key.config')
wandb_username=config.get('WANDB', 'USER_NAME')
wandb_key=config.get('WANDB', 'API_KEY')
wandb.login(key=wandb_key)
wandb.init(project='Transformer_LoT', entity=wandb_username, name=args.exp_name)

###############################################################################
# Load data
###############################################################################
corpus = get_lm_corpus(args.data, args.dataset)
ntokens = len(corpus.vocab)
args.n_token = ntokens

eval_batch_size = 10
tr_iter = corpus.get_iterator('train', args.batch_size, args.tgt_len,
    device=device, ext_len=args.ext_len)
va_iter = corpus.get_iterator('valid', eval_batch_size, args.eval_tgt_len,
    device=device, ext_len=args.ext_len)
te_iter = corpus.get_iterator('test', eval_batch_size, args.eval_tgt_len,
    device=device, ext_len=args.ext_len)
if args.auto_step:
    args.eval_interval = math.ceil(tr_iter.data.size(0) / args.tgt_len)
    args.max_step = args.max_epoch * args.eval_interval
    if args.dataset=='wt103':
        args.eval_interval = 1000


# adaptive softmax / embedding
cutoffs, tie_projs = [], [False]
if args.adaptive:
    assert args.dataset in ['wt103', 'lm1b', 'ptb']
    if args.dataset == 'wt103':
        cutoffs = [20000, 40000, 200000]
        tie_projs += [True] * len(cutoffs)
    elif args.dataset == 'lm1b':
        cutoffs = [60000, 100000, 640000]
        tie_projs += [False] * len(cutoffs)
    elif args.dataset == 'ptb':
        # TODO
        cutoffs = [20000, 40000, 200000]
        tie_projs += [False] * len(cutoffs)

###############################################################################
# Build the model
###############################################################################
def init_weight(weight):
    if args.init == 'uniform':
        nn.init.uniform_(weight, -args.init_range, args.init_range)
    elif args.init == 'normal':
        nn.init.normal_(weight, 0.0, args.init_std)

def init_bias(bias):
    nn.init.constant_(bias, 0.0)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            init_weight(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('AdaptiveEmbedding') != -1:
        if hasattr(m, 'emb_projs'):
            for i in range(len(m.emb_projs)):
                if m.emb_projs[i] is not None:
                    nn.init.normal_(m.emb_projs[i], 0.0, args.proj_init_std)
    elif classname.find('Embedding') != -1:
        if hasattr(m, 'weight'):
            init_weight(m.weight)
    elif classname.find('ProjectedAdaptiveLogSoftmax') != -1:
        if hasattr(m, 'cluster_weight') and m.cluster_weight is not None:
            init_weight(m.cluster_weight)
        if hasattr(m, 'cluster_bias') and m.cluster_bias is not None:
            init_bias(m.cluster_bias)
        if hasattr(m, 'out_projs'):
            for i in range(len(m.out_projs)):
                if m.out_projs[i] is not None:
                    nn.init.normal_(m.out_projs[i], 0.0, args.proj_init_std)
    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight, 1.0, args.init_std)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('TransformerLM') != -1:
        if hasattr(m, 'r_emb'):
            init_weight(m.r_emb)
        if hasattr(m, 'r_w_bias'):
            init_weight(m.r_w_bias)
        if hasattr(m, 'r_r_bias'):
            init_weight(m.r_r_bias)
        if hasattr(m, 'r_bias'):
            init_bias(m.r_bias)

def update_dropout(m):
    classname = m.__class__.__name__
    if classname.find('Dropout') != -1:
        if hasattr(m, 'p'):
            m.p = args.dropout

def update_dropatt(m):
    if hasattr(m, 'dropatt'):
        m.dropatt.p = args.dropatt

if args.restart:
    with open(os.path.join(args.restart_dir, 't_model.pt'), 'rb') as f:
        t_model = torch.load(f)
    if not args.fp16:
        t_model = t_model.float()
    t_model.apply(update_dropout)
    t_model.apply(update_dropatt)
else:
    t_model = MemTransformerLM(ntokens, args.n_layer, args.n_head, args.d_model,
        args.d_head, args.d_inner, args.dropout, args.dropatt,
        tie_weight=args.tied, d_embed=args.d_embed, div_val=args.div_val,
        tie_projs=tie_projs, pre_lnorm=args.pre_lnorm, tgt_len=args.tgt_len,
        ext_len=args.ext_len, mem_len=args.mem_len, cutoffs=cutoffs,
        same_length=args.same_length, attn_type=args.attn_type,
        clamp_len=args.clamp_len, sample_softmax=args.sample_softmax)
    t_model.apply(weights_init)
    t_model.word_emb.apply(weights_init) # ensure embedding init is not overridden by out_layer in case of weight sharing

    s_model = MemTransformerLM(ntokens, args.n_layer, args.n_head, args.d_model,
        args.d_head, args.d_inner, args.dropout, args.dropatt,
        tie_weight=args.tied, d_embed=args.d_embed, div_val=args.div_val,
        tie_projs=tie_projs, pre_lnorm=args.pre_lnorm, tgt_len=args.tgt_len,
        ext_len=args.ext_len, mem_len=args.mem_len, cutoffs=cutoffs,
        same_length=args.same_length, attn_type=args.attn_type,
        clamp_len=args.clamp_len, sample_softmax=args.sample_softmax)
    s_model.apply(weights_init)
    s_model.word_emb.apply(weights_init) # ensure embedding init is not overridden by out_layer in case of weight sharing

args.n_all_param = sum([p.nelement() for p in t_model.parameters()])*2
args.n_nonemb_param = sum([p.nelement() for p in t_model.layers.parameters()])*2

if args.fp16:
    t_model, s_model = t_model.half(), s_model.half()

if args.multi_gpu:
    t_model, s_model = t_model.to(device), s_model.to(device)
    if args.gpu0_bsz >= 0:
        t_para_model = BalancedDataParallel(args.gpu0_bsz // args.batch_chunk, t_model, dim=1).to(device)
        s_para_model = BalancedDataParallel(args.gpu0_bsz // args.batch_chunk, s_model, dim=1).to(device)
    else:
        t_para_model = nn.DataParallel(t_model, dim=1).to(device)
        s_para_model = nn.DataParallel(s_model, dim=1).to(device)
else:
    t_para_model = t_model.to(device)
    s_para_model = s_model.to(device)

#### optimizer
if args.optim.lower() == 'sgd':
    if args.sample_softmax > 0:
        t_dense_params, t_sparse_params = [], []
        for param in t_model.parameters():
            if param.size() == t_model.word_emb.weight.size():
                t_sparse_params.append(param)
            else:
                t_dense_params.append(param)
        t_optimizer_sparse = optim.SGD(t_sparse_params, lr=args.lr * 2)
        t_optimizer = optim.SGD(t_dense_params, lr=args.lr, momentum=args.mom)

        s_dense_params, s_sparse_params = [], []
        for param in s_model.parameters():
            if param.size() == s_model.word_emb.weight.size():
                s_sparse_params.append(param)
            else:
                s_dense_params.append(param)
        s_optimizer_sparse = optim.SGD(s_sparse_params, lr=args.lr * 2)
        s_optimizer = optim.SGD(s_dense_params, lr=args.lr, momentum=args.mom)
    else:
        t_optimizer = optim.SGD(t_model.parameters(), lr=args.lr,
            momentum=args.mom)
        s_optimizer = optim.SGD(s_model.parameters(), lr=args.lr,
            momentum=args.mom)
elif args.optim.lower() == 'adam':
    if args.sample_softmax > 0:
        t_dense_params, t_sparse_params = [], []
        for param in t_model.parameters():
            if param.size() == t_model.word_emb.weight.size():
                t_sparse_params.append(param)
            else:
                t_dense_params.append(param)
        t_optimizer_sparse = optim.SparseAdam(t_sparse_params, lr=args.lr)
        t_optimizer = optim.Adam(t_dense_params, lr=args.lr)

        s_dense_params, s_sparse_params = [], []
        for param in s_model.parameters():
            if param.size() == s_model.word_emb.weight.size():
                s_sparse_params.append(param)
            else:
                s_dense_params.append(param)
        s_optimizer_sparse = optim.SparseAdam(s_sparse_params, lr=args.lr)
        s_optimizer = optim.Adam(s_dense_params, lr=args.lr)
    else:
        t_optimizer = optim.Adam(t_model.parameters(), lr=args.lr)
        s_optimizer = optim.Adam(s_model.parameters(), lr=args.lr)
elif args.optim.lower() == 'adagrad':
    t_optimizer = optim.Adagrad(t_model.parameters(), lr=args.lr)
    s_optimizer = optim.Adagrad(s_model.parameters(), lr=args.lr)

#### scheduler
if args.scheduler == 'cosine':
    # here we do not set eta_min to lr_min to be backward compatible
    # because in previous versions eta_min is default to 0
    # rather than the default value of lr_min 1e-6
    t_scheduler = optim.lr_scheduler.CosineAnnealingLR(t_optimizer,
        args.max_step, eta_min=args.eta_min) # should use eta_min arg
    s_scheduler = optim.lr_scheduler.CosineAnnealingLR(s_optimizer,
        args.max_step, eta_min=args.eta_min) # should use eta_min arg
    
    if args.sample_softmax > 0:
        t_scheduler_sparse = optim.lr_scheduler.CosineAnnealingLR(t_optimizer_sparse,
            args.max_step, eta_min=args.eta_min) # should use eta_min arg
        s_scheduler_sparse = optim.lr_scheduler.CosineAnnealingLR(s_optimizer_sparse,
            args.max_step, eta_min=args.eta_min) # should use eta_min arg
elif args.scheduler == 'inv_sqrt':
    # originally used for Transformer (in Attention is all you need)
    def lr_lambda(step):
        # return a multiplier instead of a learning rate
        if step == 0 and args.warmup_step == 0:
            return 1.
        else:
            return 1. / (step ** 0.5) if step > args.warmup_step \
                   else step / (args.warmup_step ** 1.5)
    t_scheduler = optim.lr_scheduler.LambdaLR(t_optimizer, lr_lambda=lr_lambda)
    s_scheduler = optim.lr_scheduler.LambdaLR(s_optimizer, lr_lambda=lr_lambda)
elif args.scheduler == 'dev_perf':
    t_scheduler = optim.lr_scheduler.ReduceLROnPlateau(t_optimizer,
        factor=args.decay_rate, patience=args.patience, min_lr=args.lr_min)
    s_scheduler = optim.lr_scheduler.ReduceLROnPlateau(s_optimizer,
        factor=args.decay_rate, patience=args.patience, min_lr=args.lr_min)
    if args.sample_softmax > 0:
        t_scheduler_sparse = optim.lr_scheduler.ReduceLROnPlateau(t_optimizer_sparse,
            factor=args.decay_rate, patience=args.patience, min_lr=args.lr_min)
        s_scheduler_sparse = optim.lr_scheduler.ReduceLROnPlateau(s_optimizer_sparse,
            factor=args.decay_rate, patience=args.patience, min_lr=args.lr_min)
elif args.scheduler == 'constant':
    pass

if args.cuda and args.fp16:
    t_optimizer = FP16_Optimizer(t_optimizer,
                               static_loss_scale = args.static_loss_scale,
                               dynamic_loss_scale = args.dynamic_loss_scale,
                               dynamic_loss_args = {'init_scale': 2 ** 16})
    s_optimizer = FP16_Optimizer(s_optimizer,
                               static_loss_scale = args.static_loss_scale,
                               dynamic_loss_scale = args.dynamic_loss_scale,
                               dynamic_loss_args = {'init_scale': 2 ** 16})

if args.restart:
    if os.path.exists(os.path.join(args.restart_dir, 't_optimizer.pt')) and os.path.exists(os.path.join(args.restart_dir, 's_optimizer.pt')):
        with open(os.path.join(args.restart_dir, 't_optimizer.pt'), 'rb') as f:
            opt_state_dict = torch.load(f)
            t_optimizer.load_state_dict(opt_state_dict)
        with open(os.path.join(args.restart_dir, 's_optimizer.pt'), 'rb') as f:
            opt_state_dict = torch.load(f)
            s_optimizer.load_state_dict(opt_state_dict)
    else:
        print('Optimizer was not saved. Start from scratch.')

logging('=' * 100)
for k, v in args.__dict__.items():
    logging('    - {} : {}'.format(k, v))
logging('=' * 100)
logging('#params = {}'.format(args.n_all_param))
logging('#non emb params = {}'.format(args.n_nonemb_param))

###############################################################################
# Training code
###############################################################################

def evaluate(eval_iter):
    # Turn on evaluation mode which disables dropout.
    t_model.eval()
    s_model.eval()

    # If the model does not use memory at all, make the ext_len longer.
    # Otherwise, make the mem_len longer and keep the ext_len the same.
    if args.mem_len == 0:
        t_model.reset_length(args.eval_tgt_len,
            args.ext_len+args.tgt_len-args.eval_tgt_len, args.mem_len)
        s_model.reset_length(args.eval_tgt_len,
            args.ext_len+args.tgt_len-args.eval_tgt_len, args.mem_len)
    else:
        t_model.reset_length(args.eval_tgt_len,
            args.ext_len, args.mem_len+args.tgt_len-args.eval_tgt_len)
        s_model.reset_length(args.eval_tgt_len,
            args.ext_len, args.mem_len+args.tgt_len-args.eval_tgt_len)

    # Evaluation
    t_total_len, t_total_loss = 0, 0.
    with torch.no_grad():
        t_mems = tuple()
        for i, (data, target, seq_len) in enumerate(eval_iter):
            if args.max_eval_steps > 0 and i >= args.max_eval_steps:
                break
            t_ret, t_logit = t_model(data, target, *t_mems)
            t_loss, t_mems = t_ret[0], t_ret[1:]
            t_loss = t_loss.mean()
            t_total_loss += seq_len * t_loss.float().item()
            t_total_len += seq_len

    # Switch back to the training mode
    t_model.reset_length(args.tgt_len, args.ext_len, args.mem_len)
    t_model.train()

    s_total_len, s_total_loss = 0, 0.
    with torch.no_grad():
        s_mems = tuple()
        for i, (data, target, seq_len) in enumerate(eval_iter):
            if args.max_eval_steps > 0 and i >= args.max_eval_steps:
                break
            s_ret, s_logit = s_model(data, target, *s_mems)
            s_loss, s_mems = s_ret[0], s_ret[1:]
            s_loss = s_loss.mean()
            s_total_loss += seq_len * s_loss.float().item()
            s_total_len += seq_len

    # Switch back to the training mode
    s_model.reset_length(args.tgt_len, args.ext_len, args.mem_len)
    s_model.train()

    return t_total_loss / t_total_len, s_total_loss / s_total_len


def kl_div_logits(p, q, T):
    loss_func = nn.KLDivLoss(reduction = 'batchmean', log_target=True)
    loss = loss_func(F.log_softmax(p/T, dim=-1), F.log_softmax(q/T, dim=-1)) * T * T
    return loss

current_index = 0
def train():
    # Turn on training mode which enables dropout.
    global train_step, t_train_loss, t_best_val_loss, s_train_loss, s_best_val_loss, eval_start_time, log_start_time, current_index, epoch
    t_model.train()
    s_model.train()
    if args.batch_chunk > 1:
        t_mems = [tuple() for _ in range(args.batch_chunk)]
        s_mems = [tuple() for _ in range(args.batch_chunk)]
        s_t_mems = [tuple() for _ in range(args.batch_chunk)]
        s_s_mems = [tuple() for _ in range(args.batch_chunk)]
    else:
        t_mems = tuple()
        s_mems = tuple()
        s_t_mems = tuple()
        s_s_mems = tuple()
    train_iter = tr_iter.get_varlen_iter() if args.varlen else tr_iter
    for batch, (data, target, seq_len) in enumerate(train_iter):
        t_model.zero_grad()
        s_model.zero_grad()
        if args.batch_chunk > 1:
            data_chunks = torch.chunk(data, args.batch_chunk, 1)
            target_chunks = torch.chunk(target, args.batch_chunk, 1)
            for i in range(args.batch_chunk):
                data_i = data_chunks[i].contiguous()
                target_i = target_chunks[i].contiguous()
                
                t_ret, t_logit = t_para_model(data_i, target_i, *t_mems[i])
                t_loss, t_mems[i] = t_ret[0], t_ret[1:]

                s_ret, s_logit = s_para_model(data_i, target_i, *s_mems[i])
                s_loss, s_mems[i] = s_ret[0], s_ret[1:]
                t_loss = t_loss.float().mean().type_as(t_loss) / args.batch_chunk
                s_loss = s_loss.float().mean().type_as(s_loss) / args.batch_chunk
                if epoch > args.start_epoch:
                    t_loss += args.alpha * kl_div_logits(t_logit, s_logit.detach(), args.T)
                    s_loss += args.alpha * kl_div_logits(s_logit, t_logit.detach(), args.T)
                if args.fp16:
                    t_optimizer.backward(t_loss)
                    s_optimizer.backward(s_loss)
                else:
                    t_loss.backward()
                    s_loss.backward()

                t_train_loss += t_loss.float().item()
                s_train_loss += s_loss.float().item()

        else:
            t_ret, t_logit = t_para_model(data, target, *t_mems)
            t_loss, t_mems = t_ret[0], t_ret[1:]
            s_ret, s_logit = s_para_model(data, target, *s_mems)
            s_loss, s_mems = s_ret[0], s_ret[1:]
            t_loss = t_loss.float().mean().type_as(t_loss)
            s_loss = s_loss.float().mean().type_as(s_loss)
            if epoch > args.start_epoch:
                t_loss += args.alpha * kl_div_logits(t_logit, s_logit.detach(), args.T)
                s_loss += args.alpha * kl_div_logits(s_logit, t_logit.detach(), args.T)
            if args.fp16:
                t_optimizer.backward(t_loss)
                s_optimizer.backward(s_loss)
            else:
                t_loss.backward()
                s_loss.backward()
            t_train_loss += t_loss.float().item()
            s_train_loss += s_loss.float().item()

        if args.fp16:
            t_optimizer.clip_master_grads(args.clip)
            s_optimizer.clip_master_grads(args.clip)
        else:
            torch.nn.utils.clip_grad_norm_(t_model.parameters(), args.clip)
            torch.nn.utils.clip_grad_norm_(s_model.parameters(), args.clip)

        t_optimizer.step()
        s_optimizer.step()
        if args.sample_softmax > 0:
            t_optimizer_sparse.step()
            s_optimizer_sparse.step()
        
        for _ in range(args.student_steps_ratio - 1):
            s_data, s_target, s_seq_len = train_iter.get_batch(current_index)
            current_index=(s_seq_len+current_index) % (train_iter.data.size(0) - 1)
            t_model.zero_grad()
            s_model.zero_grad()
            if args.batch_chunk > 1:
                data_chunks = torch.chunk(s_data, args.batch_chunk, 1)
                target_chunks = torch.chunk(s_target, args.batch_chunk, 1)
                for i in range(args.batch_chunk):
                    data_i = data_chunks[i].contiguous()
                    target_i = target_chunks[i].contiguous()
                    
                    t_ret, t_logit = t_para_model(data_i, target_i, *s_t_mems[i])
                    t_loss, s_t_mems[i] = t_ret[0], t_ret[1:]
                    s_ret, s_logit = s_para_model(data_i, target_i, *s_s_mems[i])
                    s_loss, s_s_mems[i] = s_ret[0], s_ret[1:]
                    s_loss = s_loss.float().mean().type_as(s_loss) / args.batch_chunk
                    s_loss += args.alpha * kl_div_logits(s_logit, t_logit.detach(), args.T)

                    if args.fp16:
                        s_optimizer.backward(s_loss)
                    else:
                        s_loss.backward()
            else:
                t_ret, t_logit = t_para_model(s_data, s_target, *s_t_mems)
                t_loss, s_t_mems = t_ret[0], t_ret[1:]
                s_ret, s_logit = s_para_model(s_data, s_target, *s_s_mems)
                s_loss, s_s_mems = s_ret[0], s_ret[1:]
                s_loss = s_loss.float().mean().type_as(s_loss)
                s_loss += args.alpha * kl_div_logits(s_logit, t_logit.detach(), args.T)
                
                if args.fp16:
                    s_optimizer.backward(s_loss)
                else:
                    s_loss.backward()
            if args.fp16:
                s_optimizer.clip_master_grads(args.clip)
            else:
                torch.nn.utils.clip_grad_norm_(s_model.parameters(), args.clip)
            s_optimizer.step()
            if args.sample_softmax > 0:
                s_optimizer_sparse.step()

        # step-wise learning rate annealing
        train_step += 1
        if args.scheduler in ['cosine', 'constant', 'dev_perf']:
            # linear warmup stage
            if train_step < args.warmup_step:
                curr_lr = args.lr * train_step / args.warmup_step
                t_optimizer.param_groups[0]['lr'] = curr_lr
                s_optimizer.param_groups[0]['lr'] = curr_lr
                if args.sample_softmax > 0:
                    t_optimizer_sparse.param_groups[0]['lr'] = curr_lr * 2
                    s_optimizer_sparse.param_groups[0]['lr'] = curr_lr * 2
            else:
                if args.scheduler == 'cosine':
                    t_scheduler.step(train_step)
                    s_scheduler.step(train_step)
                    if args.sample_softmax > 0:
                        t_scheduler_sparse.step(train_step)
                        s_scheduler_sparse.step(train_step)
        elif args.scheduler == 'inv_sqrt':
            t_scheduler.step(train_step)
            s_scheduler.step(train_step)

        if train_step % args.log_interval == 0:
            t_cur_loss = t_train_loss / args.log_interval
            s_cur_loss = s_train_loss / args.log_interval
            elapsed = time.time() - log_start_time
            log_str = '| epoch {:3d} step {:>8d} | {:>6d} batches | lr {:.3g} ' \
                      '| ms/batch {:5.2f} | teacher_loss {:5.2f} | student_loss {:5.2f}'.format(
                epoch, train_step, batch+1, t_optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, t_cur_loss, s_cur_loss)
            if args.dataset in ['enwik8', 'text8']:
                log_str += ' | teacher bpc {:9.5f} | student bpc {:9.5f}'.format(t_cur_loss / math.log(2), s_cur_loss / math.log(2))
            else:
                log_str += ' | teacher ppl {:9.3f} | student ppl {:9.5f}'.format(math.exp(t_cur_loss), math.exp(s_cur_loss))
            logging(log_str)
            wandb.log({'lr': t_optimizer.param_groups[0]['lr'], 'teacher_train_loss': t_cur_loss, 'student_train_loss': s_cur_loss}, step=train_step)
            t_train_loss = 0
            s_train_loss = 0
            log_start_time = time.time()

        if train_step % args.eval_interval == 0:
            t_val_loss, s_val_loss = evaluate(va_iter)
            logging('-' * 100)
            log_str = '| Eval {:3d} at step {:>8d} | time: {:5.2f}s ' \
                      '| teacher valid loss {:5.2f} | student valid loss {:5.2f}'.format(
                train_step // args.eval_interval, train_step,
                (time.time() - eval_start_time), t_val_loss, s_val_loss)
            if args.dataset in ['enwik8', 'text8']:
                log_str += ' | teacher bpc {:9.5f} | student bpc {:9.5f}'.format(t_val_loss / math.log(2), s_val_loss / math.log(2))
            else:
                log_str += ' | teacher valid ppl {:9.3f} | student valid ppl {:9.3f}'.format(math.exp(t_val_loss), math.exp(s_val_loss))
            logging(log_str)
            logging('-' * 100)
            wandb.log({'teacher_valid_ppl': math.exp(t_val_loss), 'student_valid_ppl': math.exp(s_val_loss)}, step=train_step)
            # Save the model if the validation loss is the best we've seen so far.
            if not t_best_val_loss or t_val_loss < t_best_val_loss:
                if not args.debug:
                    with open(os.path.join(args.work_dir, 't_model.pt'), 'wb') as f:
                        torch.save(t_model, f)
                    with open(os.path.join(args.work_dir, 't_optimizer.pt'), 'wb') as f:
                        torch.save(t_optimizer.state_dict(), f)
                    with open(os.path.join(args.work_dir, 's_model.pt'), 'wb') as f:
                        torch.save(s_model, f)
                    with open(os.path.join(args.work_dir, 's_optimizer.pt'), 'wb') as f:
                        torch.save(s_optimizer.state_dict(), f)
                t_best_val_loss = t_val_loss

            # dev-performance based learning rate annealing
            if args.scheduler == 'dev_perf':
                t_scheduler.step(t_val_loss)
                s_scheduler.step(s_val_loss)
                if args.sample_softmax > 0:
                    t_scheduler_sparse.step(t_val_loss)
                    s_scheduler_sparse.step(s_val_loss)

            eval_start_time = time.time()

        if train_step == args.max_step or epoch > args.max_epoch:
            break
    

# Loop over epochs.
train_step = 0
t_train_loss = 0
t_best_val_loss = None
s_train_loss = 0

log_start_time = time.time()
eval_start_time = time.time()

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in itertools.count(start=1):
        train()
        if train_step == args.max_step or epoch > args.max_epoch:
            logging('-' * 100)
            logging('End of training')
            break
except KeyboardInterrupt:
    logging('-' * 100)
    logging('Exiting from training early')

# Load the best saved model.
with open(os.path.join(args.work_dir, 't_model.pt'), 'rb') as f:
    t_model = torch.load(f)
t_para_model = t_model.to(device)
with open(os.path.join(args.work_dir, 's_model.pt'), 'rb') as f:
    s_model = torch.load(f)
s_para_model = s_model.to(device)
# Run on test data.
t_test_loss, s_test_loss = evaluate(te_iter)
logging('=' * 100)
if args.dataset in ['enwik8', 'text8']:
    logging('| End of training | teacher test loss {:5.2f} | teacher test bpc {:9.5f} | student test loss {:5.2f} | student test bpc {:9.5f}'.format(
        t_test_loss, t_test_loss / math.log(2), s_test_loss, s_test_loss / math.log(2)))
else:
    logging('| End of training | teacher test loss {:5.2f} | teacher test ppl {:9.3f} | student test loss {:5.2f} | student test ppl {:9.3f}'.format(
        t_test_loss, math.exp(t_test_loss), s_test_loss, math.exp(s_test_loss)))
logging('=' * 100)
wandb.log({'teacher_test_ppl': math.exp(t_test_loss), 'student_test_ppl': math.exp(s_test_loss)}, step=train_step)
