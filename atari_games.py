# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
import wandb
import json
from collections import deque
import configparser


from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=0,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="LoT-RL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--student-steps-ratio", type=int, default=5,
        help="total timesteps of the experiments")
    parser.add_argument("--T", type=float, default=1,
        help="kl loss temperature")
    parser.add_argument("--detach", type=int, default=1,
        help="whether detach in kl loss")
    parser.add_argument("--alpha", type=float, default=1,
        help="LoT regularization coefficient")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--obs-num', type=int, default=10,
        help='the number of student train samples = num_envs * obs_num * num_steps')
    parser.add_argument('--student-weight-decay', type=float, default=0.0001)
    parser.add_argument('--teacher-scheduler', type=str, default='cosine', choices=['cosine', 'multistep'])
    parser.add_argument('--decreasing-step', type=list, default=[0.2, 0.4, 0.8])
    parser.add_argument('--save', type=str, default='test')
    parser.add_argument('--threshold', type=float, default=0,
        help='the ratio of total steps when LoT begins')

    parser.add_argument("--env-id", type=str, default="BeamRiderNoFrameskip-v4",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=20000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--learning-rate-min", type=float, default=2.5e-6,
        help="the minimum learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=8,
        help="the number of parallel game environments")
    parser.add_argument("--num-agent", type=int, default=2,
        help="the number of parallel game agent")
    parser.add_argument("--test-ensemble", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Test the ensemble agent or not")
    parser.add_argument("--smooth-return", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Smooth the return or not")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), logits


class Agent_ensemble(nn.Module):
    def __init__(self, agent_list):
        super().__init__()
        self.agent_list = agent_list
        self.num_models = len(agent_list)

    def get_action(self, x, action=None):

        # print("x shape:", x.shape)

        logits = []
        for i in range(self.num_models):
            hidden = self.agent_list[i].network(x / 255.0)
            logits.append(self.agent_list[i].actor(hidden))
        # print("original logits shape:", logits[0].shape)
        logits = esb_util.reduce_ensemble_logits(logits)

        # print("logits shape:", logits.shape)

        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic


def kl_div_logits(p, q, T):
    loss_func = nn.KLDivLoss(reduction = 'batchmean', log_target=True)
    loss = loss_func(F.log_softmax(p/T, dim=-1), F.log_softmax(q/T, dim=-1)) * T * T
    return loss


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(vars(args), indent=4))
    run_name = f"{args.env_id}__alpha{args.alpha}__N{args.student_steps_ratio}__seed{args.seed}__{int(time.time())}"
    config=configparser.ConfigParser()
    config.read('key.config')
    wandb_username=config.get('WANDB', 'USER_NAME')
    wandb_key=config.get('WANDB', 'API_KEY')
    wandb.login(key=wandb_key)
    wandb.init(project=args.wandb_project_name, entity=wandb_username, name=args.exp_name)
    wandb.define_metric('student_step')
    wandb.define_metric("student kl loss", step_metric="student_step")
    wandb.define_metric("student lr", step_metric="student_step")
    #### number of update epochs
    num_updates = args.total_timesteps // args.batch_size
    print(f'total updates: {num_updates}')
    obs_cache = deque(maxlen=args.obs_num)

    # TRY NOT TO MODIFY: seeding
    set_seed(args.seed)
    torch.cuda.set_device(int(args.gpu))
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # env setup
    teacher_envs = gym.vector.SyncVectorEnv(
            [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
        )
    assert isinstance(teacher_envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    # student use teacher env
    teacher_agent, student_agent = Agent(teacher_envs).to(device), Agent(teacher_envs).to(device)
    teacher_optimizer = optim.Adam(teacher_agent.parameters(), lr=args.learning_rate, eps=1e-5)
    student_optimizer = optim.Adam(student_agent.parameters(), lr=args.learning_rate, eps=1e-5, weight_decay=args.student_weight_decay)
    if args.teacher_scheduler == 'cosine':
        teacher_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = teacher_optimizer, T_max = num_updates, 
                        eta_min = args.learning_rate_min)
    else:
        teacher_scheduler = torch.optim.lr_scheduler.MultiStepLR(teacher_optimizer, milestones=[int(num_updates * _) for _ in args.decreasing_step], gamma=0.5)
    student_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = student_optimizer, T_max = num_updates, 
                        eta_min = args.learning_rate_min)

    # ALGO Logic: Storage setup
    teacher_obs = torch.zeros((args.num_steps, args.num_envs) + teacher_envs.single_observation_space.shape).to(device)
    teacher_actions = torch.zeros((args.num_steps, args.num_envs) + teacher_envs.single_action_space.shape).to(device)
    teacher_logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    teacher_rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    teacher_dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    teacher_values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: variables necessary to record the game while starting the game
    teacher_finished_runs = 0
    teacher_finished_frames = 0
    teacher_avg_return = 0.0
    teacher_avg_length = 0.0
    teacher_next_obs = torch.Tensor(teacher_envs.reset()).to(device)
    teacher_next_done = torch.zeros(args.num_envs).to(device)

    student_step = 0
    alpha = 0
    start_time = time.time()
    # Teacher Agent Play Games
    for update in range(1, num_updates + 1):
        for step in range(0, args.num_steps):
            teacher_obs[step] = teacher_next_obs
            teacher_dones[step] = teacher_next_done
            # ALGO LOGIC: action logic
            with torch.no_grad():
                teacher_action, teacher_logprob, _, teacher_value, _ = teacher_agent.get_action_and_value(teacher_next_obs)
                teacher_values[step] = teacher_value.flatten()
            teacher_actions[step] = teacher_action
            teacher_logprobs[step] = teacher_logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            teacher_next_obs, teacher_reward, teacher_done, teacher_info = teacher_envs.step(teacher_action.cpu().numpy())
            teacher_rewards[step] = torch.tensor(teacher_reward).to(device).view(-1)
            teacher_next_obs, teacher_next_done = torch.Tensor(teacher_next_obs).to(device), torch.Tensor(teacher_done).to(device)
        
            for item in teacher_info:
                teacher_finished_frames+=1
                if "episode" in item.keys():
                    teacher_finished_runs += 1
                    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), f"Teacher Agent play result: finished_runs={teacher_finished_runs}, episodic_return={item['episode']['r']}")
                    if args.smooth_return:
                        teacher_avg_return = 0.9 * teacher_avg_return + 0.1 * item["episode"]["r"]
                        teacher_avg_length = 0.9 * teacher_avg_length + 0.1 * item["episode"]["l"]
                    else:
                        teacher_avg_return = item["episode"]["r"]
                        teacher_avg_length = item["episode"]["l"]
                    wandb.log({'teacher avg return': teacher_avg_return, 'teacher return': item["episode"]["r"]}, step=teacher_finished_frames)

        with torch.no_grad():
            teacher_next_value = teacher_agent.get_value(teacher_next_obs).reshape(1, -1)
            teacher_advantages = torch.zeros_like(teacher_rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - teacher_next_done
                    nextvalues = teacher_next_value
                else:
                    nextnonterminal = 1.0 - teacher_dones[t + 1]
                    nextvalues = teacher_values[t + 1]
                delta = teacher_rewards[t] + args.gamma * nextvalues * nextnonterminal - teacher_values[t]
                teacher_advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            teacher_returns = teacher_advantages + teacher_values

        teacher_b_obs = teacher_obs.reshape((-1,) + teacher_envs.single_observation_space.shape)
        teacher_b_logprobs = teacher_logprobs.reshape(-1)
        teacher_b_actions = teacher_actions.reshape((-1,) + teacher_envs.single_action_space.shape)
        teacher_b_advantages = teacher_advantages.reshape(-1)
        teacher_b_returns = teacher_returns.reshape(-1)
        teacher_b_values = teacher_values.reshape(-1)

        # Optimizing the policy and value network
        teacher_b_inds = np.arange(args.batch_size)
        teacher_clipfracs = []
        obs_cache.append(teacher_b_obs)

        for epoch in range(args.update_epochs):
            np.random.shuffle(teacher_b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                teacher_mb_inds = teacher_b_inds[start:end]
                    
                _, teacher_newlogprob, teacher_entropy, teacher_newvalue, teacher_logits = teacher_agent.get_action_and_value(teacher_b_obs[teacher_mb_inds], teacher_b_actions.long()[teacher_mb_inds])
                teacher_logratio = teacher_newlogprob - teacher_b_logprobs[teacher_mb_inds]
                teacher_ratio = teacher_logratio.exp()
                _, _, _, _, student_logits = student_agent.get_action_and_value(teacher_b_obs[teacher_mb_inds], teacher_b_actions.long()[teacher_mb_inds])
                
                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    teacher_old_approx_kl = (-teacher_logratio).mean()
                    teacher_approx_kl = ((teacher_ratio - 1) - teacher_logratio).mean()
                    teacher_clipfracs += [((teacher_ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
                    
                teacher_mb_advantages = teacher_b_advantages[teacher_mb_inds]
                if args.norm_adv:
                    teacher_mb_advantages = (teacher_mb_advantages - teacher_mb_advantages.mean()) / (teacher_mb_advantages.std() + 1e-8)

                # Policy loss
                teacher_pg_loss1 = -teacher_mb_advantages * teacher_ratio
                teacher_pg_loss2 = -teacher_mb_advantages * torch.clamp(teacher_ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                teacher_pg_loss = torch.max(teacher_pg_loss1, teacher_pg_loss2).mean()
                # Value loss
                teacher_newvalue = teacher_newvalue.view(-1)
                if args.clip_vloss:
                    teacher_v_loss_unclipped = (teacher_newvalue - teacher_b_returns[teacher_mb_inds]) ** 2
                    teacher_v_clipped = teacher_b_values[teacher_mb_inds] + torch.clamp(
                        teacher_newvalue - teacher_b_values[teacher_mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    teacher_v_loss_clipped = (teacher_v_clipped - teacher_b_returns[teacher_mb_inds]) ** 2
                    teacher_v_loss_max = torch.max(teacher_v_loss_unclipped, teacher_v_loss_clipped)
                    teacher_v_loss = 0.5 * teacher_v_loss_max.mean()
                else:
                    teacher_v_loss = 0.5 * ((teacher_newvalue - teacher_b_returns[teacher_mb_inds]) ** 2).mean()
                # CE loss
                teacher_entropy_loss = teacher_entropy.mean()

                teacher_loss = teacher_pg_loss - args.ent_coef * teacher_entropy_loss + teacher_v_loss * args.vf_coef
                if args.detach:
                    teacher_loss += alpha * kl_div_logits(teacher_logits, student_logits.detach(), args.T)
                else:
                    teacher_loss += alpha * kl_div_logits(teacher_logits, student_logits, args.T)

                if args.detach:
                    student_loss = kl_div_logits(student_logits, teacher_logits.detach(), args.T) 
                else:
                    student_loss = kl_div_logits(student_logits, teacher_logits, args.T)

                loss =  student_loss + teacher_loss
                ##### Compute final gradient
                teacher_optimizer.zero_grad()
                student_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(teacher_agent.parameters(), args.max_grad_norm)
                nn.utils.clip_grad_norm_(student_agent.parameters(), args.max_grad_norm)
                teacher_optimizer.step()
                student_optimizer.step()

        student_step+=4
        teacher_y_pred, teacher_y_true = teacher_b_values.cpu().numpy(), teacher_b_returns.cpu().numpy()
        teacher_var_y = np.var(teacher_y_true)
        teacher_explained_var = np.nan if teacher_var_y == 0 else 1 - np.var(teacher_y_true - teacher_y_pred) / teacher_var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        wandb.log({'teacher lr': teacher_optimizer.param_groups[0]["lr"]
                   ,'teacher value loss': teacher_v_loss
                   ,'teacher policy loss': teacher_pg_loss}, step=teacher_finished_frames)
        wandb.log({'student kl loss': student_loss, 'student lr': student_optimizer.param_groups[0]["lr"], 'student_step': student_step})

        # student additional train
        for _ in range(args.student_steps_ratio):
            alpha = args.alpha
            for e in range(len(obs_cache)):
                teacher_b_obs = obs_cache[e]
                np.random.shuffle(teacher_b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    teacher_mb_inds = teacher_b_inds[start:end]
                    _, _, _, _, teacher_logits = teacher_agent.get_action_and_value(teacher_b_obs[teacher_mb_inds], teacher_b_actions.long()[teacher_mb_inds])
                    _, _, _, _, student_logits = student_agent.get_action_and_value(teacher_b_obs[teacher_mb_inds], teacher_b_actions.long()[teacher_mb_inds])
                    if args.detach:
                        student_loss = kl_div_logits(student_logits, teacher_logits.detach(), args.T)
                    else:
                        student_loss = kl_div_logits(student_logits, teacher_logits, args.T)
                    ##### Compute gradient
                    student_optimizer.zero_grad()
                    student_loss.backward()
                    nn.utils.clip_grad_norm_(student_agent.parameters(), args.max_grad_norm)
                    student_optimizer.step()
                print(f'student additional train: student steps {student_step}, kl loss {student_loss}, lr {student_optimizer.param_groups[0]["lr"]}' )
                wandb.log({'student kl loss': student_loss, 'student lr': student_optimizer.param_groups[0]["lr"], 'student_step': student_step})
                student_step+=1
        
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            ####### Cosine anneal
            teacher_scheduler.step()
            student_scheduler.step()

    teacher_envs.close()
    torch.save(student_agent.state_dict(), args.save+'_student')
    torch.save(teacher_agent.state_dict(), args.save+'_teacher')
    wandb.finish()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "Finished all runs")