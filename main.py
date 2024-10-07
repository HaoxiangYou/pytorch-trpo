import argparse
import os
import time

import gym
import scipy.optimize
import numpy as np

import torch
from torch.optim.lbfgs import LBFGS

from models import *
from replay_memory import Memory
from running_state import ZFilter
from torch.autograd import Variable
from trpo import trpo_step
from utils import *
from logger import Logger

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--exp-name', type=str, required=True, help="name of experiments")
parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                    help='discount factor (default: 0.995)')
parser.add_argument('--env-name', default="Reacher-v1", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--tau', type=float, default=0.97, metavar='G',
                    help='gae (default: 0.97)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--step_size', type=float, metavar='G',
                    help='fixed step (natural gradient setting) or max-kl (trpo setting)')
parser.add_argument('--no_mirror_descent', action='store_true', help="whether treat the problem as mirror descent")
parser.add_argument('--lbfgs', action="store_true", help="run lbfgs for policy; can only be trigger given no_mirror_descent flag")
parser.add_argument('--damping', type=float, default=1e-1, metavar='G',
                    help='damping (default: 1e-1)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=15000, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--epochs', type=int, default=100, 
                    help='number of training epochs')

args = parser.parse_args()

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")

logdir = (args.exp_name
        + "_"
        + args.env_name
        + "_"
        + time.strftime("%d-%m-%Y_%H-%M-%S")
    )
logdir = os.path.join(data_path, logdir)
args.logdir = logdir
if not (os.path.exists(logdir)):
    os.makedirs(logdir)

env = gym.make(args.env_name)

exp_logger = Logger(logdir)

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

torch.manual_seed(args.seed)
np.random.seed(args.seed)

policy_net = Policy(num_inputs, num_actions)
value_net = Value(num_inputs)

# If with fixed step size then natural policy gradient
# Else stay with TRPO (if mirror descent is true)

step_size = args.max_kl
is_trpo = True

if args.step_size:
    step_size = args.step_size
    is_trpo = False

if args.no_mirror_descent:
    if args.lbfgs:
        optimizer = LBFGS(policy_net.parameters(), lr=step_size)
    else:
        optimizer = torch.optim.Adam(policy_net.parameters(), step_size)

def select_action(state):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state))
    action = torch.normal(action_mean, action_std)
    return action

def update_params(batch):
    rewards = torch.Tensor(batch.reward)
    masks = torch.Tensor(batch.mask)
    actions = torch.Tensor(np.concatenate(batch.action, 0))
    states = torch.Tensor(batch.state)
    values = value_net(Variable(states))

    returns = torch.Tensor(actions.size(0),1)
    deltas = torch.Tensor(actions.size(0),1)
    advantages = torch.Tensor(actions.size(0),1)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values.data[i]
        advantages[i] = deltas[i] + args.gamma * args.tau * prev_advantage * masks[i]

        prev_return = returns[i, 0]
        prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]

    targets = Variable(returns)

    # Original code uses the same LBFGS to optimize the value loss
    def get_value_loss(flat_params):
        set_flat_params_to(value_net, torch.Tensor(flat_params))
        for param in value_net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)

        values_ = value_net(Variable(states))

        value_loss = (values_ - targets).pow(2).mean()

        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * args.l2_reg
        value_loss.backward()
        return (value_loss.data.double().numpy(), get_flat_grad_from(value_net).data.double().numpy())

    flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss, get_flat_params_from(value_net).double().numpy(), maxiter=25)
    set_flat_params_to(value_net, torch.Tensor(flat_params))

    advantages = (advantages - advantages.mean()) / advantages.std()

    action_means, action_log_stds, action_stds = policy_net(Variable(states))
    fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()
    fixed_mean, fixed_log_std, fixed_std = tuple(e.data.clone() for e in policy_net(Variable(states)))

    def get_loss(volatile=False):
        if volatile:
            with torch.no_grad():
                action_means, action_log_stds, action_stds = policy_net(Variable(states))
        else:
            action_means, action_log_stds, action_stds = policy_net(Variable(states))
                
        log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
        action_loss = -Variable(advantages) * torch.exp(log_prob - Variable(fixed_log_prob))
        return action_loss.mean()

    def closure():
        optimizer.zero_grad()
        loss = get_loss(volatile=False)
        loss.backward()
        return loss

    def get_kl():
        """
        This function in the trpo to approximate the fisher information
        """
        mean1, log_std1, std1 = policy_net(Variable(states))

        mean0 = Variable(mean1.data)
        log_std0 = Variable(log_std1.data)
        std0 = Variable(std1.data)
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)
    
    def compute_kl():
        """
        This function compute the average KL between the old policy and current policy
        """
        mean1, log_std1, std1 = policy_net(Variable(states))
        kl = log_std1 - fixed_log_std + (fixed_std.pow(2) + (fixed_mean - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True).mean()

    if not args.no_mirror_descent:
        trpo_step(policy_net, get_loss, get_kl, compute_kl, step_size, args.damping, is_trpo)
    else:
        previous_params = get_flat_params_from(policy_net)
        fval = get_loss().data
        
        optimizer.zero_grad()
        loss = get_loss()
        loss.backward()
        gradients = get_flat_grad_from(policy_net)
        
        if args.lbfgs:
            optimizer.step(closure)
        else:
            optimizer.step()
        new_params = get_flat_params_from(policy_net)

        newfval = get_loss().data
        actual_improve = fval - newfval
        expected_improve = (-gradients * (new_params - previous_params)).sum()
        ratio = actual_improve / expected_improve
        print("a/e/r", actual_improve.item(), expected_improve.item(), ratio.item())

        print("kl:", compute_kl().data)

running_state = ZFilter((num_inputs,), clip=5)
running_reward = ZFilter((1,), demean=False, clip=10)

for i_episode in range(args.epochs):
    memory = Memory()

    num_steps = 0
    reward_batch = 0
    num_episodes = 0
    while num_steps < args.batch_size:
        state = env.reset()
        state = running_state(state)

        reward_sum = 0
        for t in range(10000): # Don't infinite loop while learning
            action = select_action(state)
            action = action.data[0].numpy()
            next_state, reward, done, _ = env.step(action)
            reward_sum += reward

            next_state = running_state(next_state)

            mask = 1
            if done:
                mask = 0

            memory.push(state, np.array([action]), mask, next_state, reward)

            if args.render:
                env.render()
            if done:
                break

            state = next_state
        num_steps += (t-1)
        num_episodes += 1
        reward_batch += reward_sum

    reward_batch /= num_episodes
    batch = memory.sample()
    update_params(batch)

    if i_episode % args.log_interval == 0:
        print('Episode {}\tLast reward: {}\tAverage reward {:.2f}'.format(
            i_episode, reward_sum, reward_batch))
        exp_logger.log_scalar(reward_batch, "Average reward", i_episode)
        exp_logger.flush()