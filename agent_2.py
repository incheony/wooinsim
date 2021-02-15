import os
import gym
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple, deque

import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    T.manual_seed(seed)
    T.cuda.manual_seed(seed)
    T.backends.cudnn.benchmark = False
    T.backends.cudnn.deterministic = True
    print("..........")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'n_reward'))

def get_demo_traj():
    gamma_demo = 1.0
    gamma_decay = 0.99
    n_reward_demo = 0
    demo_memory = ReplayMemory(1000)
    data = np.load("demo_traj_2.npy", allow_pickle=True)
    for i in range(len(data)):
        for j in range(len(data[i])):
            state = data[i][j][0]
            action = data[i][j][1]
            reward = data[i][j][2]
            new_state = data[i][j][3]
            if j % 10 == 0 and j != 0:
                gamma_demo = 1
                n_reward_demo = 0
            gamma_demo = gamma_demo * gamma_decay
            n_reward_demo = n_reward_demo + gamma_demo * reward
            demo_memory.push(Transition(state, action, new_state, reward, n_reward_demo))

    print(f"Number of demo samples {demo_memory.position}")
    return demo_memory

class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])

class ReplayMemory_PER:
    # stored as ( s, a, r, s_ ) in SumTree
    # values come from the DQfD paper
    e = 0.001
    a = 0.4
    beta = 0.6
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def push(self, error, trans):
        p = self._get_priority(error)
        self.tree.add(p, trans)

    def sample(self, batch_size):
        batch = []
        idxs = []
        seg = self.tree.total() / batch_size
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(batch_size):
            a = seg * i
            b = seg * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_prob = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_prob, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def __len__(self):
        return self.tree.n_entries

class ReplayMemory(object):

    # capacity == -1 means unlimited capacity
    def __init__(self, capacity=-1):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, trans):
        if len(self.memory) < self.capacity or self.capacity < 0:
            self.memory.append(None)
        self.memory[self.position] = trans
        self.position = self.position + 1
        if self.capacity > 0:
            self.position = self.position % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

##########################################################################
############                                                  ############
############                  DQfDNetwork 구현                 ############
############                                                  ############
##########################################################################
class DQfDNetwork(nn.Module):
    def __init__(self, dtype, input_shape=4, num_actions=1):
        super(DQfDNetwork, self).__init__()
        self.dtype = dtype

        self.lin1 = nn.Linear(input_shape, 256)
        self.lin2 = nn.Linear(256, 256)
        self.lin3 = nn.Linear(256, 256)
        self.lin4 = nn.Linear(256, num_actions)
        self.type(dtype)

    def forward(self, states):
        x = self.lin1(states)
        x = F.leaky_relu(self.lin2(x))
        x = F.leaky_relu(self.lin3(x))
        return self.lin4(x)

##########################################################################
############                                                  ############
############                  DQfDagent 구현                   ############
############                                                  ############
##########################################################################

class DQfDAgent(object):
    def __init__(self, env, use_per, n_episode):
        super(DQfDAgent, self).__init__()

        seed_torch()

        self.n_EPISODES = n_episode

        self.env = env
        self.env.seed(0)

        self.lr = 0.00005
        self.batch_size = 32
        self.gamma_decay = 0.99
        self.epsilon_start = 1.0
        self.epsilon_end = 0.0
        self.epsilon_steps = 1000
        self.steps_done = 0
        self.opt_step = 0
        self.tau = 0.005

        self.demo_prop = 0.1
        self.margin = 10.0
        self.lr_sup = 1.0
        self.lr_n_step = 1.0

        self.alpha = 0.1

        self.use_per = use_per

        if self.use_per:
            self.memory = ReplayMemory_PER(100000)
        else:
            self.memory = ReplayMemory(100000)

        self.q_network = DQfDNetwork(T.FloatTensor, self.env.observation_space.shape[0], self.env.action_space.n)
        self.target_q_network = DQfDNetwork(T.FloatTensor, self.env.observation_space.shape[0], self.env.action_space.n)

        self.optimizer = optim.Adam(self.q_network.parameters(), self.lr, weight_decay=1e-5)
        self.demo_memory = get_demo_traj()

        self.update_network_parameters(1)

    def get_action(self, state):
        sample = random.random()
        self.steps_done += 1
        if sample > min(self.steps_done / self.epsilon_steps, 1) * (self.epsilon_end - self.epsilon_start) + self.epsilon_start:
            q_vals = self.q_network(T.Tensor(state)).data
            return q_vals.argmax().numpy()
        else:
            return self.env.action_space.sample()

    def pretrain(self):
        ## Do pretrain for 1000 steps
        for i in range(1000):
            self.optimize(1, 1.0)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_q_network.named_parameters()
        value_params = self.q_network.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        # copy behaviour network parameters to target network parameters
        for name in value_state_dict:
            value_state_dict[name] = tau * value_state_dict[name].clone() + \
                                     (1 - tau) * target_value_state_dict[name].clone()

        self.target_q_network.load_state_dict(value_state_dict)

    def optimize(self, batch_size=32, proportion=0.3, opt_step=0):
        # argument: proportion => number of samples from demo
        demo_samples = int(batch_size * proportion)
        demo_trans = []
        if demo_samples > 0:
            demo_trans = self.demo_memory.sample(demo_samples)

        if demo_samples != batch_size:
            if self.use_per:
                agent_trans, idx, _ = self.memory.sample(batch_size - demo_samples)
            else:
                agent_trans = self.memory.sample(batch_size - demo_samples)
            transitions = demo_trans + agent_trans
        else:
            transitions = demo_trans
        batch = Transition(*zip(*transitions))

        # creating PyTorch tensors for the transitions and calculating the q vals for the actions taken
        null_mask = T.BoolTensor(tuple(map(lambda s: s is not None, batch.next_state)))
        next_state_batch_ = [s for s in batch.next_state if s is not None]
        next_state_batch = T.Tensor(next_state_batch_)
        state_batch = T.Tensor(batch.state)
        action_batch = T.LongTensor(batch.action).unsqueeze(1)
        reward_batch = T.Tensor(batch.reward)
        n_reward_batch = T.Tensor(batch.n_reward)

        q_vals = self.q_network(state_batch)
        state_action_values = q_vals.gather(1, action_batch).squeeze()

        # comparing the q values to the values expected using the next states and reward
        next_state_values = T.Tensor(T.zeros(batch_size))
        next_state_values[null_mask] = self.target_q_network(next_state_batch).data.max(1)[0]
        expected_state_action_values = (next_state_values * self.gamma_decay) + reward_batch

        # q loss
        q_loss = F.mse_loss(state_action_values, expected_state_action_values, reduction="mean")  # td_error

        # 10 steps loss
        n_step_loss = F.mse_loss(state_action_values, n_reward_batch, reduction="mean")

        # calculating the supervised loss
        num_actions = q_vals.size(1)
        margins = (T.ones(num_actions, num_actions) - T.eye(num_actions)) * self.margin
        batch_margins = margins[action_batch.data.squeeze()]
        q_vals = q_vals + T.Tensor(batch_margins).type(T.FloatTensor)
        # supervised loss from demo samples, network to be trained = q_vals, target network = state_action_values
        # state_action_values come from the demo sample with good rewards
        supervised_loss = (q_vals.max(1)[0] - state_action_values).pow(2)[:demo_samples].sum()

        # total loss
        loss = q_loss + self.lr_sup * supervised_loss + self.lr_n_step * n_step_loss

        # optimization step and logging
        self.optimizer.zero_grad()
        loss.backward()
        T.nn.utils.clip_grad_norm_(self.q_network.parameters(), 100)
        self.optimizer.step()

        if opt_step % 10000 == 0:
            self.update_network_parameters(self.tau)

        # print(self.opt_step)
        self.opt_step += 1


    def train(self):
        ###### 1. DO NOT MODIFY FOR TESTING ######
        test_mean_episode_reward = deque(maxlen=20)
        test_over_reward = False
        test_min_episode = np.inf
        ###### 1. DO NOT MODIFY FOR TESTING ######

        f = open("result.txt", "a")

        # Do pretrain
        print("Pre-training with 1000 steps")
        self.pretrain()
        print("Pre-training done...")
        mean_rewards = []

        for e in range(self.n_EPISODES):
            ########### 2. DO NOT MODIFY FOR TESTING ###########
            test_episode_reward = 0
            ########### 2. DO NOT MODIFY FOR TESTING  ###########

            state = self.env.reset()
            transitions = []
            done = False

            while not done:

                # selecting an action and playing it
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                # reward = -test_episode_reward / 100 if done and test_episode_reward < 499 else reward

                ########### 3. DO NOT MODIFY FOR TESTING ###########
                test_episode_reward += reward
                ########### 3. DO NOT MODIFY FOR TESTING  ###########
                reward = T.Tensor([reward])

                # storing the transition in a temporary replay buffer which is held in order to calculate n-step returns
                transitions.insert(0, Transition(state, action, next_state, reward, T.zeros(1)))
                gamma = 1
                new_trans = []
                for trans in transitions:
                    new_trans.append(trans._replace(n_reward=trans.n_reward + gamma * reward))
                    gamma = gamma * self.gamma_decay
                transitions = new_trans

                # for 10 steps loss
                if not done:
                    q_vals = self.q_network(T.Tensor(next_state)).data
                    if len(transitions) >= 10:
                        last_trans = transitions.pop()
                        last_trans = last_trans._replace(n_reward=last_trans.n_reward + gamma * q_vals.max())
                        if self.use_per:
                            q_old = self.q_network(T.Tensor(last_trans.state)).detach().numpy()[last_trans.action]
                            q_new = self.q_network(T.Tensor(last_trans.next_state)).detach().numpy()[last_trans.action]
                            q_new = (q_new * self.gamma_decay) + last_trans.reward.numpy()[0]
                            q_old = (1 - self.alpha) * q_old + (self.alpha * q_new)
                            error = abs(q_new - q_old)
                            error = np.clip(error, 0, 1)      # td-error
                            self.memory.push(error, last_trans)
                        else:
                            self.memory.push(last_trans)
                else:
                    for i in range(len(transitions)):
                        if self.use_per:
                            q_old = self.q_network(T.Tensor(transitions[i].state)).detach().numpy()[transitions[i].action]
                            if i == 0:
                                q_new = transitions[i].reward.numpy()[0]
                            else:
                                q_new = self.q_network(T.Tensor(transitions[i].next_state)).detach().numpy()[transitions[i].action]
                                q_new = (q_new * self.gamma_decay) + transitions[i].reward.numpy()[0]
                            q_old = (1 - self.alpha) * q_old + (self.alpha * q_new)
                            error = abs(q_new - q_old)
                            error = np.clip(error, 0, 1)  # td-error

                            self.memory.push(error, transitions[i])
                        else:
                            self.memory.push(transitions[i])

                state = next_state

                # optimization step for the network the network
                if len(self.memory) >= 1000:
                    self.optimize(128, self.demo_prop, self.opt_step)

                ########### 4. DO NOT MODIFY FOR TESTING  ###########
                if done:
                    test_mean_episode_reward.append(test_episode_reward)
                    print(f"Episode {e}: {test_episode_reward}")
                    f.write(str(test_episode_reward) + " ")
                    if (np.mean(test_mean_episode_reward) > 475) and (len(test_mean_episode_reward) == 20):
                        test_over_reward = True
                        test_min_episode = e
                ########### 4. DO NOT MODIFY FOR TESTING  ###########
            mean_rewards.append(np.mean(test_mean_episode_reward))
            ########### 5. DO NOT MODIFY FOR TESTING  ###########
            if test_over_reward:
                print("END train function")
                break
            ########### 5. DO NOT MODIFY FOR TESTING  ###########
        f.write("\n")
        f.close()
        self.env.close()

        ########### 6. DO NOT MODIFY FOR TESTING  ###########
        return test_min_episode, np.mean(test_mean_episode_reward)
        ########### 6. DO NOT MODIFY FOR TESTING  ###########