import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from api.definer.config import *


class ActorNet(nn.Module):  # define the network structure for actor and critic
    def __init__(self, s_dim, a_dim):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(s_dim, 30)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization of FC1
        self.out = nn.Linear(30, a_dim)
        self.out.weight.data.normal_(0, 0.1)  # initialization of OUT

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.out(x)
        x = torch.tanh(x)
        actions = x * 2  # for the game "Pendulum-v0", action range is [-2, 2]
        return actions


class CriticNet(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(CriticNet, self).__init__()
        self.fcs = nn.Linear(s_dim, 30)
        self.fcs.weight.data.normal_(0, 0.1)
        self.fca = nn.Linear(a_dim, 30)
        self.fca.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(30, 1)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, s, a):
        x = self.fcs(s)
        y = self.fca(a)
        actions_value = self.out(F.relu(x + y))
        return actions_value


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_high_bound):
        self.a_dim, self.s_dim, self.a_high_bound = a_dim, s_dim, a_high_bound
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0  # serves as updating the memory data

        # Create the 4 network objects
        self.actor_eval = ActorNet(s_dim, a_dim)
        self.actor_target = ActorNet(s_dim, a_dim)
        self.critic_eval = CriticNet(s_dim, a_dim)
        self.critic_target = CriticNet(s_dim, a_dim)

        # create 2 optimizers for actor and critic
        self.actor_optimizer = torch.optim.Adam(self.actor_eval.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = torch.optim.Adam(self.critic_eval.parameters(), lr=LR_CRITIC)

        # Define the loss function for critic network update
        self.loss_func = nn.MSELoss()

    def store_transition(self, state, action, reward, state_next):  # how to store the episodic data to buffer
        transition = np.hstack((state, action, [reward], state_next))
        index = self.pointer % MEMORY_CAPACITY  # replace the old data with new data
        self.memory[index, :] = transition
        self.pointer += 1

    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        return self.actor_eval(state)[0].detach()

    def learn(self):
        # softly update the target networks
        for x in self.actor_target.state_dict().keys():
            eval('self.actor_target.' + x + '.data.mul_((1-TAU))')
            eval('self.actor_target.' + x + '.data.add_(TAU*self.actor_eval.' + x + '.data)')
        for x in self.critic_target.state_dict().keys():
            eval('self.critic_target.' + x + '.data.mul_((1-TAU))')
            eval('self.critic_target.' + x + '.data.add_(TAU*self.critic_eval.' + x + '.data)')

            # sample from buffer a mini-batch data
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        batch_trans = self.memory[indices, :]

        # extract data from mini-batch of transitions including s, a, r, s_
        batch_s = torch.FloatTensor(batch_trans[:, :self.s_dim])
        batch_a = torch.FloatTensor(batch_trans[:, self.s_dim:self.s_dim + self.a_dim])
        batch_r = torch.FloatTensor(batch_trans[:, -self.s_dim - 1: -self.s_dim])
        batch_s_ = torch.FloatTensor(batch_trans[:, -self.s_dim:])

        # make action and evaluate its action values
        a = self.actor_eval(batch_s)
        q = self.critic_eval(batch_s, a)
        actor_loss = -torch.mean(q)

        # optimize the loss of actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # compute the target Q value using the information of next state
        a_target = self.actor_target(batch_s_)
        q_tmp = self.critic_target(batch_s_, a_target)
        q_target = batch_r + GAMMA * q_tmp

        # compute the current q value and the loss
        q_eval = self.critic_eval(batch_s, batch_a)
        td_error = self.loss_func(q_target, q_eval)

        # optimize the loss of critic network
        self.critic_optimizer.zero_grad()
        td_error.backward()
        self.critic_optimizer.step()