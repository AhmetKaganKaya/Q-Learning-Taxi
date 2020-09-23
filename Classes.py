import torch
import torch.nn as nn
import numpy as np



class Agent():
    def __init__(self, gamma, eps, lr, n_actions, n_states):
        self.gamma = gamma
        self.eps = eps
        self.lr = lr
        self.n_states = n_states
        self.n_actions = n_actions
        self.table = np.zeros((self.n_states, self.n_actions))

    def choose_action(self, observation):
        if np.random.random() > self.eps:
            action = np.argmax(self.table[observation])
        else:
            action = np.random.choice(self.n_actions)
        return action

    def learn(self, state, next_state, action, reward, done):
        optimal_action = np.argmax(self.table[next_state])
        self.table[state, action] =  (1- self.lr) * self.table[state, action] + \
                                     self.lr * (reward + self.gamma * self.table[next_state, optimal_action] * (1 - int(done)))










