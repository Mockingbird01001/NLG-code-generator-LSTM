import torch
import torch.nn as nn
import numpy as np
class QNet(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, action_dim))
    def forward(self, state):
        return self.net(state)
class QNetTwin(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net_state = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                       nn.Linear(mid_dim, mid_dim), nn.ReLU())
        self.net_q1 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, action_dim))
        self.net_q2 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, action_dim))
    def forward(self, state):
        tmp = self.net_state(state)
        return self.net_q1(tmp)
    def get_q1_q2(self, state):
        tmp = self.net_state(state)
        return self.net_q1(tmp), self.net_q2(tmp)
class Actor(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, action_dim))
    def forward(self, state):
        return self.net(state).tanh()
    def get_action(self, state, action_std):
        action = self.net(state).tanh()
        noise = (torch.randn_like(action) * action_std).clamp(-0.5, 0.5)
        return (action + noise).clamp(-1.0, 1.0)
class ActorSAC(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net_state = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                       nn.Linear(mid_dim, mid_dim), nn.ReLU())
        self.net_a_avg = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                       nn.Linear(mid_dim, action_dim))
        self.net_a_std = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                       nn.Linear(mid_dim, action_dim))
        self.num_logprob = -np.log(action_dim)
        self.log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))
        self.log_alpha = nn.Parameter(torch.zeros((1, action_dim)) - np.log(action_dim), requires_grad=True)
    def forward(self, state):
        tmp = self.net_state(state)
        return self.net_a_avg(tmp).tanh()
    def get_action(self, state):
        t_tmp = self.net_state(state)
        a_avg = self.net_a_avg(t_tmp)
        a_std = self.net_a_std(t_tmp).clamp(-16, 2).exp()
        return torch.normal(a_avg, a_std).tanh()
    def get_action_logprob(self, state):
        t_tmp = self.net_state(state)
        a_avg = self.net_a_avg(t_tmp)
        a_std_log = self.net_a_std(t_tmp).clamp(-16, 2)
        a_std = a_std_log.exp()
        noise = torch.randn_like(a_avg, requires_grad=True)
        action = a_avg + a_std * noise
        a_tan = action.tanh()
        logprob = -(a_std_log + self.log_sqrt_2pi + ((a_avg - action) / a_std).pow(2) * 0.5)
        logprob = logprob - (-a_tan.pow(2) + 1.000001).log()
        return a_tan, logprob.sum(1, keepdim=True)
    def get_obj_alpha(self, logprob):
        return -(self.log_alpha * (logprob - self.num_logprob).detach()).mean()
class ActorPPO(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                 nn.Linear(mid_dim, action_dim), )
        self.a_logstd = nn.Parameter(torch.zeros((1, action_dim)) - 0.5, requires_grad=True)
        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))
    def forward(self, state):
        return self.net(state).tanh()
    def get_action(self, state):
        a_avg = self.net(state)
        noise = torch.randn_like(a_avg)
        action = a_avg + noise * self.a_logstd.exp()
        return action, noise
    def get_new_logprob_entropy(self, state, action):
        a_avg = self.net(state)
        a_std = self.a_logstd.exp()
        delta = ((a_avg - action) / a_std).pow(2) * 0.5
        logprob = -(self.a_logstd + self.sqrt_2pi_log + delta).sum(1)
        dist_entropy = (logprob.exp() * logprob).mean()
        return logprob, dist_entropy
    def get_old_logprob(self, _action, noise):
        return -(self.a_logstd + self.sqrt_2pi_log + noise.pow(2) * 0.5).sum(1)
class ActorDiscretePPO(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                 nn.Linear(mid_dim, action_dim))
        self.soft_max = nn.Softmax(dim=-1)
        self.Categorical = torch.distributions.Categorical
    def forward(self, state):
        return self.net(state)
    def get_action(self, state):
        a_prob = self.soft_max(self.net(state))
        action = torch.multinomial(a_prob, 1, True).reshape(state.size(0))
        return action, a_prob
    def get_new_logprob_entropy(self, state, action):
        a_prob = self.soft_max(self.net(state))
        dist = self.Categorical(a_prob)
        return dist.log_prob(action.squeeze(1).long()), dist.entropy().mean()
    def get_old_logprob(self, action, a_prob):
        dist = self.Categorical(a_prob)
        return dist.log_prob(action.long().squeeze(1))
class Critic(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, 1))
    def forward(self, state, action):
        return self.net(torch.cat((state, action), dim=1))
class CriticAdv(nn.Module):
    def __init__(self, mid_dim, state_dim, _action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, 1))
    def forward(self, state):
        return self.net(state)
class CriticTwin(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net_sa = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, mid_dim), nn.ReLU())
        self.net_q1 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                    nn.Linear(mid_dim, 1))
        self.net_q2 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                    nn.Linear(mid_dim, 1))
    def forward(self, state, action):
        tmp = self.net_sa(torch.cat((state, action), dim=1))
        return self.net_q1(tmp)
    def get_q1_q2(self, state, action):
        tmp = self.net_sa(torch.cat((state, action), dim=1))
        return self.net_q1(tmp), self.net_q2(tmp)
