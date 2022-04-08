import time
import numpy as np
import numpy.random as rd
import gym
import torch
import torch.nn as nn
class EvaluateRewardSV:
    def __init__(self, env):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def get_eva_reward__sv(self, act, max_step, action_max, is_discrete, is_render=False):
        reward_sum = 0
        state = self.env.reset()
        for _ in range(max_step):
            states = torch.tensor((state,), dtype=torch.float32, device=self.device)
            actions = act(states)
            if is_discrete:
                actions = actions.argmax(dim=1)
            action = actions.cpu().data.numpy()[0]
            next_state, reward, done, _ = self.env.step(action * action_max)
            reward_sum += reward
            if is_render:
                self.env.render()
            if done:
                break
            state = next_state
        return reward_sum
class QNet(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, action_dim), )
    def forward(self, s):
        q = self.net(s)
        return q
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, action_dim), nn.Tanh(), )
    def forward(self, s):
        a = self.net(s)
        return a
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, 1), )
    def forward(self, s, a):
        x = torch.cat((s, a), dim=1)
        q = self.net(x)
        return q
class BufferList:
    def __init__(self, memo_max_len):
        self.memories = list()
        self.max_len = memo_max_len
        self.now_len = len(self.memories)
    def add_memo(self, memory_tuple):
        self.memories.append(memory_tuple)
    def init_before_sample(self):
        del_len = len(self.memories) - self.max_len
        if del_len > 0:
            del self.memories[:del_len]
        self.now_len = len(self.memories)
    def random_sample(self, batch_size, device):
        indices = rd.randint(self.now_len, size=batch_size)
        '''convert list into array'''
        arrays = [list()
                  for _ in range(5)]
        for index in indices:
            items = self.memories[index]
            for item, array in zip(items, arrays):
                array.append(item)
        '''convert array into torch.tensor'''
        tensors = [torch.tensor(np.array(ary), dtype=torch.float32, device=device)
                   for ary in arrays]
        return tensors
class BufferArray:
    def __init__(self, memo_max_len, state_dim, action_dim, ):
        memo_dim = 1 + 1 + state_dim + action_dim + state_dim
        self.memories = np.empty((memo_max_len, memo_dim), dtype=np.float32)
        self.next_idx = 0
        self.is_full = False
        self.max_len = memo_max_len
        self.now_len = self.max_len if self.is_full else self.next_idx
        self.state_idx = 1 + 1 + state_dim
        self.action_idx = self.state_idx + action_dim
    def add_memo(self, memo_tuple):
        self.memories[self.next_idx] = np.hstack(memo_tuple)
        self.next_idx = self.next_idx + 1
        if self.next_idx >= self.max_len:
            self.is_full = True
            self.next_idx = 0
    def extend_memo(self, memo_array):
        size = memo_array.shape[0]
        next_idx = self.next_idx + size
        if next_idx < self.max_len:
            self.memories[self.next_idx:next_idx] = memo_array
        if next_idx >= self.max_len:
            if next_idx > self.max_len:
                self.memories[self.next_idx:self.max_len] = memo_array[:self.max_len - self.next_idx]
            self.is_full = True
            next_idx = next_idx - self.max_len
            self.memories[0:next_idx] = memo_array[-next_idx:]
        else:
            self.memories[self.next_idx:next_idx] = memo_array
        self.next_idx = next_idx
    def init_before_sample(self):
        self.now_len = self.max_len if self.is_full else self.next_idx
    def random_sample(self, batch_size, device):
        indices = rd.randint(self.now_len, size=batch_size)
        memory = self.memories[indices]
        if device:
            memory = torch.tensor(memory, device=device)
        '''convert array into torch.tensor'''
        tensors = (
            memory[:, 0:1],
            memory[:, 1:2],
            memory[:, 2:self.state_idx],
            memory[:, self.state_idx:self.action_idx],
            memory[:, self.action_idx:],
        )
        return tensors
def soft_target_update(target, online, tau=5e-3):
    for target_param, param in zip(target.parameters(), online.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
def run__tutorial_discrete_action():
    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    state_dim = 4
    action_dim = 2
    action_max = int(1)
    target_reward = 195.0
    is_discrete = True
    ''' I copy the code from AgentDQN to the following for tutorial.'''
    net_dim = 2 ** 7
    learning_rate = 2e-4
    max_buffer = 2 ** 12
    max_epoch = 2 ** 12
    max_step = 2 ** 9
    gamma = 0.99
    batch_size = 2 ** 6
    criterion = torch.nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ''' QNet is an actor or critic? DQN is not a Actor-Critic Method.
    AgentDQN chooses action with the largest q value outputing by Q_Network. Q_Network is an actor.
    AgentDQN outputs q_value by Q_Network. Q_Network is also a critic.
    '''
    act = QNet(state_dim, action_dim, net_dim).to(device)
    act.train()
    act_optim = torch.optim.Adam(act.parameters(), lr=learning_rate)
    act_target = QNet(state_dim, action_dim, net_dim).to(device)
    act_target.load_state_dict(act.state_dict())
    act_target.eval()
    buffer = BufferArray(max_buffer, state_dim, action_dim=1)
    '''training loop'''
    self_state = env.reset()
    self_steps = 0
    self_r_sum = 0.0
    total_step = 0
    evaluator = EvaluateRewardSV(env)
    max_reward = evaluator.get_eva_reward__sv(act, max_step, action_max, is_discrete)
    start_time = time.time()
    for epoch in range(max_epoch):
        explore_rate = 0.1
        rewards = list()
        steps = list()
        for _ in range(max_step):
            if rd.rand() < explore_rate:
                action = rd.randint(action_dim)
            else:
                states = torch.tensor((self_state,), dtype=torch.float32, device=device)
                actions = act_target(states).argmax(dim=1).cpu().data.numpy()
                action = actions[0]
            next_state, reward, done, _ = env.step(action)
            self_r_sum += reward
            self_steps += 1
            mask = 0.0 if done else gamma
            buffer.add_memo((reward, mask, self_state, action, next_state))
            self_state = next_state
            if done:
                rewards.append(self_r_sum)
                self_r_sum = 0.0
                steps.append(self_steps)
                self_steps = 0
                self_state = env.reset()
        total_step += sum(steps)
        avg_reward = np.average(rewards)
        print(end=f'Reward:{avg_reward:6.1f}    Step:{total_step:8}    ')
        '''update_parameters'''
        loss_c_sum = 0.0
        update_times = max_step
        buffer.init_before_sample()
        for _ in range(update_times):
            with torch.no_grad():
                rewards, masks, states, actions, next_states = buffer.random_sample(batch_size, device)
                next_q_target = act_target(next_states).max(dim=1, keepdim=True)[0]
                q_target = rewards + masks * next_q_target
            act.train()
            actions = actions.type(torch.long)
            q_eval = act(states).gather(1, actions)
            critic_loss = criterion(q_eval, q_target)
            loss_c_sum += critic_loss.item()
            act_optim.zero_grad()
            critic_loss.backward()
            act_optim.step()
            soft_target_update(act_target, act, tau=5e-2)
            ''' A small tau can stabilize training in harder env.
            You can change tau into smaller tau 5e-3. But this env is too easy.
            You can try the harder env and other DRL Algorithms in run__xx() in AgentRun.py
            '''
        loss_c_avg = loss_c_sum / update_times
        print(end=f'Loss:{loss_c_avg:6.1f}    ')
        eva_reward_list = [evaluator.get_eva_reward__sv(act, max_step, action_max, is_discrete)
                           for _ in range(3)]
        eva_reward = np.average(eva_reward_list)
        print(f'TrueRewward:{eva_reward:6.1f}')
        if eva_reward > max_reward:
            max_reward = eva_reward
        if max_reward > target_reward:
            print(f"|\tReach target_reward: {max_reward:6.1f} > {target_reward:6.1f}")
            break
    used_time = int(time.time() - start_time)
    print(f"|\tTraining UsedTime: {used_time}s")
    '''open a window and show the env'''
    for _ in range(4):
        eva_reward = evaluator.get_eva_reward__sv(act, max_step, action_max, is_discrete, is_render=True)
        print(f'|Evaluated reward is: {eva_reward}')
def run__tutorial_continuous_action():
    env_name = 'Pendulum-v0'
    env = gym.make(env_name)
    state_dim = 3
    action_dim = 1
    action_max = 2.0
    target_reward = -200.0
    is_discrete = False
    ''' I copy the code from AgentDQN to the following for tutorial.'''
    net_dim = 2 ** 5
    learning_rate = 2e-4
    max_buffer = 2 ** 14
    max_epoch = 2 ** 12
    max_step = 2 ** 8
    gamma = 0.99
    batch_size = 2 ** 7
    update_freq = 2 ** 7
    criterion = torch.nn.SmoothL1Loss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    act_dim = net_dim
    act = Actor(state_dim, action_dim, act_dim).to(device)
    act.train()
    act_optim = torch.optim.Adam(act.parameters(), lr=learning_rate)
    act_target = Actor(state_dim, action_dim, act_dim).to(device)
    act_target.load_state_dict(act.state_dict())
    act_target.eval()
    cri_dim = int(net_dim * 1.25)
    cri = Critic(state_dim, action_dim, cri_dim).to(device)
    cri.train()
    cri_optim = torch.optim.Adam(cri.parameters(), lr=learning_rate)
    cri_target = Critic(state_dim, action_dim, cri_dim).to(device)
    cri_target.load_state_dict(cri.state_dict())
    cri_target.eval()
    from AgentZoo import BufferArray
    buffer = BufferArray(max_buffer, state_dim, action_dim)
    '''training loop'''
    self_state = env.reset()
    self_steps = 0
    self_r_sum = 0.0
    total_step = 0
    explore_noise = 0.05
    evaluator = EvaluateRewardSV(env)
    max_reward = evaluator.get_eva_reward__sv(act, max_step, action_max, is_discrete)
    start_time = time.time()
    while total_step < max_step:
        for _ in range(max_step):
            action = rd.uniform(-1, 1, size=action_dim)
            next_state, reward, done, _ = env.step(action * action_max)
            mask = 0.0 if done else gamma
            buffer.add_memo((reward, mask, self_state, action, next_state))
            total_step += 1
            if done:
                self_state = env.reset()
                break
            self_state = next_state
    for epoch in range(max_epoch):
        explore_rate = 0.5
        reward_list = list()
        step_list = list()
        for _ in range(max_step):
            states = torch.tensor((self_state,), dtype=torch.float32, device=device)
            actions = act_target(states).cpu().data.numpy()
            action = actions[0]
            if rd.rand() < explore_rate:
                action = rd.normal(action, explore_noise).clip(-1, +1)
            next_state, reward, done, _ = env.step(action * action_max)
            self_r_sum += reward
            self_steps += 1
            mask = 0.0 if done else gamma
            buffer.add_memo((reward, mask, self_state, action, next_state))
            self_state = next_state
            if done:
                reward_list.append(self_r_sum)
                self_r_sum = 0.0
                step_list.append(self_steps)
                self_steps = 0
                self_state = env.reset()
        total_step += sum(step_list)
        avg_reward = np.average(reward_list)
        print(end=f'Reward:{avg_reward:8.1f}  Step:{total_step:8}  ')
        '''update_parameters'''
        loss_a_sum = 0.0
        loss_c_sum = 0.0
        update_times = max_step
        buffer.init_before_sample()
        for i in range(update_times):
            for _ in range(2):
                with torch.no_grad():
                    reward, mask, state, action, next_state = buffer.random_sample(batch_size, device)
                    next_action = act_target(next_state)
                    next_q_target = cri_target(next_state, next_action)
                    q_target = reward + mask * next_q_target
                q_eval = cri(state, action)
                critic_loss = criterion(q_eval, q_target)
                loss_c_sum += critic_loss.item()
                cri_optim.zero_grad()
                critic_loss.backward()
                cri_optim.step()
            action_pg = act(state)
            actor_loss = -cri(state, action_pg).mean()
            loss_a_sum += actor_loss.item()
            act_optim.zero_grad()
            actor_loss.backward()
            act_optim.step()
            '''soft target update'''
            '''hard target update'''
            if i % update_freq == 0:
                cri_target.load_state_dict(cri.state_dict())
                act_target.load_state_dict(act.state_dict())
        loss_c_avg = loss_c_sum / (update_times * 2)
        loss_a_avg = loss_a_sum / update_times
        print(end=f'LossC:{loss_c_avg:6.1f}  LossA:{loss_a_avg:6.1f}  ')
        eva_reward_list = [evaluator.get_eva_reward__sv(act, max_step, action_max, is_discrete)
                           for _ in range(3)]
        eva_reward = np.average(eva_reward_list)
        print(f'TrueRewward:{eva_reward:8.1f}')
        if eva_reward > max_reward:
            max_reward = eva_reward
        if max_reward > target_reward:
            print(f"|\tReach target_reward: {max_reward:6.1f} > {target_reward:6.1f}")
            break
    used_time = int(time.time() - start_time)
    print(f"|\tTraining UsedTime: {used_time}s")
    '''open a window and show the env'''
    for _ in range(4):
        eva_reward = evaluator.get_eva_reward__sv(act, max_step, action_max, is_discrete, is_render=True)
        print(f'| Evaluated reward is: {eva_reward}')
if __name__ == '__main__':
    run__tutorial_discrete_action()
