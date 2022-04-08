import os
import time
from copy import deepcopy
import torch
import torch.nn as nn
import numpy as np
import numpy.random as rd
import gym
class ActorPPO(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                 nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                 nn.Linear(mid_dim, action_dim), )
        self.a_std_log = nn.Parameter(torch.zeros((1, action_dim)) - 0.5, requires_grad=True)
        self.sqrt_2pi_log = 0.9189385332046727
        layer_norm(self.net[-1], std=0.1)
    def forward(self, state):
        return self.net(state).tanh()
    def get_action_noise(self, state):
        a_avg = self.net(state)
        a_std = self.a_std_log.exp()
        noise = torch.randn_like(a_avg)
        action = a_avg + noise * a_std
        return action, noise
    def compute_logprob(self, state, action):
        a_avg = self.net(state)
        a_std = self.a_std_log.exp()
        delta = ((a_avg - action) / a_std).pow(2).__mul__(0.5)
        logprob = -(self.a_std_log + self.sqrt_2pi_log + delta)
        return logprob.sum(1)
class CriticAdv(nn.Module):
    def __init__(self, state_dim, mid_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                 nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                 nn.Linear(mid_dim, 1))
        layer_norm(self.net[-1], std=0.5)
    def forward(self, state):
        return self.net(state)
def layer_norm(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
class AgentPPO:
    def __init__(self):
        super().__init__()
        self.learning_rate = 1e-4
        self.ratio_clip = 0.25
        self.lambda_entropy = 0.01
        self.lambda_gae_adv = 0.98
        self.if_use_gae = True
        self.compute_reward = None
        self.state = None
        self.noise = None
        self.act = self.act_target = None
        self.cri = self.cri_target = None
        self.optimizer = None
        self.criterion = None
        self.device = None
    def init(self, net_dim, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_reward = self.compute_reward_gae if self.if_use_gae else self.compute_reward_adv
        self.act = ActorPPO(net_dim, state_dim, action_dim).to(self.device)
        self.cri = CriticAdv(state_dim, net_dim).to(self.device)
        self.criterion = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam([{'params': self.act.parameters(), 'lr': self.learning_rate},
                                           {'params': self.cri.parameters(), 'lr': self.learning_rate}])
    def select_action(self, state):
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device).detach()
        actions, noises = self.act.get_action_noise(states)
        return actions[0].cpu().numpy(), noises[0].cpu().numpy()
    def store_transition(self, env, buffer, target_step, reward_scale, gamma):
        buffer.empty_buffer_before_explore()
        actual_step = 0
        while actual_step < target_step:
            state = env.reset()
            for _ in range(env.max_step):
                action, noise = self.select_action(state)
                next_state, reward, done, _ = env.step(np.tanh(action))
                actual_step += 1
                other = (reward * reward_scale, 0.0 if done else gamma, *action, *noise)
                buffer.append_buffer(state, other)
                if done:
                    break
                state = next_state
        return actual_step
    def update_net(self, buffer, _target_step, batch_size, repeat_times=8):
        buffer.update_now_len_before_sample()
        max_memo = buffer.now_len
        with torch.no_grad():
            buf_reward, buf_mask, buf_action, buf_noise, buf_state = buffer.sample_all()
            bs = 2 ** 10
            buf_value = torch.cat([self.cri(buf_state[i:i + bs]) for i in range(0, buf_state.size(0), bs)], dim=0)
            buf_logprob = -(buf_noise.pow(2).__mul__(0.5) + self.act.a_std_log + self.act.sqrt_2pi_log).sum(1)
            buf_r_sum, buf_advantage = self.compute_reward(max_memo, buf_reward, buf_mask, buf_value)
            del buf_reward, buf_mask, buf_noise
        obj_critic = None
        for _ in range(int(repeat_times * max_memo / batch_size)):
            indices = torch.randint(max_memo, size=(batch_size,), requires_grad=False, device=self.device)
            state = buf_state[indices]
            action = buf_action[indices]
            r_sum = buf_r_sum[indices]
            logprob = buf_logprob[indices]
            advantage = buf_advantage[indices]
            new_logprob = self.act.compute_logprob(state, action)
            ratio = (new_logprob - logprob).exp()
            obj_surrogate1 = advantage * ratio
            obj_surrogate2 = advantage * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = -torch.min(obj_surrogate1, obj_surrogate2).mean()
            obj_entropy = (new_logprob.exp() * new_logprob).mean()
            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy
            value = self.cri(state).squeeze(1)
            obj_critic = self.criterion(value, r_sum)
            obj_united = obj_actor + obj_critic / (r_sum.std() + 1e-5)
            self.optimizer.zero_grad()
            obj_united.backward()
            self.optimizer.step()
        return self.act.a_std_log.mean().item(), obj_critic.item()
    def compute_reward_adv(self, max_memo, buf_reward, buf_mask, buf_value):
        buf_r_sum = torch.empty(max_memo, dtype=torch.float32, device=self.device)
        pre_r_sum = 0
        for i in range(max_memo - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
        buf_advantage = buf_r_sum - (buf_mask * buf_value.squeeze(1))
        buf_advantage = (buf_advantage - buf_advantage.mean()) / (buf_advantage.std() + 1e-5)
        return buf_r_sum, buf_advantage
    def compute_reward_gae(self, max_memo, buf_reward, buf_mask, buf_value):
        buf_r_sum = torch.empty(max_memo, dtype=torch.float32, device=self.device)
        buf_advantage = torch.empty(max_memo, dtype=torch.float32, device=self.device)
        pre_r_sum = 0
        pre_advantage = 0
        for i in range(max_memo - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
            buf_advantage[i] = buf_reward[i] + buf_mask[i] * pre_advantage - buf_value[i]
            pre_advantage = buf_value[i] + buf_advantage[i] * self.lambda_gae_adv
        buf_advantage = (buf_advantage - buf_advantage.mean()) / (buf_advantage.std() + 1e-5)
        return buf_r_sum, buf_advantage
class ReplayBuffer:
    def __init__(self, max_len, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        self.now_len = 0
        self.next_idx = 0
        self.if_full = False
        self.action_dim = action_dim
        self.if_gpu = False
        other_dim = 1 + 1 + action_dim * 2
        self.buf_other = np.empty((max_len, other_dim), dtype=np.float32)
        self.buf_state = np.empty((max_len, state_dim), dtype=np.float32)
    def append_buffer(self, state, other):
        self.buf_state[self.next_idx] = state
        self.buf_other[self.next_idx] = other
        self.next_idx += 1
        if self.next_idx >= self.max_len:
            self.if_full = True
            self.next_idx = 0
    def extend_buffer(self, state, other):
        size = len(other)
        next_idx = self.next_idx + size
        if next_idx > self.max_len:
            if next_idx > self.max_len:
                self.buf_state[self.next_idx:self.max_len] = state[:self.max_len - self.next_idx]
                self.buf_other[self.next_idx:self.max_len] = other[:self.max_len - self.next_idx]
            self.if_full = True
            next_idx = next_idx - self.max_len
            self.buf_state[0:next_idx] = state[-next_idx:]
            self.buf_other[0:next_idx] = other[-next_idx:]
        else:
            self.buf_state[self.next_idx:next_idx] = state
            self.buf_other[self.next_idx:next_idx] = other
        self.next_idx = next_idx
    def sample_batch(self, batch_size):
        indices = rd.randint(self.now_len - 1, size=batch_size)
        r_m_a = self.buf_other[indices]
        return (r_m_a[:, 0:1],
                r_m_a[:, 1:2],
                r_m_a[:, 2:],
                self.buf_state[indices],
                self.buf_state[indices + 1])
    def sample_for_ppo(self):
        all_other = torch.as_tensor(self.buf_other[:self.now_len], device=self.device)
        return (all_other[:, 0],
                all_other[:, 1],
                all_other[:, 2:2 + self.action_dim],
                all_other[:, 2 + self.action_dim:],
                torch.as_tensor(self.buf_state[:self.now_len], device=self.device))
    def update_now_len_before_sample(self):
        self.now_len = self.max_len if self.if_full else self.next_idx
    def empty_buffer_before_explore(self):
        self.next_idx = 0
        self.now_len = 0
        self.if_full = False
'''Utils'''
class Evaluator:
    def __init__(self, cwd, agent_id, eval_times, show_gap, env, device):
        self.recorder = [(0., -np.inf, 0., 0., 0.), ]
        self.r_max = -np.inf
        self.total_step = 0
        self.cwd = cwd
        self.device = device
        self.agent_id = agent_id
        self.show_gap = show_gap
        self.eva_times = eval_times
        self.env = env
        self.target_reward = env.target_reward
        self.used_time = None
        self.start_time = time.time()
        self.print_time = time.time()
        print(f"{'ID':>2}  {'Step':>8}  {'MaxR':>8} |{'avgR':>8}  {'stdR':>8}   {'objA':>8}  {'objC':>8}")
    def evaluate_save(self, act, steps, obj_a, obj_c):
        reward_list = [get_episode_return(self.env, act, self.device)
                       for _ in range(self.eva_times)]
        r_avg = np.average(reward_list)
        r_std = float(np.std(reward_list))
        if r_avg > self.r_max:
            self.r_max = r_avg
            act_save_path = f'{self.cwd}/actor.pth'
            torch.save(act.state_dict(), act_save_path)
            print(f"{self.agent_id:<2}  {self.total_step:8.2e}  {self.r_max:8.2f} |")
        self.total_step += steps
        self.recorder.append((self.total_step, r_avg, r_std, obj_a, obj_c))
        if_solve = bool(self.r_max > self.target_reward)
        if if_solve and self.used_time is None:
            self.used_time = int(time.time() - self.start_time)
            print(f"{'ID':>2}  {'Step':>8}  {'TargetR':>8} |"
                  f"{self.agent_id:<2}  {self.total_step:8.2e}  {self.target_reward:8.2f} |"
        if time.time() - self.print_time > self.show_gap:
            self.print_time = time.time()
            print(f"{self.agent_id:<2}  {self.total_step:8.2e}  {self.r_max:8.2f} |"
                  f"{r_avg:8.2f}  {r_std:8.2f}   {obj_a:8.2f}  {obj_c:8.2f}")
        return if_solve
def get_episode_return(env, act, device) -> float:
    episode_return = 0.0
    max_step = env.max_step
    if_discrete = env.if_discrete
    state = env.reset()
    for _ in range(max_step):
        s_tensor = torch.as_tensor((state,), device=device)
        a_tensor = act(s_tensor)
        if if_discrete:
            a_tensor = a_tensor.argmax(dim=1)
        action = a_tensor.cpu().numpy()[0]
        state, reward, done, _ = env.step(action)
        episode_return += reward
        if done:
            break
    return env.episode_return if hasattr(env, 'episode_return') else episode_return
'''env.py'''
class PreprocessEnv(gym.Wrapper):
    def __init__(self, env, if_print=True, data_type=np.float32):
        super(PreprocessEnv, self).__init__(env)
        self.env = env
        self.data_type = data_type
        (self.env_name, self.state_dim, self.action_dim, self.action_max,
         self.if_discrete, self.target_reward, self.max_step
         ) = get_gym_env_info(env, if_print)
        self.step = self.step_type
    def reset(self):
        state = self.env.reset()
        return state.astype(self.data_type)
    def step_type(self, action):
        state, reward, done, info = self.env.step(action * self.action_max)
        return state.astype(self.data_type), reward, done, info
def get_gym_env_info(env, if_print):
    import gym
    gym.logger.set_level(40)
    assert isinstance(env, gym.Env)
    env_name = env.unwrapped.spec.id
    state_shape = env.observation_space.shape
    state_dim = state_shape[0] if len(state_shape) == 1 else state_shape
    target_reward = getattr(env, 'target_reward', None)
    target_reward_default = getattr(env.spec, 'reward_threshold', None)
    if target_reward is None:
        target_reward = target_reward_default
    if target_reward is None:
        target_reward = 2 ** 16
    max_step = getattr(env, 'max_step', None)
    max_step_default = getattr(env, '_max_episode_steps', None)
    if max_step is None:
        max_step = max_step_default
    if max_step is None:
        max_step = 2 ** 10
    if_discrete = isinstance(env.action_space, gym.spaces.Discrete)
    if if_discrete:
        action_dim = env.action_space.n
        action_max = int(1)
    elif isinstance(env.action_space, gym.spaces.Box):
        action_dim = env.action_space.shape[0]
        action_max = float(env.action_space.high[0])
    else:
        raise RuntimeError('| Please set these value manually: if_discrete=bool, action_dim=int, action_max=1.0')
    print(f"\n| env_name: {env_name}, action space if_discrete: {if_discrete}"
          f"\n| state_dim: {state_dim}, action_dim: {action_dim}, action_max: {action_max}"
          f"\n| max_step: {max_step} target_reward: {target_reward}") if if_print else None
    return env_name, state_dim, action_dim, action_max, if_discrete, target_reward, max_step
'''DEMO'''
class Arguments:
    def __init__(self, agent=None, env=None, gpu_id=None, if_on_policy=False):
        self.agent = agent
        self.cwd = None
        self.env = env
        self.env_eval = None
        self.gpu_id = gpu_id
        '''Arguments for training (off-policy)'''
        self.net_dim = 2 ** 8
        self.batch_size = 2 ** 8
        self.repeat_times = 2 ** 0
        self.target_step = 2 ** 10
        self.max_memo = 2 ** 17
        if if_on_policy:
            self.net_dim = 2 ** 9
            self.batch_size = 2 ** 8
            self.repeat_times = 2 ** 4
            self.target_step = 2 ** 12
            self.max_memo = self.target_step
        self.reward_scale = 2 ** 0
        self.gamma = 0.99
        self.num_threads = 4
        '''Arguments for evaluate'''
        self.if_remove = True
        self.if_allow_break = True
        self.break_step = 2 ** 20
        self.eval_times = 2 ** 1
        self.show_gap = 2 ** 8
        self.random_seed = 0
    def init_before_training(self):
        self.gpu_id = '0' if self.gpu_id is None else str(self.gpu_id)
        self.cwd = f'./{self.env.env_name}_{self.gpu_id}' if self.cwd is None else self.cwd
        print(f'| GPU id: {self.gpu_id}, cwd: {self.cwd}')
        import shutil
        if self.if_remove is None:
            self.if_remove = bool(input("PRESS 'y' to REMOVE: {}? ".format(self.cwd)) == 'y')
        if self.if_remove:
            shutil.rmtree(self.cwd, ignore_errors=True)
            print("| Remove history")
        os.makedirs(self.cwd, exist_ok=True)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        torch.set_num_threads(self.num_threads)
        torch.set_default_dtype(torch.float32)
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
def train_and_evaluate(args):
    args.init_before_training()
    '''basic arguments'''
    cwd = args.cwd
    env = args.env
    agent = args.agent
    gpu_id = args.gpu_id
    env_eval = args.env_eval
    '''training arguments'''
    net_dim = args.net_dim
    max_memo = args.max_memo
    break_step = args.break_step
    batch_size = args.batch_size
    target_step = args.target_step
    repeat_times = args.repeat_times
    if_break_early = args.if_allow_break
    gamma = args.gamma
    reward_scale = args.reward_scale
    '''evaluating arguments'''
    show_gap = args.show_gap
    eval_times = args.eval_times
    env_eval = deepcopy(env) if env_eval is None else deepcopy(env_eval)
    del args
    '''init: environment'''
    max_step = env.max_step
    state_dim = env.state_dim
    action_dim = env.action_dim
    if_discrete = env.if_discrete
    env_eval = deepcopy(env) if env_eval is None else deepcopy(env_eval)
    '''init: Agent, ReplayBuffer, Evaluator'''
    agent.init(net_dim, state_dim, action_dim)
    buffer = ReplayBuffer(max_len=max_memo + max_step, state_dim=state_dim, action_dim=1 if if_discrete else action_dim)
    evaluator = Evaluator(cwd=cwd, agent_id=gpu_id, device=agent.device, env=env_eval,
                          eval_times=eval_times, show_gap=show_gap)
    '''prepare for training'''
    agent.state = env.reset()
    total_step = 0
    '''start training'''
    if_reach_goal = False
    while not ((if_break_early and if_reach_goal)
               or total_step > break_step
               or os.path.exists(f'{cwd}/stop')):
        with torch.no_grad():
            steps = agent.explore_env(env, buffer, target_step, reward_scale, gamma)
        total_step += steps
        obj_a, obj_c = agent.update_net(buffer, target_step, batch_size, repeat_times)
        with torch.no_grad():
            if_reach_goal = evaluator.evaluate_save(agent.act, steps, obj_a, obj_c)
def demo():
    args = Arguments(if_on_policy=True)
    args.agent = AgentPPO()
    '''choose environment'''
    gym.logger.set_level(40)
    env = gym.make('Pendulum-v0')
    env.target_reward = -200
    args.env = PreprocessEnv(env=env)
    args.reward_scale = 2 ** -3
    args.net_dim = 2 ** 7
    args.batch_size = 2 ** 7
    "TotalStep: 8e5, TargetReward: 200, UsedTime: 1500s"
    "TotalStep: 8e5, TargetReward: 300, UsedTime: 1800s"
    '''train and evaluate'''
    train_and_evaluate(args)
if __name__ == '__main__':
    demo()
