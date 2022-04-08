import os
import time
import torch
import numpy as np
import numpy.random as rd
from elegantrl2.tutorial.agent import ReplayBuffer
from elegantrl2.tutorial.env import deepcopy_or_rebuild_env
class Arguments:
    def __init__(self, agent=None, env=None, gpu_id=None, if_on_policy=False):
        self.agent = agent
        self.cwd = None
        self.env = env
        self.env_eval = None
        self.gpu_id = gpu_id
        self.rollout_num = 2
        self.num_threads = 8
        '''Arguments for training (off-policy)'''
        self.learning_rate = 1e-4
        self.soft_update_tau = 2 ** -8
        self.gamma = 0.99
        self.reward_scale = 2 ** 0
        if if_on_policy:
            self.net_dim = 2 ** 9
            self.batch_size = self.net_dim
            self.repeat_times = 2 ** 4
            self.target_step = 2 ** 12
            self.max_memo = self.target_step
            self.if_per_or_gae = False
        else:
            self.net_dim = 2 ** 8
            self.batch_size = self.net_dim
            self.target_step = 2 ** 10
            self.repeat_times = 2 ** 0
            self.max_memo = 2 ** 17
            self.if_per_or_gae = False
        '''Arguments for evaluate'''
        self.eval_gap = 2 ** 6
        self.eval_times1 = 2 ** 2
        self.eval_times2 = 2 ** 4
        self.random_seed = 0
        self.break_step = 2 ** 20
        self.if_remove = True
        self.if_allow_break = True
    def init_before_training(self, process_id=0):
        if self.agent is None:
            raise RuntimeError('\n| Why agent=None? Assignment args.agent = AgentXXX please.')
        if not hasattr(self.agent, 'init'):
            raise RuntimeError('\n| Should be agent=AgentXXX() instead of agent=AgentXXX')
        if self.env is None:
            raise RuntimeError('\n| Why env=None? Assignment args.env = XxxEnv() please.')
        if isinstance(self.env, str) or not hasattr(self.env, 'env_name'):
            raise RuntimeError('\n| What is env.env_name? use env=PreprocessEnv(env). It is a Wrapper.')
        '''set None value automatically'''
        if self.gpu_id is None:
            self.gpu_id = '0'
        if self.cwd is None:
            agent_name = self.agent.__class__.__name__
            self.cwd = f'./{self.env.env_name}_{agent_name}'
        if process_id == 0:
            print(f'| GPU id: {self.gpu_id}, cwd: {self.cwd}')
            import shutil
            if self.if_remove is None:
                self.if_remove = bool(input("PRESS 'y' to REMOVE: {}? ".format(self.cwd)) == 'y')
            if self.if_remove:
                shutil.rmtree(self.cwd, ignore_errors=True)
                print("| Remove history")
            os.makedirs(self.cwd, exist_ok=True)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.set_num_threads(self.num_threads)
        torch.set_default_dtype(torch.float32)
        gpu_id = self.gpu_id[process_id] if isinstance(self.gpu_id, list) else self.gpu_id
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
def train_and_evaluate(args):
    args.init_before_training()
    '''basic arguments'''
    cwd = args.cwd
    env = args.env
    agent = args.agent
    gpu_id = args.gpu_id
    '''training arguments'''
    net_dim = args.net_dim
    max_memo = args.max_memo
    break_step = args.break_step
    batch_size = args.batch_size
    target_step = args.target_step
    repeat_times = args.repeat_times
    learning_rate = args.learning_rate
    if_break_early = args.if_allow_break
    gamma = args.gamma
    reward_scale = args.reward_scale
    soft_update_tau = args.soft_update_tau
    '''evaluating arguments'''
    show_gap = args.eval_gap
    eval_times1 = args.eval_times1
    eval_times2 = args.eval_times2
    env_eval = deepcopy_or_rebuild_env(env) if args.env_eval is None else args.env_eval
    del args
    '''init: environment'''
    max_step = env.max_step
    state_dim = env.state_dim
    action_dim = env.action_dim
    if_discrete = env.if_discrete
    '''init: Agent, ReplayBuffer, Evaluator'''
    agent.init(net_dim, state_dim, action_dim, learning_rate)
    if_on_policy = agent.if_on_policy
    '''init: ReplayBuffer'''
    agent.state = env.reset()
    if if_on_policy:
        buffer = ReplayBuffer(max_len=target_step + max_step, if_on_policy=if_on_policy,
                              state_dim=state_dim, action_dim=action_dim, if_discrete=if_discrete, )
        steps = 0
    else:
        buffer = ReplayBuffer(max_len=max_memo, if_on_policy=if_on_policy,
                              state_dim=state_dim, action_dim=action_dim, if_discrete=if_discrete, )
        with torch.no_grad():
            trajectory_list = explore_before_training(env, target_step, reward_scale, gamma)
        steps = len(trajectory_list)
        buffer.extend_buffer_from_list(trajectory_list)
        agent.update_net(buffer, target_step, batch_size, repeat_times)
        agent.act_target.load_state_dict(agent.act.state_dict()) if getattr(agent, 'act_target', None) else None
        agent.cri_target.load_state_dict(agent.cri.state_dict()) if getattr(agent, 'cri_target', None) else None
    total_step = steps
    '''init: Evaluator'''
    evaluator = Evaluator(cwd=cwd, agent_id=gpu_id, device=agent.device, env=env_eval,
                          eval_times1=eval_times1, eval_times2=eval_times2, eval_gap=show_gap)
    '''start training'''
    if_train = True
    while if_train:
        with torch.no_grad():
            trajectory_list = agent.explore_env(env, target_step, reward_scale, gamma)
        steps = len(trajectory_list)
        total_step += steps
        buffer.extend_buffer_from_list(trajectory_list)
        logging_tuple = agent.update_net(buffer, batch_size, repeat_times, soft_update_tau)
        with torch.no_grad():
            if_reach_goal = evaluator.evaluate_save(agent.act, steps, logging_tuple)
        if_train = not ((if_break_early and if_reach_goal)
                        or total_step > break_step
                        or os.path.exists(f'{cwd}/stop'))
class Evaluator:
    def __init__(self, cwd, agent_id, eval_times1, eval_times2, eval_gap, env, device):
        self.recorder = list()
        self.r_max = -np.inf
        self.total_step = 0
        self.cwd = cwd
        self.device = device
        self.agent_id = agent_id
        self.eval_gap = eval_gap
        self.eval_times1 = eval_times1
        self.eval_times2 = eval_times2
        self.env = env
        self.target_return = env.target_return
        self.used_time = None
        self.start_time = time.time()
        self.eval_time = -1
        print(f"{'ID':>2} {'Step':>8} {'MaxR':>8} |"
              f"{'avgR':>8} {'stdR':>8} |{'avgS':>5} {'stdS':>4} |"
              f"{'objC':>8} {'etc.':>8}")
    def evaluate_save(self, act, steps, log_tuple) -> bool:
        self.total_step += steps
        if time.time() - self.eval_time > self.eval_gap:
            self.eval_time = time.time()
            rewards_steps_list = [get_episode_return_and_step(self.env, act, self.device) for _ in
                                  range(self.eval_times1)]
            r_avg, r_std, s_avg, s_std = self.get_r_avg_std_s_avg_std(rewards_steps_list)
            if r_avg > self.r_max:
                rewards_steps_list += [get_episode_return_and_step(self.env, act, self.device)
                                       for _ in range(self.eval_times2 - self.eval_times1)]
                r_avg, r_std, s_avg, s_std = self.get_r_avg_std_s_avg_std(rewards_steps_list)
            if r_avg > self.r_max:
                self.r_max = r_avg
                act_save_path = f'{self.cwd}/actor.pth'
                torch.save(act.state_dict(), act_save_path)
                print(f"{self.agent_id:<2} {self.total_step:8.2e} {self.r_max:8.2f} |")
            self.recorder.append((self.total_step, r_avg, r_std, *log_tuple))
            if_reach_goal = bool(self.r_max > self.target_return)
            if if_reach_goal and self.used_time is None:
                self.used_time = int(time.time() - self.start_time)
                print(f"{'ID':>2} {'Step':>8} {'TargetR':>8} |{'avgR':>8} {'stdR':>8} |"
                      f"{self.agent_id:<2} {self.total_step:8.2e} {self.target_return:8.2f} |"
                      f"{r_avg:8.2f} {r_std:8.2f} |"
            print(f"{self.agent_id:<2} {self.total_step:8.2e} {self.r_max:8.2f} |"
                  f"{r_avg:8.2f} {r_std:8.2f} |{s_avg:5.0f} {s_std:4.0f} |"
                  f"{' '.join(f'{n:8.2f}' for n in log_tuple)}")
        else:
            if_reach_goal = False
        return if_reach_goal
    @staticmethod
    def get_r_avg_std_s_avg_std(rewards_steps_list):
        rewards_steps_ary = np.array(rewards_steps_list, dtype=np.float)
        r_avg, s_avg = rewards_steps_ary.mean(axis=0)
        r_std, s_std = rewards_steps_ary.std(axis=0)
        return r_avg, r_std, s_avg, s_std
def get_episode_return_and_step(env, act, device) -> (float, int):
    episode_return = 0.0
    episode_step = 1
    max_step = env.max_step
    if_discrete = env.if_discrete
    state = env.reset()
    for episode_step in range(max_step):
        s_tensor = torch.as_tensor((state,), device=device)
        a_tensor = act(s_tensor)
        if if_discrete:
            a_tensor = a_tensor.argmax(dim=1)
        action = a_tensor.detach().cpu().numpy()[0]
        state, reward, done, _ = env.step(action)
        episode_return += reward
        if done:
            break
    episode_return = getattr(env, 'episode_return', episode_return)
    return episode_return, episode_step
def explore_before_training(env, target_step, reward_scale, gamma) -> list:
    trajectory_list = list()
    if_discrete = env.if_discrete
    action_dim = env.action_dim
    state = env.reset()
    for _ in range(target_step):
        if if_discrete:
            action = rd.randint(action_dim)
            next_s, reward, done, _ = env.step(action)
            other = (reward * reward_scale, 0.0 if done else gamma, action)
        else:
            action = rd.uniform(-1, 1, size=action_dim)
            next_s, reward, done, _ = env.step(action)
            other = (reward * reward_scale, 0.0 if done else gamma, *action)
        trajectory_list.append((state, other))
        state = env.reset() if done else next_s
    return trajectory_list
