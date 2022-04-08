from eRL.run import Arguments, train_and_evaluate, train_and_evaluate__multiprocessing
from eRL.env import decorate_env
import eRL.agent as agent
import gym
gym.logger.set_level(40)
def demo1__discrete_action_space():
    args = Arguments(agent_rl=None, env=None, gpu_id=None)
    args.agent_rl = agent.AgentD3QN
    args.env = decorate_env(env=gym.make('CartPole-v0'))
    args.net_dim = 2 ** 7
    train_and_evaluate(args)
def demo2():
    if_on_policy = False
    args = Arguments(if_on_policy=if_on_policy)
    if if_on_policy:
        args.agent_rl = agent.AgentGaePPO
    else:
        args.agent_rl = agent.AgentModSAC
    env = gym.make('Pendulum-v0')
    env.target_reward = -200
    args.env = decorate_env(env=env)
    args.net_dim = 2 ** 7
    train_and_evaluate(args)
def demo3():
    args = Arguments(if_on_policy=True)
    args.agent_rl = agent.AgentGaePPO
    from eRL.env import FinanceMultiStockEnv
    args.env = FinanceMultiStockEnv(if_train=True)
    args.env_eval = FinanceMultiStockEnv(if_train=False)
    args.break_step = int(5e6)
    args.net_dim = 2 ** 8
    args.max_step = args.env.max_step
    args.max_memo = (args.max_step - 1) * 8
    args.batch_size = 2 ** 11
    args.repeat_times = 2 ** 4
    args.eval_times1 = 2 ** 3
    args.rollout_num = 8
    args.if_break_early = False
    train_and_evaluate__multiprocessing(args)
def demo41():
    args = Arguments(if_on_policy=True)
    args.agent_rl = agent.AgentGaePPO
    import pybullet_envs
    dir(pybullet_envs)
    args.env = decorate_env(gym.make('ReacherBulletEnv-v0'))
    args.break_step = int(5e4 * 8)
    args.repeat_times = 2 ** 3
    args.reward_scale = 2 ** 1
    args.eval_times1 = 2 ** 2
    args.eval_times1 = 2 ** 6
    args.rollout_num = 4
    train_and_evaluate__multiprocessing(args)
def demo42():
    args = Arguments(if_on_policy=True)
    args.agent_rl = agent.AgentGaePPO
    import pybullet_envs
    dir(pybullet_envs)
    args.env = decorate_env(gym.make('AntBulletEnv-v0'))
    args.break_step = int(5e6 * 8)
    args.reward_scale = 2 ** -3
    args.repeat_times = 2 ** 4
    args.net_dim = 2 ** 9
    args.batch_size = 2 ** 8
    args.max_memo = 2 ** 12
    args.show_gap = 2 ** 6
    args.eval_times1 = 2 ** 2
    args.rollout_num = 4
    train_and_evaluate__multiprocessing(args)
def demo5():
    args = Arguments(if_on_policy=False)
    args.agent_rl = agent.AgentSharedSAC
    import pybullet_envs
    dir(pybullet_envs)
    args.env = decorate_env(gym.make('AntBulletEnv-v0'))
    args.break_step = int(1e6 * 8)
    args.reward_scale = 2 ** -2
    args.max_memo = 2 ** 19
    args.net_dim = 2 ** 7
    args.eva_size = 2 ** 5
    args.show_gap = 2 ** 8
    train_and_evaluate(args)
def render_pybullet():
    from eRL import agent
    args = Arguments(if_on_policy=True)
    args.agent_rl = agent.AgentGaePPO
    import pybullet_envs
    dir(pybullet_envs)
    args.env = decorate_env(gym.make('ReacherBulletEnv-v0'))
    args.gpu_id = 3
    args.init_before_training()
    net_dim = args.net_dim
    state_dim = args.env.state_dim
    action_dim = args.env.action_dim
    cwd = "ReacherBulletEnv-v0_2"
    env = args.env
    max_step = args.max_step
    agent = args.agent_rl(net_dim, state_dim, action_dim)
    agent.save_or_load_model(cwd, if_save=False)
    import cv2
    import os
    frame_save_dir = f'{cwd}/frame'
    os.makedirs(frame_save_dir, exist_ok=True)
    '''methods 1: Print Error in remote server: Invalid GLX version: major 1, minor 2'''
    '''methods 2: Print Error in remote server: Invalid GLX version: major 1, minor 2'''
    env.render()
    state = env.reset()
    for i in range(max_step):
        print(i)
        actions, _noises = agent.select_actions((state,))
        action = actions[0]
        next_state, reward, done, _ = env.step(action)
        frame = env.render()
        if isinstance(frame, np.ndarray):
            cv2.imshow(env.env_name, frame)
            cv2.waitKey(20)
            cv2.imwrite(f'{frame_save_dir}/{i:06}.png', frame)
        if done:
            break
        state = next_state
    print('end')
render_pybullet()
