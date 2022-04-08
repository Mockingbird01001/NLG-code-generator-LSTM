import gym
from elegantrl2.tutorial.run import Arguments, train_and_evaluate
from elegantrl2.tutorial.env import PreprocessEnv
def demo_discrete_action_off_policy():
    args = Arguments()
    from elegantrl2.tutorial.agent import AgentDoubleDQN
    args.agent = AgentDoubleDQN()
    '''choose environment'''
    if_train_cart_pole = 0
    if if_train_cart_pole:
        args.env = PreprocessEnv(env='CartPole-v0')
        args.net_dim = 2 ** 7
        args.target_step = args.env.max_step * 2
    if_train_lunar_lander = 1
    if if_train_lunar_lander:
        args.env = PreprocessEnv(env=gym.make('LunarLander-v2'))
        args.net_dim = 2 ** 8
        args.batch_size = args.net_dim
    '''train and evaluate'''
    train_and_evaluate(args)
def demo_discrete_action_on_policy():
    args = Arguments(if_on_policy=True)
    from elegantrl2.tutorial.agent import AgentDiscretePPO
    args.agent = AgentDiscretePPO()
    '''choose environment'''
    if_train_cart_pole = 0
    if if_train_cart_pole:
        args.env = PreprocessEnv(env='CartPole-v0')
        args.net_dim = 2 ** 7
        args.batch_size = args.net_dim * 2
        args.repeat_times = 2 ** 4
        args.target_step = args.env.max_step * 8
        args.if_per_or_gae = True
    if_train_lunar_lander = 1
    if if_train_lunar_lander:
        args.env = PreprocessEnv(env=gym.make('LunarLander-v2'))
        args.agent.cri_target = False
        args.reward_scale = 2 ** -1
        args.net_dim = 2 ** 8
        args.batch_size = args.net_dim * 4
        args.target_step = args.env.max_step * 4
        args.repeat_times = 2 ** 5
        args.if_per_or_gae = True
    '''train and evaluate'''
    train_and_evaluate(args)
def demo_continuous_action_off_policy():
    args = Arguments()
    from elegantrl2.tutorial.agent import AgentSAC
    args.agent = AgentSAC()
    '''choose environment'''
    if_train_pendulum = 1
    if if_train_pendulum:
        env = gym.make('Pendulum-v0')
        env.target_return = -200
        args.env = PreprocessEnv(env=env)
        args.reward_scale = 2 ** -3
        args.net_dim = 2 ** 7
        args.batch_size = args.net_dim
        args.target_step = args.env.max_step * 4
    if_train_lunar_lander = 0
    if if_train_lunar_lander:
        args.env = PreprocessEnv(env=gym.make('LunarLanderContinuous-v2'))
        args.reward_scale = 2 ** 0
    if_train_bipedal_walker = 0
    if if_train_bipedal_walker:
        args.env = PreprocessEnv(env=gym.make('BipedalWalker-v3'))
        args.reward_scale = 2 ** 0
        args.gamma = 0.97
        args.if_per_or_gae = True
    '''train and evaluate'''
    train_and_evaluate(args)
def demo_continuous_action_on_policy():
    args = Arguments(if_on_policy=True)
    from elegantrl2.tutorial.agent import AgentPPO
    args.agent = AgentPPO()
    '''choose environment'''
    if_train_pendulum = 1
    if if_train_pendulum:
        env = gym.make('Pendulum-v0')
        env.target_return = -200
        args.env = PreprocessEnv(env=env)
        args.reward_scale = 2 ** -3
        args.net_dim = 2 ** 7
        args.batch_size = args.net_dim * 2
        args.target_step = args.env.max_step * 16
    if_train_lunar_lander = 0
    if if_train_lunar_lander:
        args.env = PreprocessEnv(env=gym.make('LunarLanderContinuous-v2'))
        args.reward_scale = 2 ** 0
    if_train_bipedal_walker = 0
    if if_train_bipedal_walker:
        args.env = PreprocessEnv(env=gym.make('BipedalWalker-v3'))
        args.reward_scale = 2 ** 0
        args.gamma = 0.97
        args.if_per_or_gae = True
    '''train and evaluate'''
    train_and_evaluate(args)
if __name__ == '__main__':
    demo_discrete_action_off_policy()
