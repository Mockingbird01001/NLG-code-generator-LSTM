from elegantrl2.env import *
from elegantrl2.run import *
def demo_continuous_action_on_policy_temp_mg():
    args = Arguments(if_on_policy=True)
    from elegantrl2.agent import AgentPPO
    args.agent = AgentPPO()
    args.agent.cri_target = True
    args.learning_rate = 2 ** -14
    args.random_seed = 19435
    args.gpu_id = (0, 1)
    '''choose environment'''
    if_train_pendulum = 1
    if if_train_pendulum:
        env = gym.make('Pendulum-v0')
        env.target_return = -200
        env.target_return = -1150
        args.env = PreprocessEnv(env=env)
        args.reward_scale = 2 ** -3
        args.net_dim = 2 ** 7
        args.batch_size = args.net_dim * 2
        args.target_step = args.env.max_step * 16
    if_train_bipedal_walker = 0
    if if_train_bipedal_walker:
        env_name = 'BipedalWalker-v3'
        env = gym.make(env_name)
        args.env = PreprocessVecEnv(env=env, env_num=2)
        args.env_eval = PreprocessEnv(env=env_name)
        args.reward_scale = 2 ** 0
        args.gamma = 0.97
        args.target_step = args.env.max_step * 4
        args.repeat_times = 2 ** 4
        args.if_per_or_gae = True
        args.agent.lambda_entropy = 0.04
        args.break_step = int(8e6)
    if_train_finance_rl = 0
    if if_train_finance_rl:
        from envs.FinRL.StockTrading import StockTradingEnv, StockTradingVecEnv
        args.env = StockTradingVecEnv(if_eval=False, gamma=args.gamma, env_num=2)
        args.env_eval = StockTradingEnv(if_eval=True, gamma=args.gamma)
        args.agent.cri_target = True
        args.learning_rate = 2 ** -14
        args.random_seed = 19435
        args.net_dim = int(2 ** 8 * 1.5)
        args.batch_size = args.net_dim * 4
        args.target_step = args.env.max_step
        args.repeat_times = 2 ** 4
        args.eval_gap = 2 ** 8
        args.eval_times1 = 2 ** 0
        args.eval_times2 = 2 ** 1
        args.break_step = int(8e6)
        args.if_allow_break = False
    '''train and evaluate'''
    args.worker_num = 2
    if isinstance(args.gpu_id, int) or isinstance(args.gpu_id, str):
        train_and_evaluate_mp(args)
    elif isinstance(args.gpu_id, tuple) or isinstance(args.gpu_id, list):
        train_and_evaluate_mg(args)
    else:
        print(f"Error in args.gpu_id {args.gpu_id}, type {type(args.gpu_id)}")
if __name__ == '__main__':
    demo_continuous_action_on_policy_temp_mg()
