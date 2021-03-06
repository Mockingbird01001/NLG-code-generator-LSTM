from elegantrl2.demo import *
from StockTrading import *
def demo_custom_env_finance_rl():
    from elegantrl2.agent import AgentPPO
    '''choose an DRL algorithm'''
    args = Arguments(if_on_policy=True)
    args.agent = AgentPPO()
    args.agent.lambda_entropy = 0.02
    args.gpu_id = sys.argv[-1][-4]
    args.random_seed = 1943210
    "TotalStep: 10e4, TargetReturn: 3.0, UsedTime:  200s, FinanceStock-v1"
    args.gamma = 0.999
    args.env = StockTradingVecEnv(if_eval=False, gamma=args.gamma, env_num=2)
    args.env_eval = StockTradingEnv(if_eval=True, gamma=args.gamma)
    args.net_dim = 2 ** 9
    args.batch_size = args.net_dim * 4
    args.target_step = args.env.max_step * 2
    args.repeat_times = 2 ** 4
    args.eval_gap = 2 ** 8
    args.eval_times1 = 2 ** 0
    args.eval_times2 = 2 ** 1
    args.break_step = int(16e6)
    '''train and evaluate'''
    args.worker_num = 2
    train_and_evaluate_mp(args)
demo_custom_env_finance_rl()
