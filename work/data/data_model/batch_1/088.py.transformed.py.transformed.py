from elegantrl2.demo import *
from envs.FinRL.StockTrading import *
def demo_custom_env_finance_rl():
    from elegantrl2.agent import AgentPPO
    '''choose an DRL algorithm'''
    args = Arguments(if_on_policy=True)
    args.agent = AgentPPO()
    args.agent.lambda_entropy = 0.02
    args.gpu_id = (3, 4)
    "TotalStep: 10e4, TargetReturn: 3.0, UsedTime:  200s, FinanceStock-v1"
    args.gamma = 0.9995
    args.env = StockTradingEnv(if_eval=False, gamma=args.gamma)
    args.net_dim = int(2 ** 8 * 1.5)
    args.batch_size = args.net_dim * 4
    args.target_step = args.env.max_step
    args.repeat_times = 2 ** 4
    args.eval_gap = 2 ** 8
    args.eval_times1 = 2 ** 0
    args.eval_times2 = 2 ** 1
    args.break_step = int(8e6)
    '''train and evaluate'''
    args.worker_num = 2
    train_and_evaluate_mg(args)
demo_custom_env_finance_rl()
