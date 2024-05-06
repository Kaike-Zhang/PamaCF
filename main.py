from meta_config import args
from utls.model_config import *
from utls.trainer import *
from utls.utilize import init_run, restore_stdout_stderr

from datetime import datetime


def main(seed=2024, date_file="2024010100"):
    args.seed = seed
    init_path = f"./log/{args.dataset}/{args.model}/{args.defense}/{date_file}"
    model_path = f"./checkpoints/{args.dataset}/{args.model}/{args.defense}/{date_file}"

    init_run(log_path=init_path, args=args, seed=args.seed)


    glo = globals()
    global_config = vars(args)
    print(global_config)

    global_config["model_config"] = glo[f"get_{global_config['model']}_config"](global_config)
    print(global_config["model"])
    print(global_config["model_config"])

    if global_config['defense'] != "wo_defense":
        global_config["defense_config"] = glo[f"get_{global_config['defense']}_config"](global_config)
        print(global_config["defense"])
        print(global_config["defense_config"])
        trainer =  glo[f"{global_config['model']}{global_config['defense']}Trainer"](global_config)
    else:
        trainer =  glo[f"{global_config['model']}Trainer"](global_config)
    trainer.train(model_path)

    restore_stdout_stderr()

if __name__ == '__main__':
    times = 5
    date_file = datetime.now().strftime('%Y%m%d%H')

    for t in range(times):
        main(seed=2024+t, date_file=date_file)

