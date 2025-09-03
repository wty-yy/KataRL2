import yaml
from typing import Optional
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter
from katarl2.common.utils import flatten_dict
from katarl2.common import path_manager

@dataclass
class LogConfig:
    # wandb/swanlab's project name
    project_name: str = 'KataRL2'

    """ WandB """
    # upload tensorbaord log data to wandb
    use_wandb: bool = False
    # wandb's entity name
    wandb_entity: Optional[str] = None

    """ SwanLab """
    # upload tensorboard log data to swanlab
    use_swanlab: bool = False
    # swanlab's workspace
    swanlab_workspace: Optional[str] = None

def get_tensorboard_writer(cfg: LogConfig, args: Optional[dict]=None) -> SummaryWriter:

    if cfg.use_wandb:
        import wandb

        wandb.init(
            dir=str(path_manager.PATH_ROOT),
            project=cfg.project_name,
            entity=cfg.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=path_manager.RUN_NAME,
        )
    
    if cfg.use_swanlab:
        import swanlab
        
        path_swanlab_log = path_manager.PATH_ROOT / "swanlog"
        path_swanlab_log.mkdir(parents=True, exist_ok=True)
        swanlab.init(
            project=cfg.project_name,
            workspace=cfg.swanlab_workspace,
            experiment_name=path_manager.RUN_NAME,
            config=flatten_dict(args),
            logdir=str(path_swanlab_log)
        )
        swanlab.sync_tensorboard_torch()

    PATH_TB_LOGS = path_manager.PATH_LOGS / "tb"
    writer = SummaryWriter(str(PATH_TB_LOGS))
    if args is not None:
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in flatten_dict(args).items()])),
        )

        PATH_CONFIG = path_manager.PATH_LOGS / "config.yaml"
        with open(PATH_CONFIG, "w") as f:
            f.write(yaml.dump(args, indent=2, allow_unicode=True))

    return writer
