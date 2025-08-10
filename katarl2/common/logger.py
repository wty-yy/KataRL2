from typing import Optional
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter
from katarl2.common.utils import flatten_dict
from katarl2.common.path_manager import path_manager

@dataclass
class LogConfig:
    # upload log data to wandb
    use_wandb: bool = False
    # wandb's project name
    wandb_project_name: str = 'KataRL2'
    # wandb's entity name
    wandb_entity: Optional[str] = None

def get_tensorboard_writer(cfg: LogConfig, args: Optional[dict]=None) -> SummaryWriter:

    if cfg.use_wandb:
        import wandb

        wandb.init(
            dir=str(path_manager.PATH_LOGS),
            project=cfg.wandb_project_name,
            entity=cfg.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=path_manager.RUN_NAME,
        )

    PATH_TB_LOGS = path_manager.PATH_LOGS / "tb"
    writer = SummaryWriter(str(PATH_TB_LOGS))
    if args is not None:
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in flatten_dict(args).items()])),
        )

    return writer
