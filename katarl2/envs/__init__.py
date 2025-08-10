from katarl2.envs.env_cfg import EnvConfig

def get_env_name(cfg: EnvConfig) -> str:
    """ eg: 'Hoop-v4__gymnasium' """
    return f"{cfg.env_name}__{cfg.env_type}"
