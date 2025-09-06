from dataclasses import dataclass

def apply_fn_dict(d: dict, fn) -> dict:
    """ 对字典的每个value递归地应用函数fn """
    new_d = {}
    if hasattr(d, '__dict__'):
        d = vars(d)
    for k, v in d.items():
        if isinstance(v, dict) or hasattr(v, '__dict__'):
            new_d[k] = apply_fn_dict(v, fn)
        else:
            new_d[k] = fn(v)
    return new_d

def flatten_dict(d, prefix="") -> dict:
    """ 将dict或tyro解析的配置类, 展平为dict """
    flatten = {}
    if hasattr(d, '__dict__'):
        d = vars(d)
    for k, v in d.items():
        if isinstance(v, dict) or hasattr(v, '__dict__'):
            flatten.update(flatten_dict(v, prefix+'.'+k if len(prefix) else k))
        else:
            flatten[prefix+'.'+k if len(prefix) else k] = v
    return flatten

def cvt_string_time(t_sec) -> str:
    """ 将时间t_sec转为 hour:minute:second """
    t_sec = int(t_sec)  # 取整秒
    hours = t_sec // 3600
    minutes = (t_sec % 3600) // 60
    seconds = t_sec % 60
    if hours > 0:
        return f"{hours:02}:{minutes:02}:{seconds:02}"
    return f"{minutes:02}:{seconds:02}"

if __name__ == '__main__':
    """ debug flatten_dict """
    # import tyro
    # from katarl2.agents.sac.sac_cfg import SACConfig
    # @dataclass
    # class Args:
    #     a: SACConfig
    #     b: SACConfig

    # cfg = tyro.cli(Args)
    # print(cfg)
    # print(flatten_dict(cfg))

    print(cvt_string_time(3601))
    print(cvt_string_time(200))
