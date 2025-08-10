from dataclasses import dataclass

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

    """ debug apply_func_dict """
    d = {'a': 1, 'b': 3, 'c': {'aa': 2, 'bb': 4}}
    print(apply_func_for_dict(d, lambda x: x + 3))
    print(flatten_dict(d))
