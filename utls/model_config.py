def get_MF_config(org_config):
    config = {
        "device": org_config["device"],
        "dim": int(org_config["out_dim"])
        }
    return config


def get_LightGCN_config(org_config):
    config = {
        "device": org_config["device"],
        "dim": int(org_config["out_dim"]),
        "n_layers": 3,
        }
    return config


def get_APR_config(org_config):
    config = {
        "adv_reg": 1.0,
        "eps": org_config["eps"],
        "begin_adv": 5,
        }
    return config


def get_PamaCF_config(org_config):
    config = {
        "adv_reg": org_config["adv_reg"],
        "eps": org_config["eps"],
        "begin_adv": 5,
        "user_lmb": org_config["user_lmb"],
        }
    return config