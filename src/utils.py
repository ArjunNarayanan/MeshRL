import yaml


def load_yaml_config(config_fn):
    print("\nLOADING CONFIG FILE AT : ", config_fn)
    with open(config_fn, "r") as config_file:
        config = yaml.safe_load(config_file)
    return config
