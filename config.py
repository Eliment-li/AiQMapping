import os

from munch import Munch
import torch
import yaml

from utils.file.file_util import get_root_dir


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class ConfigSingleton(metaclass=Singleton):
    def __init__(self):
        self.config = None
        self.config_private = None
        self.load_config()

    def load_config(self):
        rootdir = get_root_dir()
        # load public config
        with open(rootdir+os.path.sep+'config.yml', 'r') as file:
            config = yaml.safe_load(file)
            config['device'] = "cuda" if torch.cuda.is_available() else "cpu"
            self.config = Munch(config)

        with open(rootdir+os.path.sep+'config_private.yml', 'r') as file:
            config_private = yaml.safe_load(file)
            self.config_private = Munch(config_private)
        # load privateconfig

    def get_config(self):
        return self.config
    def get_config_private(self):
        return self.config_private

if __name__ == '__main__':
    args = ConfigSingleton().get_config()
    args_pri = ConfigSingleton().get_config_private()
    print(args_pri.tianyan_token)
