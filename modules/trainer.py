import torch
import os
from utils.flags import get_parser
from utils.general import get_model_instance
from utils.checkpoint import Checkpoint

# TODO: Add __init__ for all modules and then __all__ in all of them
# to faciliate easy loading


class Trainer:
    def __init__(self):
        parser = get_parser()
        self.args = parser.parse_args()
        self.args.gpu = self.args.gpu and torch.cuda.is_available()
        self.checkpoint = Checkpoint(self.args)

    def load(self):
        self.model = get_model_instance(self.args)

        if self.model is None:
            # TODO: Add logger statement for valid model here
            os.exit(1)
        self.checkpoint.load_state_dict(self.model)

    def train(self):
        return
