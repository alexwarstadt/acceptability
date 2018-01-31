import torch
import os


class Checkpoint:
    def __init__(self, args):
        self.args = args
        self.experiment_ckpt_path = os.path.join(self.args.save_loc,
                                                 self.args.experiment_name)

    def load_state_dict(self, model):
        # TODO: Add logging for checkpoint loaded in both

        # First check if resume arg has been passed
        # TODO: Add resume to parse_args
        if self.args.resume and os.path.exists(self.args.resume):
            model.load_state_dict(torch.load(self.args.resume))

        # Then check if current experiement has a checkpoint
        elif os.path.exists(experiment_ckpt_path):
            model.load_state_dict(torch.load(experiment_ckpt_path))

    def save_state_dict(self, model):
        torch.save(self.experiment_ckpt_path, model.state_dict())
