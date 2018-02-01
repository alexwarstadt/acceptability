import torch
import os
from utils.flags import get_parser
from utils.general import get_model_instance
from modules.dataset import get_datasets, get_iter
from utils.checkpoint import Checkpoint

# TODO: Add __init__ for all modules and then __all__ in all of them
# to faciliate easy loading


class Trainer:
    def __init__(self):
        parser = get_parser()
        self.args = parser.parse_args()
        self.args.gpu = self.args.gpu and torch.cuda.is_available()
        self.checkpoint = Checkpoint(self.args)

        self.load_datasets()

    def load_datasets(self):
        self.train_dataset, self.val_dataset, self.test_dataset, \
            sentence_field = get_datasets(self.args)

        self.train_loader = get_iter(self.args, self.train_dataset)
        self.val_loader = get_iter(self.args, self.val_dataset)
        self.test_loader = get_iter(self.args, self.test_dataset)

        vocab = sentence_field.vocab
        self.embedding = nn.Embedding(len(vocab), len(vocab.vectors[0]))
        self.embedding.weight.data.copy_(vocab.vectors)
        self.embedding.weight.requires_grad = False

    def load(self):
        self.model = get_model_instance(self.args)

        if self.model is None:
            # TODO: Add logger statement for valid model here
            os.exit(1)
        self.checkpoint.load_state_dict(self.model)

        if self.args.gpu:
            self.model = self.model.cuda()

        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                                 self.model.parameters()),
                                          lr=self.args.learning_rate)
        self.criterion = torch.nn.BCELoss()

    def train(self):
        for i in range(self.args.epochs):
            for idx, data in enumerate(self.train_loader):
                x, y = data.sentence, data.label

                self.optimizer.zero_grad()

                output = self.model(x)
                loss = self.criterion(output, y.long())
                loss.backward()

                self.optimizer.step()

                if idx % self.args.stages_per_epoch == 0:
                    # TODO: Validate here
                    self.validate(self.valid_loader)

            if i % 10 == 0:
                # At the some interval validate train loader
                self.validate(self.train_loader)

    def validate(self, loader):
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0

        for data in loader:
            x, y = data.sentence, data.label

            output = self.model(x)
            loss = nn.function.cross_entropy(output, y.long(),
                                             size_average=False)
            total_loss = loss.data[0]
            total += len(y)

            if not self.args.gpu:
                correct += (y ==
                            output.max(1)[1]).data.cpu().numpy().sum()
            else:
                correct += (y == output.max(1)[1]).data.sum()
        self.model.train()

        avg_loss = total_loss / total

        return correct / total * 100, avg_loss
