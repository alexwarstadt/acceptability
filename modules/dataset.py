import os
import nltk
from torch.utils.data import Dataset
from torchtext import vocab, data


class AcceptabilityDataset(Dataset):
    def __init__(self, path, vocab_name):
        self.pairs = []
        self.sentences = []
        if not os.exists(path):
            # TODO: log failure here
            os.exit(1)

        self.vocab = vocab.pretrained_aliases[vocab_name]
        with open(path, 'r') as f:
            for line in f:
                line = line.split("\t")

                if len(line) >= 4:
                    self.sentences.append(line[3].strip())
                    self.pairs.append((line[3].strip(), line[1], line[0]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        return self.pairs[index]


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def get_datasets(args):
    sentence = data.Field(
        sequential=True,
        fix_length=args.crop_pad_length,
        tokenize=tokenizer,
        tensor_type=torch.cuda.LongTensor if args.gpu else torch.LongTensor,
        lower=True
    )

    train_dataset, val_dataset, test_dataset = data.TabularDataset.splits(
        path=args.data_dir,
        train="train",
        validation="valid",
        test="test",
        format="tsv",
        fields=[
            ('source', None),
            ('label', data.Field(
                use_vocab=False,
                sequential=False,
                tensor_type=torch.cuda.LongTensor if args.gpu
                else torch.LongTensor
            )),
            ('mark', None),
            ('sentence', sentence)
        ]
    )

    sentence.build_vocab(
        train_dataset,
        vectors=args.embedding
    )

    return train_dataset, val_dataset, test_dataset, sentence


def get_iter(args, dataset):
    return data.Iterator(
        dataset,
        batch_size=args.batch_size,
        device=0 if args.gpu else -1
    )
