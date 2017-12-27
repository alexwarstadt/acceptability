import math
import torch


class Confusion:

    def __init__(self, tp=0, fp=0, tn=0, fn=0):
        self.tp = tp
        self.fp = fp
        self.tn = tn
        self.fn = fn

    def add(self, other):
        self.tp += other.tp
        self.fp += other.fp
        self.tn += other.tn
        self.fn += other.fn

    def f1(self):
        try:
            f = float(2 * self.tp) / float((2 * self.tp) + self.fp + self.fn)
        except ZeroDivisionError:
            f = 0
        return f

    def matthews(self):
        """tp*tn - fp*fn / sqrt( tp+fp tp+fn tn+fp tn+fn )"""
        try:
            m = float((self.tp * self.tn) - (self.fp * self.fn)) / \
                math.sqrt(float((self.tp + self.fp) * (self.tp + self.fn) * (self.tn + self.fp) * (self.tn + self.fn)))
        except ZeroDivisionError:
            m = 0
        return m

    def percentages(self):
        total = float(self.tp + self.fp + self.tn + self.fn)
        if total == 0:
            total = 1
        return float(self.tp)/total, float(self.tn)/total, float(self.fp)/total, float(self.fn)/total

    def accuracy(self):
        return float(self.tp + self.tn) / float(self.tp + self.fp + self.tn + self.fn)


def print_min_and_max(outputs, batch):
    max_prob, max_i_sentence = torch.topk(outputs.data, 1, 0)
    min_prob, min_i_sentence = torch.topk(outputs.data * -1, 1, 0)
    max_sentence = batch.sentences_view[max_i_sentence[0][0]]
    min_sentence = batch.sentences_view[min_i_sentence[0][0]]
    print("max:", max_prob[0][0], max_sentence)
    print("min:", min_prob[0][0] * -1, min_sentence)

    
def f1(tp, fp, tn, fn):
    return 2 * tp / ((2 * tp) + fp + fn)


def matthews(tp, fp, tn, fn):
    """tp*tn - fp*fn / sqrt( tp+fp tp+fn tn+fp tn+fn )"""
    try:
        m = ((tp * tn) - (fp * fn)) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    except ZeroDivisionError:
        m = 0
    return m

def percentages(tp, fp, tn, fn):
    total = tp + fp + tn + fn
    return tp/total, fp/total, tn/total, fn/total