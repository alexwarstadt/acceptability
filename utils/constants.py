START = "<s>"
STOP = "</s>"
NUMBER = "N"
UNK = "<unk>"
special_words = [START, STOP, NUMBER, UNK]


# @staticmethod
# def log_prob_to_prob(output):
#     prob_matrix = torch.zeros(output.size())
#     for i in range(output.size()[0]):
#         prob_matrix[i] = np.exp(output.data[i])
#     return prob_matrix
#
#
# @staticmethod
# def perplexity(N, prob_sum):
#     return math.exp((-1 * prob_sum) / N)
#
#
# @staticmethod
# def sample(weights, prob_sum):
#     threshold = random.uniform(0, prob_sum)
#     i = 0
#     while threshold > 0 and i < len(weights):
#         threshold -= weights[i]
#         i += 1
#     return i - 1, weights[i - 1]