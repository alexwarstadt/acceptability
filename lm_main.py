import lm as my_lm
import data_utils as du
import torch
import model
import torch.nn as nn

# model_trainer = my_lm.ModelTrainer(raw_corpus="data/bnc/bnc.txt", embedding_path="embeddings/glove.6B.300d.txt", crop_pad_length=30, unked=False,
#                              embeddings_size=300, dim_hidden=350, n_layers=1,
#                              learning_rate=.0001, model_type="LSTM")

# model_trainer2 = ModelTrainer(raw_corpus="data/bnc/bnc.txt", embedding_path="embeddings/glove.6B.300d.txt", crop_pad_length=30, unked=False,
#                              embeddings_size=300, dim_hidden=300, n_layers=1,
#                              learning_rate=.0001, model_type="LSTM")
#
# for w1, w2 in zip(model_trainer.dm.vocab, model_trainer2.dm.vocab):
#     if model_trainer.dm.word_to_tensor(w1) != model_trainer2.dm.word_to_tensor(w2):
#         print(w1, w2)




def generate(n, out_path, gpu):
    lm = model.MyLSTM(input_size=300, hidden_size=350, output_size=20001, n_layers=1, nonlinearity=nn.Tanh())
    lm.load_state_dict(torch.load('models/model_7-14_15:33:4_110500'))
    dm = du.DataManager('../data/bnc-30', '../data/bnc-30/embeddings_20000.txt',
                        '../data/bnc-30/vocab_20000.txt', 300, crop_pad_length=30)
    mu = my_lm.ModelUtils(dm, gpu)
    out = open(out_path, "w+")
    lines = ""
    for i in range(n):
        lines += ("lm	0	*	<s> " + mu.generate_sans_probability(29, lm) + "</s>\n")
        if i % 1000 == 0 or i == n-1:
            out.write(lines)
            print(i, "lines printed")
            lines = ""
    out.close()

def generate_batch(n, batch, out_path, gpu):
    lm = model.MyLSTM(input_size=300, hidden_size=350, output_size=20001, n_layers=1, nonlinearity=nn.Tanh())
    lm.load_state_dict(torch.load('models/model_7-14_15:33:4_110500'))
    dm = du.DataManager('/scratch/asw462/data/bnc-30', '/scratch/asw462/data/bnc-30/embeddings_20000.txt',
                        '/scratch/asw462/data/bnc-30/vocab_20000.txt', 300, crop_pad_length=30)
    mu = my_lm.ModelUtils(dm, gpu)
    out = open(out_path, "w+")
    lines = []
    for i in range(n):
        lines.extend(mu.generate_batch(29, batch, lm))
        if i % 1 == 0 or i == n - 1:
            lines = ["lm	0	*	<s> " + l + "</s>\n" for l in lines]
            for l in lines:
                out.write(l)
            print(i, "lines printed")
            lines = []
    out.close()

# generate_batch(1000000, 10, "acceptability_corpus/lm_generated5", False)

generate(1000000, "acceptability_corpus/lm_generated6", True)


# model_trainer.run_train()






# N_EXPERIMENTS = 1
# hidden_size_range = (350, 350)
# n_layers_range = (1, 1)
# learning_rate_range = (.0001, .0001)

# for i in range(N_EXPERIMENTS):
#     n_params = math.inf
#     dim_hidden = random.randint(hidden_size_range[0], hidden_size_range[1])
#     n_layers = random.randint(n_layers_range[0], n_layers_range[1])
#     learning_rate = random.uniform(learning_rate_range[0], learning_rate_range[1])
#     model_trainer = ModelTrainer(raw_corpus="data/bnc/bnc.txt", embedding_path="embeddings/glove.6B.300d.txt",
#                                  embeddings_size=300, dim_hidden=dim_hidden, n_layers=n_layers,
#                                  learning_rate=learning_rate, model_type="LSTM")
#     n_params = model_trainer.model.n_params()
#     print("n_params", n_params)
#     model_trainer.run_train()