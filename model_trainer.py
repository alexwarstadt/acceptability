import classifier_data_utils as cdu
import time
from classifier_utils import *
from torch.autograd import Variable


LOGS = open("logs/rnn-logs", "a")
OUTPUT_PATH = "models/rnn_classifier"




class ModelTrainer(object):
    def __init__(self, corpus_path, embedding_path, vocab_path, embedding_size, model, stages_per_epoch, prints_per_stage,
                 convergence_threshold, max_epochs, gpu, learning_rate=.01):
        self.model = model
        self.corpus_path = corpus_path
        self.embedding_size = embedding_size
        self.stages_per_epoch = stages_per_epoch
        self.prints_per_stage = prints_per_stage
        self.convergence_threshold = convergence_threshold
        self.max_epochs = max_epochs
        self.gpu = gpu
        if self.gpu:
            self.model = self.model.cuda()
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)  # or use RMSProp
        self.dm = cdu.DataManagerInMemory(corpus_path, embedding_path,
                            vocab_path, 300, crop_pad_length=30)
        self.loss = torch.nn.BCELoss()
        now = time.localtime()
        time_stamp = str(now.tm_mon) + "-" + str(now.tm_mday) + "_" \
                     + str(now.tm_hour) + ":" + str(now.tm_min) + ":" + str(now.tm_sec)
        self.output_path = OUTPUT_PATH + "_" + time_stamp

    def to_string(self):
        return "data\t\t\t" + self.corpus_path + "\n" + \
            "input size\t\t" + str(self.embedding_size) + "\n" + \
            "hidden size\t\t" + str(self.model.hidden_size) + "\n" + \
            "learning rate\t" + str(self.learning_rate) + "\n" + \
            "output\t\t\t" + str(self.output_path)


    def get_batch_output(self, batch):
        hidden = self.model.init_hidden(batch.batch_size)
        input = torch.Tensor(len(batch.tensor_view), batch.batch_size, self.embedding_size)
        if self.gpu:
            hidden = hidden.cuda()
            input = input.cuda()
        for i, t in enumerate(batch.tensor_view):
            input[i] = t
        outputs, hidden = self.model.forward(Variable(input), hidden)
        return outputs, hidden

    def backprop(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_batch_loss(self, outputs, output_targets):
        # self.loss.weight = torch.Tensor([self.dm.corpus_bias for i in range(len(output_targets))])
        loss = self.loss(outputs, Variable(torch.FloatTensor(output_targets)).view(-1, 1))
        return loss

    def batch_confusion(self, outputs, output_targets):
        if self.gpu:
            outputs = outputs.cpu()
            output_targets = output_targets.cpu()
        tp, fp, tn, fn = 0, 0, 0, 0
        for out, target in zip(outputs, output_targets):
            if out.data[0] > .5 and target > .5:
                tp += 1
            if out.data[0] > .5 and target < .5:
                fp += 1
            if out.data[0] < .5 and target < .5:
                tn += 1
            if out.data[0] < .5 and target > .5:
                fn += 1
        return Confusion(tp, fp, tn, fn)

    def print_min_and_max(self, outputs, batch):
        max_prob, max_i_sentence = torch.topk(outputs.data, 1, 0)
        min_prob, min_i_sentence = torch.topk(outputs.data * -1, 1, 0)
        max_sentence = batch.sentences_view[max_i_sentence[0][0]]
        min_sentence = batch.sentences_view[min_i_sentence[0][0]]
        print("max:", max_prob[0][0], max_sentence)
        print("min:", min_prob[0][0] * -1, min_sentence)

    def get_metrics(self, outputs, batch):
        targets = batch.targets_view
        if self.gpu:
            targets = targets.cuda()
        loss = self.get_batch_loss(outputs, targets)
        confusion = self.batch_confusion(outputs, targets)
        return loss, confusion

    def run_batch(self, batch, backprop):
        outputs, hidden = self.get_batch_output(batch)
        loss, confusion = self.get_metrics(outputs, batch)
        if backprop:
            self.backprop(loss)
        if self.gpu:
            loss = loss.cpu()
        return outputs, loss.data[0], confusion

    def print_stats(self, loss, confusion):
        print("avg loss\t" + self.my_round(loss))
        print("accuracy\t" + self.my_round(confusion.accuracy()))
        print("matthews\t" + self.my_round(confusion.matthews()))
        print('f1\t\t\t' + self.my_round(confusion.f1()))
        print("tp={0[0]:.4g}, fp={0[1]:.4g}, tn={0[2]:.4g}, fn={0[3]:.4g}".format(confusion.percentages()))

    def logs(self, n_batches, train_avg_loss, valid_avg_loss, t_confusion, v_confusion, model_saved):
        LOGS.write("\t" + str(n_batches) + "\t")
        LOGS.write("\t" + self.my_round(train_avg_loss) + "\t")
        LOGS.write("\t" + self.my_round(valid_avg_loss) + "\t")
        LOGS.write("\t" + self.my_round(t_confusion.matthews()) + "\t")
        LOGS.write("\t" + self.my_round(v_confusion.matthews()) + "\t")
        LOGS.write("\t" + self.my_round(t_confusion.f1()) + "\t")
        LOGS.write("\t" + self.my_round(v_confusion.f1()) + "\t")
        LOGS.write("\t" + str(model_saved) + "\n")
        LOGS.flush()

    @staticmethod
    def my_round(n):
        return "{0:.4g}".format(n)

    def run_stage(self, epoch, backprop, stages_per_epoch, prints_per_stage):
        has_next = True
        n_batches = 0
        stage_batches = int(math.ceil(epoch.n_batches/stages_per_epoch))
        print_batches = int(math.ceil(stage_batches/prints_per_stage))
        print_loss = 0
        print_confusion = Confusion()
        stage_loss = 0
        stage_confusion = Confusion()
        while has_next and n_batches < stage_batches:
            n_batches += 1
            batch, has_next = epoch.get_new_batch()
            _, loss, confusion = self.run_batch(batch, backprop)
            print_loss += loss
            print_confusion.add(confusion)
            stage_loss += loss
            stage_confusion.add(confusion)
            if n_batches % print_batches == 0:
                self.print_stats(print_loss/print_batches, print_confusion)
                print_loss = 0
                print_confusion = Confusion()
        if prints_per_stage > 1:
            self.print_stats(stage_loss/n_batches, stage_confusion)
        return stage_loss/n_batches, stage_confusion


    def run_epoch(self, max_matthews, n_stages_not_converging, n_stages):
        train = cdu.CorpusEpoch(self.dm.training_pairs, self.dm)
        valid = cdu.CorpusEpoch(self.dm.valid_pairs, self.dm)
        for _ in range(self.stages_per_epoch):
            if n_stages_not_converging > self.convergence_threshold:
                raise RuntimeError
            n_stages += 1
            print("-------------training-------------")
            train_loss, train_confusion = self.run_stage(train, True, self.stages_per_epoch, self.prints_per_stage)
            print("-------------validation-------------")
            valid_loss, valid_confusion = self.run_stage(valid, False, self.stages_per_epoch, 1)
            if valid_confusion.matthews() > max_matthews:
                max_matthews = valid_confusion.matthews()
                n_stages_not_converging = 0
                torch.save(self.model.state_dict(), self.output_path)
                print("MODEL SAVED")
                self.logs(n_stages, train_loss, valid_loss, train_confusion, valid_confusion, True)
            else:
                n_stages_not_converging += 1
                self.logs(n_stages, train_loss, valid_loss, train_confusion, valid_confusion, True)
        return max_matthews, n_stages_not_converging, n_stages



    def run(self):
        print("======================================================================")
        print("                              TRAINING")
        print("======================================================================")
        print(self.to_string())
        LOGS.write("\n\n" + self.to_string() + "\n")
        LOGS.write(
            "# batches | train avg loss | valid avg loss | t matthews | v matthews | t f1 | v f1 | model saved\n" +
            "----------|----------------|----------------|------------|------------|------|------|------------\n")
        epoch = 0
        n_stages = 0
        max_matthews = 0
        n_stages_not_converging = 0
        # try:
        while epoch < self.max_epochs:
            epoch += 1
            print("===========================EPOCH %d=============================" % epoch)
            max_matthews, n_stages_not_converging, n_stages = self.run_epoch(max_matthews, n_stages_not_converging, n_stages)
        # finally:
        #     self.model.load_state_dict(torch.load(self.output_path))
        #     print("=====================TEST==================")
        #     test_loss, test_confusion = self.run_stage(cdu.CorpusEpoch(self.dm.test_pairs, self.dm), False, 1, 1)
        #     LOGS.write("accuracy\t" + self.my_round(test_confusion.accuracy()) + "\n")
        #     LOGS.write("matthews\t" + self.my_round(test_confusion.matthews()) + "\n")
        #     LOGS.write('f1\t\t\t' + self.my_round(test_confusion.f1()) + "\n")
        #     LOGS.write("tp={0[0]:.4g}, fp={0[1]:.4g}, tn={0[2]:.4g}, fn={0[3]:.4g}".format(test_confusion.percentages()) + "\n")
        #     LOGS.close()




