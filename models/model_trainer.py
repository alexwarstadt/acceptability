import gflags
from datetime import datetime
from utils.classifier_utils import *
from torch.autograd import Variable
from utils import classifier_data_utils as cdu



class ModelTrainer(object):
    """
    Class overseeing training. 
    Contains model and data, main training loop, runs batches, calculates loss,
    does backprop, does evaluation, writes to logs
    """
    def __init__(self, FLAGS, model):
        """
        :param FLAGS: passed in at training.run_train.py
        :param model: model to be trained
        """
        self.FLAGS = FLAGS
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.FLAGS.learning_rate)  # or use RMSProp
        self.dm = cdu.DataManagerInMemory(self.FLAGS.data_dir, self.FLAGS.embedding_path,
                                          self.FLAGS.vocab_path, self.FLAGS.embedding_size, self.FLAGS.crop_pad_length)
        self.loss = torch.nn.BCELoss()
        if self.FLAGS.gpu:
            self.model = self.model.cuda()
        now = datetime.now()
        # self.time_stamp = "%d-%d_%d-%d-%d_%d" % (now.month, now.day, now.hour, now.minute, now.second, now.microsecond)
        self.OUTPUT_PATH = FLAGS.ckpt_path + FLAGS.experiment_name
        self.LOGS_PATH = FLAGS.log_path + "LOGS-" + FLAGS.experiment_name
        self.OUT_LOGS_PATH = FLAGS.log_path + "OUTPUTS-" + FLAGS.experiment_name
        self.LOGS = open(self.LOGS_PATH, "a")
        self.OUT_LOGS = open(self.OUT_LOGS_PATH, "a")

    def to_string(self):
        return "data\t\t\t" + self.FLAGS.data_dir + "\n" + \
            "input size\t\t" + str(self.FLAGS.embedding_size) + "\n" + \
            "hidden size\t\t" + str(self.model.hidden_size) + "\n" + \
            "learning rate\t" + str(self.FLAGS.learning_rate) + "\n" + \
            "output\t\t\t" + str(self.OUTPUT_PATH)


    def get_batch_output(self, batch):
        """
        calls model.forward(). This method is usually overridden in child classes
        :param batch: object of type utils.classifier_data_utils.Batch
        :return: outputs and hidden state returned by model.forward
        """
        hidden = self.model.init_hidden(batch.batch_size)
        input = torch.FloatTensor(len(batch.tensor_view), batch.batch_size, self.FLAGS.embedding_size)
        if self.FLAGS.gpu:
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
        loss = self.loss(outputs, Variable(output_targets).view(-1, 1))
        return loss

    def batch_confusion(self, outputs, output_targets, sources, by_source):
        """
        get confusion matrix for a batch. true positive / true negative / false positive / false negative
        :param outputs: model outputs
        :param output_targets: list of targets from 
        :param sources: source of each data point
        :param by_source: dict containing confusion matrix for each source
        :return: confusion matrix for batch, updated by_source
        """
        if self.FLAGS.gpu:
            outputs = outputs.cpu()
            output_targets = output_targets.cpu()
        tp, fp, tn, fn = 0, 0, 0, 0
        if self.FLAGS.by_source:
            for out, target, source in zip(outputs, output_targets, sources):
                if source not in by_source:
                    by_source[source] = Confusion()
                # if out.data[0] >= .5 and target >= .5:
                if out.data[0] > .5 and target > .5:
                    tp += 1
                    by_source[source].tp += 1
                # if out.data[0] >= .5 and target <= .5:
                if out.data[0] > .5 and target < .5:
                    fp += 1
                    by_source[source].fp += 1
                if out.data[0] < .5 and target < .5:
                    tn += 1
                    by_source[source].tn += 1
                if out.data[0] < .5 and target > .5:
                    fn += 1
                    by_source[source].fn += 1
        else:
            for out, target in zip(outputs, output_targets):
                # if out.data[0] >= .5 and target >= .5:
                if out.data[0] > .5 and target > .5:
                    tp += 1
                # if out.data[0] >= .5 and target <= .5:
                if out.data[0] > .5 and target < .5:
                    fp += 1
                if out.data[0] < .5 and target < .5:
                    tn += 1
                if out.data[0] < .5 and target > .5:
                    fn += 1
        return Confusion(tp, fp, tn, fn), by_source

    def print_min_and_max(self, outputs, batch):
        max_prob, max_i_sentence = torch.topk(outputs.data, 1, 0)
        min_prob, min_i_sentence = torch.topk(outputs.data * -1, 1, 0)
        max_sentence = batch.sentences_view[max_i_sentence[0][0]]
        min_sentence = batch.sentences_view[min_i_sentence[0][0]]
        print("max:", max_prob[0][0], max_sentence)
        print("min:", min_prob[0][0] * -1, min_sentence)

    def get_metrics(self, outputs, batch, by_source=None):
        targets = torch.Tensor(batch.targets_view)
        # targets = Variable(torch.FloatTensor(batch.targets_view)).view(-1, 1)
        if self.FLAGS.gpu:
            targets = targets.cuda()
        loss = self.get_batch_loss(outputs, targets)
        confusion, by_source = self.batch_confusion(outputs, targets, batch.source_view, by_source)
        return loss, confusion, by_source

    def run_batch(self, batch, backprop, by_source=None):
        outputs, hidden = self.get_batch_output(batch)
        loss, confusion, by_source = self.get_metrics(outputs, batch, by_source)
        if backprop:
            self.backprop(loss)
        if self.FLAGS.gpu:
            loss = loss.cpu()
        return outputs, loss.data[0], confusion, by_source

    def print_stats(self, loss, confusion):
        print("avg loss\t" + self.my_round(loss))
        print("accuracy\t" + self.my_round(confusion.accuracy()))
        print("matthews\t" + self.my_round(confusion.matthews()))
        print('f1\t\t\t' + self.my_round(confusion.f1()))
        print("tp={0[0]:.4g}, tn={0[1]:.4g}, fp={0[2]:.4g}, fn={0[3]:.4g}".format(confusion.percentages()))

    def logs(self, n_batches, train_avg_loss, valid_avg_loss, t_confusion, v_confusion, model_saved):
        """
        write stats to log file
        """
        self.LOGS.write("\t" + str(n_batches) + "\t")
        self.LOGS.write("\t" + self.my_round(train_avg_loss) + "\t")
        self.LOGS.write("\t" + self.my_round(valid_avg_loss) + "\t")
        self.LOGS.write("\t" + self.my_round(t_confusion.matthews()) + "\t")
        self.LOGS.write("\t" + self.my_round(v_confusion.matthews()) + "\t")
        self.LOGS.write("\t" + self.my_round(t_confusion.f1()) + "\t")
        self.LOGS.write("\t" + self.my_round(v_confusion.f1()) + "\t")
        self.LOGS.write("\t" + "tp={0[0]:.4g}, tn={0[1]:.4g}, fp={0[2]:.4g}, fn={0[3]:.4g}".format(v_confusion.percentages()) + "\t")
        self.LOGS.write("\t" + str(model_saved) + "\n")
        self.LOGS.flush()

    def cluster_logs(self, n_batches, train_avg_loss, valid_avg_loss, t_confusion, v_confusion, model_saved, training_stats_by_source, valid_stats_by_source):
        """
        write stats to log file, spacing updated for cluster
        """
        self.LOGS.write("\t" + str(n_batches))
        self.LOGS.write("\t" + self.my_round(train_avg_loss) + "\t")
        self.LOGS.write("\t" + self.my_round(valid_avg_loss) + "\t")
        self.LOGS.write("\t" + self.my_round(t_confusion.matthews()) + "\t")
        self.LOGS.write("\t" + self.my_round(v_confusion.matthews()) + "\t")
        self.LOGS.write("\t" + self.my_round(t_confusion.f1()))
        self.LOGS.write("\t" + self.my_round(v_confusion.f1()) + "\t")
        self.LOGS.write("\t" + "tp={0[0]:.2g}, tn={0[1]:.2g}, fp={0[2]:.2g}, fn={0[3]:.2g}".format(v_confusion.percentages()) + "\t")
        self.LOGS.write("\t" + str(model_saved))
        if self.FLAGS.by_source:
            self.log_by_source(training_stats_by_source, valid_stats_by_source)
        self.LOGS.write("\n")
        self.LOGS.flush()

    def log_by_source(self, training_stats_by_source, valid_stats_by_source):
        """
        write stats to log file broken down by data source
        """
        for source in list(set(training_stats_by_source.keys()).union(valid_stats_by_source.keys())):
            if source in training_stats_by_source:
                t_confusion = training_stats_by_source[source]
            else:
                t_confusion = Confusion()
            if source in valid_stats_by_source:
                v_confusion = valid_stats_by_source[source]
            else:
                v_confusion = Confusion()
            self.LOGS.write("\t" + source)
            self.LOGS.write("\t" + self.my_round(t_confusion.matthews()))
            self.LOGS.write("\t" + self.my_round(v_confusion.matthews()))
            self.LOGS.write("\t" + self.my_round(t_confusion.f1()))
            self.LOGS.write("\t" + self.my_round(v_confusion.f1()))
            self.LOGS.write("\t" + "tp={0[0]:.2g}, tn={0[1]:.2g}, fp={0[2]:.2g}, fn={0[3]:.2g}".format(
                v_confusion.percentages()) + "\t")

    @staticmethod
    def my_round(n):
        return "{0:.4g}".format(n)

    def run_stage(self, epoch, backprop, stages_per_epoch, prints_per_stage):
        """
        Training loop for a single evaluation stage (sub-epoch)
        :param epoch: an object of type utils.classifier_data_utils.CorpusEpoch
        :param backprop: boolean switch true if in training mode, false in evaluation mode
        :param stages_per_epoch: number of evaluation stages per epoch
        :param prints_per_stage: number of prints to output per evaluation stage (not writing to logs)
        :return: stats for the evaluation stage
        """
        has_next = True
        n_batches = 0
        stage_batches = int(math.ceil(float(epoch.n_batches)/stages_per_epoch))
        print_batches = int(math.ceil(float(stage_batches)/prints_per_stage))
        print_loss = 0
        print_confusion = Confusion()
        stage_loss = 0
        stage_confusion = Confusion()
        outputs, batch = None, None
        stats_by_source = {}
        while has_next and n_batches < stage_batches:
            n_batches += 1
            batch, has_next = epoch.get_new_batch()
            outputs, loss, confusion, stats_by_source = self.run_batch(batch, backprop, stats_by_source)
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
        self.log_outputs(outputs, batch)
        return stage_loss/n_batches, stage_confusion, stats_by_source


    def run_epoch(self, max_matthews, n_stages_not_converging, n_stages):
        """
        Training loop for a single epoch
        :param max_matthews: highest dev set matthews score so far, for early stop
        :param n_stages_not_converging: counter for number of stages without setting new best Matthews on dev set
        :param n_stages: counter for number of stages started so far in current epoch
        :return: updated versions of all inputs
        """
        train = cdu.CorpusEpoch(list(self.dm.training_pairs), self.dm)
        valid = cdu.CorpusEpoch(list(self.dm.valid_pairs), self.dm)
        for _ in range(self.FLAGS.stages_per_epoch):
            if n_stages_not_converging > self.FLAGS.convergence_threshold:
                raise NotConvergingError
            n_stages += 1
            print("-------------training-------------")
            train_loss, train_confusion, training_stats_by_source = self.run_stage(train, True, self.FLAGS.stages_per_epoch, self.FLAGS.prints_per_stage)
            print("-------------validation-------------")
            valid_loss, valid_confusion, valid_stats_by_source = self.run_stage(valid, False, self.FLAGS.stages_per_epoch, 1)
            if valid_confusion.matthews() > max_matthews:
                max_matthews = valid_confusion.matthews()
                n_stages_not_converging = 0
                torch.save(self.model.state_dict(), self.OUTPUT_PATH)
                print("MODEL SAVED")
                self.cluster_logs(n_stages, train_loss, valid_loss, train_confusion, valid_confusion, True, training_stats_by_source, valid_stats_by_source)
            else:
                n_stages_not_converging += 1
                self.cluster_logs(n_stages, train_loss, valid_loss, train_confusion, valid_confusion, False, training_stats_by_source, valid_stats_by_source)
        return max_matthews, n_stages_not_converging, n_stages

    def start_up_print_and_logs(self):
        print("======================================================================")
        print("                              TRAINING")
        print("======================================================================")
        print(self.to_string())
        self.LOGS.write("\n\n" + self.to_string() + "\n")
        self.LOGS.write(
            "# batches | train avg loss | valid avg loss | t matthews | v matthews | t f1 | v f1 |      confusion      |model saved\n" +
            "----------|----------------|----------------|------------|------------|------|------|---------------------|-----------\n")
        self.LOGS.flush()


    def log_outputs(self, outputs, batch):
        if self.FLAGS.gpu:
            outputs = outputs.cpu()
        to_write = "\n\nNEW STAGE\n"
        for o, t, s in zip(outputs.data, batch.targets_view, batch.sentences_view):
            to_write += "%f\t%f\t%s" % (o[0], t, s)
        # self.OUT_LOGS.write("sentence!\n")
        # self.OUT_LOGS.write(str(outputs))
        # self.OUT_LOGS.write(str(batch))
        self.OUT_LOGS.write(to_write)
        self.OUT_LOGS.flush()


    def run(self):
        """The outer loop of the model trainer"""
        self.start_up_print_and_logs()
        epoch = 0
        n_stages = 0
        max_matthews = 0
        n_stages_not_converging = 0
        try:
            while epoch < self.FLAGS.max_epochs:
                epoch += 1
                print("===========================EPOCH %d=============================" % epoch)
                max_matthews, n_stages_not_converging, n_stages = self.run_epoch(max_matthews, n_stages_not_converging, n_stages)
        except NotConvergingError:
            self.model.load_state_dict(torch.load(self.OUTPUT_PATH))
            print("=====================TEST==================")
            test_loss, test_confusion = self.run_stage(cdu.CorpusEpoch(self.dm.test_pairs, self.dm), False, 1, 1)
            self.LOGS.write("accuracy\t" + self.my_round(test_confusion.accuracy()) + "\n")
            self.LOGS.write("matthews\t" + self.my_round(test_confusion.matthews()) + "\n")
            self.LOGS.write('f1\t\t\t' + self.my_round(test_confusion.f1()) + "\n")
            self.LOGS.write("\t" + "tp={0[0]:.2g}, tn={0[1]:.2g}, fp={0[2]:.2g}, fn={0[3]:.2g}".format(test_confusion.percentages()) + "\n")
        finally:
            self.LOGS.close()


class NotConvergingError(Exception):
    """
    Raised for early stop when model stops converging, used to exit outer training loop
    """
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)