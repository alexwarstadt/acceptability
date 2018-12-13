import os
import numpy as np

dir = "/Users/alexwarstadt/Workspace/acceptability_playground/verb_classes/"
data_dir = "/Users/alexwarstadt/Workspace/acceptability/acceptability_corpus/artificial/verb_classes/"

for output_dir in os.listdir(dir + "outputs"):
    table = []
    scores_table = []
    sentences = [x.strip() for x in open(data_dir + output_dir + "/test.tsv")]
    for f in os.listdir(dir + "outputs/" + output_dir):
        outputs = [int(x) for x in open(dir + "outputs/" + output_dir + "/" + f) if x != '']
        scores = [l for l in open(dir + output_dir + "/" + f[0:-4] + ".log")]
        scores_table.append([scores[-1].split()[2], scores[-1].split()[4], scores[-4].split()[2]])
        # except ValueError:
        #     pass
        table.append(outputs)
    out_file = open(dir + "tables/" + output_dir + ".tsv", "w")

    table = np.array(table)
    table = table.transpose()

    out_file.write("test MCC\t\t" + "\t".join([str(x[0]) for x in scores_table]) + "\n")
    out_file.write("test Accuracy\t\t" + "\t".join([str(x[1]) for x in scores_table]) + "\n")
    out_file.write("dev MCC\tgold\t" + "\t".join([str(x[2]) for x in scores_table]) + "\n")

    for row, sentence in zip(table, sentences):
        s = sentence.split("\t")[3].strip()
        l = sentence.split("\t")[1].strip()
        line = "\t".join([s, l] + [str(x) for x in row])
        out_file.write(line + "\n")

    out_file.close()

