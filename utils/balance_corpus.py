import random

file = open("/Users/alexwarstadt/Workspace/acceptability/acceptability_corpus/corpus_table")

poz_list, neg_list = [], []

for line in file:
    vals = line.split("\t")
    if float(vals[1]) >= 0.5:
        poz_list.append(line)
    else:
        neg_list.append(line)

ratio = float(len(poz_list)-len(neg_list)) / float(len(poz_list))
print(len(poz_list))
print(len(neg_list))
print ratio


file = open("/Users/alexwarstadt/Workspace/acceptability/acceptability_corpus/corpus_table")

out = open("/Users/alexwarstadt/Workspace/acceptability/acceptability_corpus/corpus_table_balanced", "w")

for line in file:
    vals = line.split("\t")
    if float(vals[1]) >= 0.5:
        if random.uniform(0,1) > ratio:
            out.write(line)
    else:
        out.write(line)
