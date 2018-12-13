vocab = []

vocab_file = open("/Users/alexwarstadt/Workspace/data/vocabs/vocab_100k.tsv")

for x in vocab_file:
    vocab.append(x.strip())




verbs = {}

file = open("../acceptability_corpus/artificial/all_verbs/all_verbs.csv")

header = file.readline()

order = []

def merge(list1, list2):
    to_return = []
    for x, y in zip(list1, list2):
        if x == y:
            to_return.append(x)
        elif x == "x":
            to_return.append(y)
        elif y == "x":
            to_return.append(x)
        else:
            to_return.append("1")
            # print vals[0], x, y
    return to_return

for line in file:
    vals = line.strip().split(",")
    if vals[0] not in vocab:
        print(vals[0])
    if vals[0] not in order:
        order.append(vals[0])
    if vals[0] not in verbs:
        verbs[vals[0]] = vals[1:]
    else:
        former = verbs[vals[0]]
        verbs[vals[0]] = merge(former, vals[1:])


out = open("../acceptability_corpus/artificial/all_verbs/all_verbs2.csv", "w")
out.write(header)
for x in order:
    out.write(",".join([x] + verbs[x]) + "\n")

out.close()


