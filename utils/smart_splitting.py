import random

file = "/Users/alexwarstadt/Workspace/acceptability/acceptability_corpus/tokenized/all.tsv"

sources = {}

for line in open(file):
    vals = line.split("\t")
    if vals[0] in sources.keys():
        sources[vals[0]].append(line)
    else:
        sources[vals[0]] = [line]


splits = {}

for source in sources:
    splits[source] = [0].extend([str(len(sources[source]))])

splits["in"] = []
splits["out"] = []

for n in range(10):
    print "n = %d" % n
    sources_copy = sources.copy()
    in_domain = []
    out_domain = []

    while sources_copy.keys() != []:
        source = random.choice(sources_copy.keys())
        if len(out_domain) < 1500:
            out_domain.extend(sources_copy[source])
            splits[source].append("0")
        else:
            in_domain.extend(sources_copy[source])
            splits[source].append("1")
        del sources_copy[source]

    splits["out"].append(str(len(out_domain)))
    splits["in"].append(str(len(in_domain)))

    in_domain_file = open("/Users/alexwarstadt/Workspace/acceptability/acceptability_corpus/splits/in_domain_%s.tsv" % n, "w")
    out_domain_file = open("/Users/alexwarstadt/Workspace/acceptability/acceptability_corpus/splits/out_domain_%s.tsv" % n, "w")

    for line in in_domain:
        in_domain_file.write(line)
    for line in out_domain:
        out_domain_file.write(line)

    in_domain_file.close()
    out_domain_file.close()

splits.keys().sort()
for s in splits.keys():
    print s + "\t" + "\t".join(splits[s])

pass