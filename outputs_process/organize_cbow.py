import operator
import re

# A script for processing the CBOW logs into a table


file = open("../temp")

lines=[]
for line in file:
    if line.startswith("\t") or line.startswith("experiment name") or line.startswith("\n"):
        lines.append(line)

d = {}
k = ""
v = []

for line in lines:
    if line.startswith("exp"):
        k = line
        v = []
    if line.startswith("\t"):
        v.append(line)
    else:
        d[k] = v


def abbr(s):
    if s == "levin":
        return "lv"
    elif s == "levin_balanced":
        return "lvb"
    elif s == "aj_all":
        return "aj"
    elif s == "aj_balanced":
        return "ajb"
    elif s == "discriminator":
        return "discr"
    else:
        vs = s.split("_")
        if vs[0] == "permuted":
            return "shuff 0.%s-0.%s" % tuple(vs[1:])
        else:
            return "swap %s-%sx\t" % tuple(vs[1:])


def pretty_print(vs):
    new_vs = []
    for v in vs:
        try:
            float(v)
            new_vs.append(v.zfill(6))
        except ValueError:
            conf = v.split(",")
            if len(conf) == 4:
                v = ""
                for x in conf:
                    x = x.split("=")
                    v += x[0] + "=" + x[1].strip().zfill(5) + ","
            new_vs.append(v)
    return reduce(lambda x, y: x + '\t' + y, new_vs).strip()


def chunk(lastv):
    """break up the stats into lists by source, containing v-matthews, v-f1, v-confusion"""
    vs = lastv.split('\t')
    chunks = []
    chunk = []
    chunk.append(vs[8])                 # v matthews
    chunk.append(vs[11])                # v f1
    chunk.append(vs[13])                # v confusion
    chunks.append(chunk)
    for x in lastv.split("\t\t")[6:]:
        xs = x.split('\t')
        if xs[0] == "True" or xs[0] == "False":             # first
            xs = xs[1:7]
        elif len(xs) == 12:             # double for some reason
            del xs[1]
            del xs[2]
            chunks.append(xs[0:4])
            xs = xs[4:]
        try:
            del xs[1]                       # delete training scores
            del xs[2]
        except IndexError:
            pass
        chunks.append(xs)
    return chunks


outlines = []
for k,v in d.iteritems():
    stats = []
    if k!="":
        ks = k.split("-")
        day = re.search("day[\d\.]*", ks[4])
        if day is None:
            day = "day7"
        else:
            day = day.group()
        sweepname = ks[0].split("\t")[-1]
        sweepname = sweepname.split("_")
        sweepname[-1] = sweepname[-1].zfill(3)
        sweepname = "_".join(sweepname)
        stats.append(sweepname)
        stats.append(str(ks[1][2:]).zfill(5))
        stats.append(ks[2][6:])
        stats.append(abbr(ks[3][4:]))
        stats.append("max" if ks[4].strip()[-1] == "1" else "no")
    else:
        continue
    if v != []:
        trues = filter(lambda x: "True" in x, v)
        if len(trues) > 0:
            lastv = trues[-1]
        else:
            lastv = v[-1]
        # vs = lastv.split("\t")
        chunks = chunk(lastv)
        for c in chunks:
            stats.append(pretty_print(c))

    else:
        stats.extend([0,0,0])
    outline = ""
    for s in stats:
        outline += str(s) + "\t"
    print(outline)
    outlines.append(outline)



for line in outlines:
    vals = filter(lambda s: s != "", line.split("\t"))
    # try:
    d[line] = float(vals[5])
    # except ValueError:
    #     pass

sorted_d = sorted(d.items(), key=operator.itemgetter(1), reverse=True)

for k,v in sorted_d:
    print(k)