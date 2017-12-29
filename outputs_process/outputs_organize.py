import operator
import re




file = open("../sweep_outputs")

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
                    v += x[0] + "=" + x[1].zfill(5) + ","
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
        if xs[0] == "True":             # first
            xs = xs[1:7]
        elif len(xs) == 12:             # double for some reason
            chunks.append(xs[0:6])
            xs = xs[6:]
        del xs[1]
        del xs[2]
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
        del ks[4]
        del ks[4]
        stats.append(ks[0].split("\t")[-1])
        stats.append(str(ks[1][2:]).zfill(5))
        stats.append(ks[2][6:])
        stats.append(abbr(ks[3][4:]))
        stats.append(str(ks[4][6:]).zfill(4))
        stats.append(abbr(ks[5][4:]))
        stats.append(day)
        stats.append(ks[6][10:11])
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
    d[line] = float(vals[8])

sorted_d = sorted(d.items(), key=operator.itemgetter(1), reverse=True)

for k,v in sorted_d:
    print(k)