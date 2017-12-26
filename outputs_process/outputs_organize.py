import operator




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
        return s

outlines = []
for k,v in d.iteritems():
    stats = []
    if k!="":
        ks = k.split("-")
        del ks[4]
        del ks[4]
        stats.append(ks[0].split("\t")[-1])
        stats.append(str(ks[1][2:]).zfill(5))
        stats.append(ks[2][6:])
        stats.append(abbr(ks[3][4:]))
        stats.append(str(ks[4][6:]).zfill(4))
        stats.append(abbr(ks[5][4:]))
        stats.append(ks[6][10:11])
    else:
        continue
    if v != []:
        lastv = v[-1]
        vs = lastv.split("\t")
        stats.append(vs[8])
        stats.append(vs[11])
        stats.append(vs[13])
    else:
        stats.extend([0,0,0])
    outline = ""
    for s in stats:
        outline += str(s) + "\t"
    print(outline)
    outlines.append(outline)



for line in outlines:
    vals = line.split("\t")
    d[line] = float(vals[-4])

sorted_d = sorted(d.items(), key=operator.itemgetter(1), reverse=True)

for k,v in sorted_d:
    print(k)