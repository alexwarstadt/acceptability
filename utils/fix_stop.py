
files = ["/scratch/asw462/data/perm-5-6/train.txt",
         "/scratch/asw462/data/perm-5-6/test.txt",
         "/scratch/asw462/data/perm-5-6/valid.txt",
         "/scratch/asw462/data/perm-5-6/all.txt",
         "/scratch/asw462/data/perm-1-6/train.txt",
         "/scratch/asw462/data/perm-1-6/test.txt",
         "/scratch/asw462/data/perm-1-6/valid.txt",
         "/scratch/asw462/data/perm-1-6/all.txt",
         "/scratch/asw462/data/perm-1-2/train.txt",
         "/scratch/asw462/data/perm-1-2/test.txt",
         "/scratch/asw462/data/perm-1-2/all.txt",
         "/scratch/asw462/data/perm-1-2/valid.txt",
         "/scratch/asw462/data/perm-3-4/train.txt",
         "/scratch/asw462/data/perm-3-4/test.txt",
         "/scratch/asw462/data/perm-3-4/valid.txt",
         "/scratch/asw462/data/perm-3-4/all.txt"
         ]

for file in files:
    out = open(file[0:-4] + "-2.txt", "w")
    for line in open(file):
        tabs = line.split("\t")
        words = tabs[-1].split()[0:31]
        sentence = reduce(lambda s1, s2: s1 + " " + s2, words, "")
        tabs[-1] = sentence
        line = reduce(lambda s1, s2: s1 + "\t" + s2, tabs, "")
        out.write(line.strip() + "\n")