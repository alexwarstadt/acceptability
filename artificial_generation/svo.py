
animate = ["Michael",
    "Christopher",
    "Jason",
    "David",
    "James",
    "Nicole",
    "Jessica",
    "Elizabeth",
    "Rebecca",
    "Kelly"]

inanimate = ["the book",
             "the article",
             "the letter",
             "the story",
             "the chapter"]

verb_a_i = ["read",
            "wrote"]

verb_a_a = ["met",
            "talked to"]

# out = open("../acceptability_corpus/artificial/svo.tsv", "w")
# # Write sov, svo, ...
# for s in animate:
#     for o in inanimate:
#         for v in verb_a_i:
#             for pattern in [("1", s, v, o),
#                             ("0", s, o, v),
#                             ("0", v, s, o),
#                             ("0", v, o, s),
#                             # ("0", o, s, v),
#                             ("0", o, v, s)]:
#                 out.write("\t%s\t\t%s %s %s.\n" % pattern)
# out.close()

# out = open("../acceptability_corpus/artificial/svs.tsv", "w")
# for s1 in animate:
#     for s2 in animate:
#         if s1 != s2:
#             for v in verb_a_a:
#                 for pattern in [("1", s1, v, s2),
#                                 # ("1", s1, s2, v),
#                                 ("0", v, s1, s2)]:
#                     out.write("\t%s\t\t%s %s %s.\n" % pattern)
# out.close()


out = open("../acceptability_corpus/artificial/john_met_the_book.tsv", "w")
for s in animate:
    for o in inanimate:
        for v in verb_a_a:
            for pattern in [("0", s, v, o),
                            ("0", s, o, v),
                            ("0", v, s, o),
                            ("0", v, o, s),
                            ("0", o, s, v),
                            ("0", o, v, s)]:
                out.write("\t%s\t\t%s %s %s.\n" % pattern)
out.close()
