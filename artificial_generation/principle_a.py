pronouns = ["I",
            "you",
            "he",
            "she",
            "we",
            "they"]



tv_simple_pres = ["shaved",
                   "saw",
                   "talked to",
                   "hurt",
                   "told a funny story about"]

tv_likes_to = ["tried to shave",
               "tried to see",
               "tried to talk to",
               "tried to hurt",
               "tried to tell a funny story about"]

modifiers = ["yesterday",
               "this morning",
               "last year",
               "fifteen minutes before John arrived",
               ""]

reflexives = ["himself",
              "herself",
              "themselves",
              "yourself",
              "myself",
              "ourselves"]

verbs = ["saw",
         "talked to",
         "surprised",
         "amused",
         "hurt",
         "hit",
         "shocked"]



out = open("../acceptability_corpus/artificial/principle_a_large.tsv", "w")

for pronoun in pronouns:
    for refl in reflexives:
        good_pairs = [("I", "myself"),
                      ("you", "yourself"),
                      ("he", "himself"),
                      ("she", "herself"),
                      ("we", "ourselves"),
                      ("they", "themselves")]
        label = 1 if (pronoun, refl) in good_pairs else 0
        for verb in verbs:
            out.write("\t%s\t\t%s %s %s.\n" % (label, pronoun, verb, refl))



# for name in names:
#     for modifier in modifiers:
#         for refl in reflexives:
#             name_gender = name[1]
#             cases = {"himself": lambda g: 1 if g <= 0.5 else 0,
#                      "herself": lambda g: 1 if g >= 0.5 else 0,
#                      "itself": lambda g: 0}
#             label = cases[refl](name_gender)
#             for tv in tv_simple_pres:
#                 print("%s\t%s %s %s %s." % (label, name[0], tv, refl, modifier))
#             for tv in tv_likes_to:
#                 print("%s\t%s %s %s %s." % (label, name[0], tv, refl, modifier))


