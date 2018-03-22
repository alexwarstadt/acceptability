# names = [("Michael", 0),
#     ("Christopher", 0),
#     ("Jason", 0),
#     ("David", 0),
#     ("James", 0),
#     ("John", 0),
#     ("Robert", 0),
#     ("Brian", 0),
#     ("William", 0),
#     ("Matthew", 0),
#     ("Joseph", 0),
#     ("Daniel", 0),
#     ("Kevin", 0),
#     ("Eric", 0),
#     ("Jeffrey", 0),
#     ("Richard", 0),
#     ("Scott", 0),
#     ("Mark", 0),
#     ("Steven", 0),
#     ("Thomas", 0),
#     ("Timothy", 0),
#     ("Anthony", 0),
#     ("Charles", 0),
#     ("Joshua", 0),
#     ("Ryan", 0),
#     ("Jennifer", 1),
#     ("Amy", 1),
#     ("Melissa", 1),
#     ("Michelle", 1),
#     ("Kimberly", 1),
#     ("Lisa", 1),
#     ("Angela", 1),
#     ("Heather", 1),
#     ("Stephanie", 1),
#     ("Nicole", 1),
#     ("Jessica", 1),
#     ("Elizabeth", 1),
#     ("Rebecca", 1),
#     ("Kelly", 1),
#     ("Mary", 1),
#     ("Christina", 1),
#     ("Amanda", 1),
#     ("Julie", 1),
#     ("Sarah", 1),
#     ("Laura", 1),
#     ("Christine", 1),
#     ("Tammy", 1),
#     ("Karen", 1),
#     ("Susan", 1),
#     ("Andrea", 1),
#     ("SHEA", 0.5),
#     ("ROUSE", 0.5),
#     ("HARTLEY", 0.5),
#     ("MAYFIELD", 0.5),
#     ("ELDER", 0.5),
#     ("RANKIN", 0.5),
#     ("COWAN", 0.5),
#     ("LUCERO", 0.5),
#     ("ARROYO", 0.5),
#     ("SLAUGHTER", 0.5),
#     ("HAAS", 0.5),
#     ("OCONNELL", 0.5),
#     ("MINOR", 0.5),
#     ("KENDALL", 0.5),
#     ("BOUCHER", 0.5),
#     ("ARCHER", 0.5),
#     ("BOGGS", 0.5),
#     ("ODELL", 0.5),
#     ("DOUGHERTY", 0.5),
#     ("ANDERSEN", 0.5),
#     ("NEWELL", 0.5),
#     ("CROWE", 0.5),
#     ("WANG", 0.5),
#     ("FRIEDMAN", 0.5),
#     ("BLAND", 0.5),
#     ("SWAIN", 0.5),
#     ("HOLLEY", 0.5),
#     ("PEARCE", 0.5),
#     ("CHILDS", 0.5),
#     ("YARBROUGH", 0.5),
#     ("GALVAN", 0.5),
#     ("PROCTOR", 0.5),
#     ("MEEKS", 0.5),
#     ("LOZANO", 0.5),
#     ("MORA", 0.5),
#     ("RANGEL", 0.5),
#     ("BACON", 0.5),
#     ("VILLANUEVA", 0.5),
#     ("SCHAEFER", 0.5),
#     ("ROSADO", 0.5),
#     ("HELMS", 0.5),
#     ("BOYCE", 0.5),
#     ("GOSS", 0.5),
#     ("STINSON", 0.5),
#     ("SMART", 0.5),
#     ("LAKE", 0.5),
#     ("IBARRA", 0.5),
#     ("HUTCHINS", 0.5),
#     ("COVINGTON", 0.5),
#     ("REYNA", 0.5),
# ]

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
         "amused"]



# for pronoun in pronouns:
#     for refl in reflexives:
#         good_pairs = [("I", "myself"),
#                       ("you", "yourself"),
#                       ("he", "himself"),
#                       ("she", "herself"),
#                       ("we", "ourselves"),
#                       ("they", "themselves")]
#         label = 1 if (pronoun, refl) in good_pairs else 0
#         for modifier in modifiers:
#             for tv in tv_simple_pres:
#                 print("\t%s\t\t%s %s %s %s." % (label, pronoun, tv, refl, modifier))
#             for tv in tv_likes_to:
#                 print("\t%s\t\t%s %s %s %s." % (label, pronoun, tv, refl, modifier))
#




out = open("../acceptability_corpus/artificial/principle_a_3.tsv", "w")

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


