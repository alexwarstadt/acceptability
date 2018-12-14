#Template: animate V inanimate


s = {1: ["alex",
         "anne",
         "brian",
         "betty",
         "carter",
         "caroline",
         "dean",
         "deborah",
         "eli",
         "esme"],
     "pro": ["she",
             "he",
             "we",
             "they"],
     2: ["the boy",
         "the girl",
         "the man",
         "the woman"
         ]
     }


v = {"past": ["broke",
             "cleaned",
             "bought",
             "sold",
             "lifted",
             "saw",
             "admired",
             "designed",
             "remembered",
             "fixed"],
     "pres": ["breaks",
              "cleans",
              "buys",
              "sells",
              "lifts",
              "sees",
              "admires",
              "designs",
              "remembers",
              "fixes"]
     }


o_1 = {"pl": ["cars",
              "pottery",
              "boxes",
              "lamps",
              "furniture",
              "books",
              ""],
       }


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
