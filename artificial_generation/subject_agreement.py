
singular_subj_rel = ["the author",
                     "the owner",
                     "the painter",
                     "the manager"]

plural_subj_rel = ["the authors",
                   "the owners",
                   "the painters",
                   "the managers"]



singular_subj_simple = [
                        # "the boy",
                        # "she",
                        # "my friend",
                        # "that guy",
                        # "the dancer",
                        # "everyone",
                        # "Mary",
                        # "the doctor",
                        # "some woman",
                        # "Mr. Smith",
                        # "the guest of honor",
                        # "the prince",
                        # "this guy",
                        # "Ben",
                        # "one person",
                        # "the tall kid",
                        # "an old man",
                        # "her cousin",
                        # "the cat owner",
                        # "every small child",
                        # "that jerk",
                        # "the manager",
                        # "my favorite actor"

                        # "her daughter",
                        # "John",
                        # "this lady",
                        # "at least one student",
                        # "he",
                        # "the director",
                        # "the chef",
                        # "the retired professor",

                        "his brother",
                        "Betty",
                        "this person",
                        "no more than one student",
                        "that one",
                        "the police officer",
                        "the singer",
                        "the senior member",


                        ]

plural_subj_simple = [
                      # "the boys",
                      # "we",
                      # "my friends",
                      # "those guys",
                      # "the dancers",
                      # "most people",
                      # "Mary and Sarah",
                      # "the doctors",
                      # "some women",
                      # "Mr. and Mrs. Smith",
                      # "the guests of honor",
                      # "the princes",
                      # "these guys",
                      # "Ben and Joe",
                      # "two people",
                      # "the tall kids",
                      # "old men",
                      # "her cousins",
                      # "the cat owners",
                      # "several small children",
                      # "those jerks",
                      # "the managers",
                      # "my favorite actors"

                      # "John and his wife",
                      # "her daughters",
                      # "these ladies",
                      # "at least five students",
                      # "the directors",
                      # "the chefs",
                      # "they",
                      # "the retired professors"


                      "his brothers",
                        "Betty and Dave",
                        "these people",
                        "no more than ten student",
                        "those ones",
                        "the police officers",
                        "the singers",
                        "the senior members"
                      ]

singular_verb = ["is here",
                 "hasn't been there",
                 "likes to read",
                 "knows where to go",
                 "seems nice",
                 "is walking",
                 "sleeps here",
                 'has a problem',
                 "has to go",
                 "gets it",
                 "has the flu",
                 "feels great",
                 "isn't going to come",
                 "thinks that's a bad idea",
                 "doubts that",
                 "refuses to say so",
                 "doesn't want to",
                 "does agree",
                 "believes you",
                 "sees the issue",
                 "understands why",
                 "knows how",
                 "is trying to do that",
                 "wonders whether that's true",
                 "acts as if that's true",
                 "sits in this seat",
                 "works at this desk",
                 "tells the truth"
                 ]

plural_verb = ["are here",
                 "haven't been there",
                 "like to read",
                 "know where to go",
                 "seem nice",
                 "are walking",
                 "sleep here",
                 "have a problem",
                 "have to go",
                 "get it",
                 "have the flu",
                 "feel great",
                 "aren't going to come",

                 "think that's a bad idea",
                 "doubt that",
                 "refuse to say so",
                 "don't want to",
                 "do agree",
                 "believe you",
                 "see the issue",
                 "understand why",
                 "know how",
                 "are trying to do that",
                 "wonder whether that's true",
                 "act as if that's true",
                 "sit in this seat",
                 "work at this desk",
                 "tell the truth"
               ]




out = open("../acceptability_corpus/artificial/tokenized/sp_long_heldout_subj/dev.tsv", "w")

for s in singular_subj_simple:
    for v in singular_verb:
        label = "1"
        out.write("\t%s\t\t%s %s.\n" % (label, s, v))
    for v in plural_verb:
        label = "0"
        out.write("\t%s\t\t%s %s.\n" % (label, s, v))

for s in plural_subj_simple:
    for v in singular_verb:
        label = "0"
        out.write("\t%s\t\t%s %s.\n" % (label, s, v))
    for v in plural_verb:
        label = "1"
        out.write("\t%s\t\t%s %s.\n" % (label, s, v))

out.close()
