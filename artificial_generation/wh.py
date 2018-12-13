import random

# animate = ["the boy",
#             "my friend",
#             "that guy",
#             "the dancer",
#             "everyone",
#             "Mary",
#             "the doctor",
#             "some woman",
#             "John",
#             "Mr. Smith",
#             "the guest of honor",
#             "Elizabeth"
#            ]
#
# inanimate = ["the book",
#              "a car",
#              "this table",
#              "a house",
#              "the computer",
#              "the wine",
#              "the bicycle",
#              "the pen",
#              "the box of records"
#              ]
#
# q_animate_animate = ["does %s love %s",
#                      "did %s meet %s",
#                      "did %s talk to %s",
#                      "did %s attack %s",
#                      "does %s know %s",
#                      "will %s help %s",
#                      "does %s like %s",
#                      "did %s give a book to %s",
#                      "should %s introduce %s to Sue",
#                      "did %s describe %s to Sue",
#                      "did %s spend the day with %s yesterday",
#                      "can %s visit %s tomorrow",
#                      "does %s like to spend time with %s on the weekends"
#                      ]
#
# q_animate_inanimate = ["does %s own %s",
#                        "did %s buy %s",
#                        "did %s move %s",
#                        "does %s like %s",
#                        "did %s destroy %s",
#                        "can %s see %s",
#                        "will %s sell %s to Sue",
#                        "did %s give %s to Sue",
#                        "should %s hide %s from Sue",
#                        "did %s get %s for Andrew"
#                        ]


# d_animate_animate = ["%s loves %s",
#                      "%s met %s",
#                      "%s talked to %s",
#                      "%s attacked %s",
#                      "%s knows %s",
#                      "%s will help %s",
#                      "%s likes %s",
#                      "%s gave a book to %s",
#                      "%s should introduce %s to Sue",
#                      "%s described %s to Sue",
#                      ]
#
# d_animate_inanimate = ["%s owns %s",
#                        "%s bought %s",
#                        "%s moved %s",
#                        "%s likes %s",
#                        "%s destroyed %s",
#                        "%s can see %s",
#                        "%s will sell %s to Sue",
#                        "%s gave %s to Sue",
#                        "%s should hide %s from Sue",
#                        ]

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



sets = [
    {"obj": ["the book",
             "the article",
             "the letter",
             "the story",
             "the chapter"
             ],
     "pred": ["did %s read %s"
              ]
    },

    {"obj": ["the cake",
             "the salad",
             "the sandwich",
             "the steak"
             ],
     "pred": ["did %s devour %s",
              "did %s eat %s"
              ]
    },

    {"obj": ["the cake",
             "the cookies"
             ],
     "pred": ["did %s bake %s"
              ]
    },

    {"obj": ["the board",
             "the chalk"
             ],
     "pred": ["did %s erase %s"
              ]
    },

    {"obj": ["the leaves"
             ],
     "pred": ["did %s rake %s"
              ]
    },

    {"obj": ["the egg",
             "the potato"
             ],
     "pred": ["did %s boil %s",
              "did %s fry %s"
              ]
    },

    {"obj": ["the floor"
             ],
     "pred": ["did %s sweep %s",
              "did %s mop %s"
              ]
    },

    {"obj": ["the bicycle",
             "the train"
             ],
     "pred": ["did %s ride %s"
              ]
    }

]



out = open("../acceptability_corpus/artificial/wh_simple.tsv", "w")

def my_write(s):
    out.write(s.replace("  ", " "))

for set in sets:
    for s in animate:
        for q in set["pred"]:
            for o in set["obj"]:
                no_gap_q = q % (s, o)
                my_write("\t%s\t\t%s %s ?\n" % (0, "what", no_gap_q))
                gap_q = q % (s, "")
                my_write("\t%s\t\t%s %s ?\n" % (1, "what", gap_q))

# for a1 in animate:
#     for q in q_animate_animate:
#         a2 = a1
#         while a2 == a1:
#             a2 = random.choice(animate)
#         no_gap_q = q % (a1, a2)
#         my_write("\t%s\t\t%s %s ?\n" % (0, "who", no_gap_q))
#         gap_q = q % (a1, "")
#         my_write("\t%s\t\t%s %s ?\n" % (1, "who", gap_q))
        # for d in d_animate_animate:
        #     if a1 != a2:
        #         no_gap = d % (a1, a2)
        #         my_write("\t%s\t\t%s .\n" % (1, no_gap))

# for a1 in animate:
#     for q in q_animate_inanimate:
#         a2 = a1
#         while a2 == a1:
#             a2 = random.choice(inanimate)
#         no_gap_q = q % (a1, a2)
#         my_write("\t%s\t\t%s %s ?\n" % (0, "what", no_gap_q))
#         gap_q = q % (a1, "")
#         my_write("\t%s\t\t%s %s ?\n" % (1, "what", gap_q))
        # for d in d_animate_inanimate:
        #     if a1 != a2:
        #         no_gap = d % (a1, a2)
        #         my_write("\t%s\t\t%s .\n" % (1, no_gap))

# for a1 in animate:
#     for q in q_animate_animate:
#         gap_q = q % (a1, "")
#         my_write("\t%s\t\t%s %s ?\n" % (1, "who", gap_q))
#         for a2 in animate:
#             if a1 != a2:
#                 no_gap_q = q % (a1, a2)
#                 my_write("\t%s\t\t%s %s ?\n" % (0, "who", no_gap_q))
#         for d in d_animate_animate:
#             if a1 != a2:
#                 no_gap = d % (a1, a2)
#                 my_write("\t%s\t\t%s .\n" % (1,  no_gap))
#
# for a1 in animate:
#     for q in q_animate_inanimate:
#         gap_q = q % (a1, "")
#         my_write("\t%s\t\t%s %s ?\n" % (1, "what", gap_q))
#         for a2 in inanimate:
#             if a1 != a2:
#                 no_gap_q = q % (a1, a2)
#                 my_write("\t%s\t\t%s %s ?\n" % (0, "what", no_gap_q))
#         for d in d_animate_inanimate:
#             if a1 != a2:
#                 no_gap = d % (a1, a2)
#                 my_write("\t%s\t\t%s .\n" % (1,  no_gap))

out.close()




"Who did the doctor meet?"
"The doctor met Bill."
"Who did the doctor meet Bill?"


"Who did lucy say the doctor met?"
"Lucy said the doctor met Bill."
