import random
import numpy.random as np

animate = ["michael",
    "christopher",
    "jason",
    "david",
    "james",
    "nicole",
    "jessica",
    "elizabeth",
    "rebecca",
    "kelly"]

inchoative = [
    {
        "verb": ["opened",
                 "closed",
                 "shut",
                 "flung open"],
        "patient": ["the door",
                    "the gate",
                    "the hatch",
                    "the lid"]
    },

    {
        "verb": ["broke",
                 "shattered",
                 "cracked",
                 "chipped"],
        "patient": ["the vase",
                    "the glass",
                    "the bowl",
                    "the platter",
                    "the window",
                    "the screen"]
    },

    {
        "verb": ["dropped",
                 "tipped over"],
        "patient": ["the ball",
                    "the platter",
                    "the vase",
                    "computer"]
    },

    {
        "verb": ["rolled"],
        "patient": ["the ball",
                    "the wheel"]
    },

    {
        "verb": ["moved",
                 "maneuvered",
                 "steered",
                 "crashed",
                 "stopped",
                 "slowed",
                 "accelerated"],
        "patient": ["the car",
                    "the bicycle",
                    "the truck",
                    "the boat",
                    "the ship",
                    "the plane",
                    "the cart"]
    },

    {
        "verb": ["popped",
                 "burst"],
        "patient": ["the bubble",
                    "the balloon",
                    "the ball",
                    "the bag"
                    ]
    },

    {
        "verb": ["burned",
                 "scorched",
                 "singed",
                 "blackened"],
        "patient": ["the wood",
                    "the candle",
                    "the paper",
                    "the book",
                    "the toast",
                    "the steak"
                    ]
    },

    {
        "verb": ["melted",
                 "softened",
                 "liquefied"],
        "patient": ["the ice",
                    "the chocolate",
                    "the plastic",
                    "the ice cream",
                    "the popsicle"]
    },

    {
        "verb": ["bounced"
                 ],
        "patient": ["the ball",
                    "the balloon"]
    },

    {
        "verb": ["twirled",
                 "twisted"
                 ],
        "patient": ["the ribbon",
                    "the flag"]
    },

    {
        "verb": ["turned",
                 "rotated"
                 ],
        "patient": ["the sculpture",
                    "the sofa",
                    "the image"]
    },

    {
        "verb": ["vaporized",
                 "evaporated",
                 "condensed"
                 ],
        "patient": ["the alcohol",
                    "the water",
                    "the solution"]
    },

    {
        "verb": ["bent",
                 "folded",
                 "wrinkled",
                 "crumpled",
                 "warped",
                 "frayed",
                 "stretched",
                 "unfolded",
                 "flattened",
                 "smoothed"
                 ],
        "patient": ["the paper",
                    "the sheet",
                    "the poster",
                    "the leaf",
                    "the foil"]
    },

    {
        "verb": ["grew"
                 ],
        "patient": ["the tree",
                    "the plant",
                    "the tomato"]
    },

    {
        "verb": ["enlarged",
                 "expanded",
                 "shrank"
                 ],
        "patient": ["the image",
                    "the picture",
                    "the project"]
    },

    {
        "verb": ["dissolved"
                 ],
        "patient": ["the sugar",
                    "the salt",
                    "the crystals"]
    },

    {
        "verb": ["darkened",
                 "brightened",
                 "faded",
                 "dimmed"
                 ],
        "patient": ["the lights",
                    "the room",
                    "the image",
                    "the picture"]
    },

    {
        "verb": ["soaked",
                 "steeped"
                 ],
        "patient": ["the tea",
                    "the coffee",
                    "the spices",
                    "the herbs"]
    },

    {
        "verb": ["heated",
                 "cooled",
                 "warmed",
                 "defrosted"
                 ],
        "patient": ["the water",
                    "the food",
                    "the casserole",
                    "the dinner"]
    },

    {
        "verb": ["emptied",
                 "filled"
                 ],
        "patient": ["the well",
                    "the container",
                    "the bowl",
                    "the glass"]
    },

    {
        "verb": ["loosened",
                 "tightened"
                 ],
        "patient": ["the knot",
                    "the belt",
                    "the grip"]
    },

    {
        "verb": ["doubled",
                 "tripled",
                 "quadrupled"
                 ],
        "patient": ["the assets",
                    "the audience",
                    "the money",
                    "the garden",
                    "the crowd"]
    },

    {
        "verb": ["dried",
                 "moistened",
                 "dampened"
                 ],
        "patient": ["the cloth",
                    "the clothes",
                    "the plant",
                    "the grass"]
    },

    {
        "verb": ["improved",
                 "developed",
                 "expanded",
                 "worsened"
                 ],
        "patient": ["the plan",
                    "the plot",
                    "the scheme",
                    "the idea",
                    "the procedure",
                    "the organization"]
    },

]


non_inchoative = [
    {
        "verb": ["fed"],
        "patient": ["the cat",
                    "the dog",
                    "the kids"]
    },

    {
        "verb": ["destroyed"],
        "patient": ["the evidence"]
    },

    {
        "verb": ["washed",
                 "cleaned"],
        "patient": ["the floor",
                    "the dishes",
                    "the clothes"]
    },

    {
        "verb": ["mopped",
                 "swept"],
        "patient": ["the floor"]
    },

    {
        "verb": ["ate",
                 "devoured"],
        "patient": ["the apple",
                    "the cake",
                    "the orange"]
    },

    {
        "verb": ["read",
                 "wrote"],
        "patient": ["the book",
                    "the article",
                    "the novel"]
    },

    {
        "verb": ["watched",
                 "saw"],
        "patient": ["the movie",
                    "the play"]
    },

    {
        "verb": ["watered",
                 "harvested"],
        "patient": ["the crops",
                    "the tomatoes",
                    "the oranges"]
    },

    {
        "verb": ["cut",
                 "chopped",
                 "peeled",
                 "sliced",
                 "minced"],
        "patient": ["the apple",
                    "the onion",
                    "the garlic",
                    "the tomato"]
    },

    {
        "verb": ["polished",
                 "buffed",
                 "sanded"],
        "patient": ["the surface",
                    "the brass",
                    "the table",
                    "the counter",
                    "the wood"]
    },

    {
        "verb": ["understood",
                 "solved",
                 "explained",
                 "complicated"],
        "patient": ["the problem",
                    "the dilemma",
                    "the puzzle",
                    "the issue",
                    "the paradox"]
    },

    {
        "verb": ["deposited",
                 "withdrew",
                 "saved",
                 "collected"],
        "patient": ["the cash",
                    "the money",
                    "the funds"]
    },

    {
        "verb": ["recounted",
                 "wrote",
                 "read",
                 "edited",
                 "published",
                 "skimmed"],
        "patient": ["the story",
                    "the book",
                    "the anecdote",
                    "the article",
                    "the chapter",
                    "the letter"]
    },

    {
        "verb": ["admired",
                 "hated",
                 "liked",
                 "loved",
                 "enjoyed"],
        "patient": ["the movie",
                    "the picture",
                    "the painting",
                    "the novel",
                    "the sculpture",
                    "the presentation",
                    "the speech",
                    "the poem"]
    },

    {
        "verb": ["exposed",
                 "revealed",
                 "concealed",
                 "covered up",
                 "uncovered"],
        "patient": ["the secret",
                    "the mistake",
                    "the results",
                    "the outcome",
                    "the mess",
                    "the disaster"]
    },

    {
        "verb": ["saw",
                 "witnessed",
                 "perceived",
                 "glimpsed"],
        "patient": ["the crime",
                    "the theft",
                    "the accident",
                    "the mistake",
                    "the murder"]
    },

    {
        "verb": ["drank",
                 "chugged",
                 "swallowed",
                 "wasted"
                 ],
        "patient": ["the liquid",
                    "the water",
                    "the vodka",
                    "the beer",
                    "the juice"]
    },



]


# out_files = [open("../acceptability_corpus/artificial/verb_classes/inchoative/train.tsv", "w"),
#        open("../acceptability_corpus/artificial/verb_classes/inchoative/test.tsv", "w"),
#        open("../acceptability_corpus/artificial/verb_classes/inchoative/dev.tsv", "w")]
#
# weights = [0.5,0.5,0]

out_files = [open("../acceptability_corpus/artificial/verb_classes/inchoative/all.tsv", "w")]

weights = [1]

np.choice(out_files, p=weights)

for set in inchoative:
    out = np.choice(out_files, p=weights)
    for v in set["verb"]:
        # if out is out_files[1] and random.random() >= 0.8:
        #         out2 = out_files[2]
        # else:
        out2 = out
        for o in set["patient"]:
            # for s in animate:
            s = random.choice(animate)
            out2.write("inch\t%s\t\t%s %s %s .\n" % (1, s, v, o))
            out2.write("inch\t%s\t\t%s %s .\n" % (1, o, v))

for set in non_inchoative:
    out = np.choice(out_files, p=weights)
    for v in set["verb"]:
        # if out is out_files[1] and random.random() >= 0.8:
        #         out2 = out_files[2]
        # else:
        out2 = out
        for o in set["patient"]:
            # for s in animate:
            s = random.choice(animate)
            out2.write("inch\t%s\t\t%s %s %s .\n" % (1, s, v, o))
            out2.write("inch\t%s\t\t%s %s .\n" % (0, o, v))

# out = open("../acceptability_corpus/artificial/all_verbs/inchoative.csv", "w")
#
# for set in inchoative:
# 	for y in set["verb"]:
# 		out.write(y + "," + ",".join([str(x) for x in ["x","x","x",1,0,"x","x",0,0,0,0,0]]) + "\n")
#
# for set in non_inchoative:
# 	for y in set["verb"]:
# 		out.write(y + "," + ",".join([str(x) for x in ["x","x","x",0,1,0,1,0,0,0,0,0]]) + "\n")
#
# out.close()

# sl, sl_noloc, sl_nowith, inch, non_inch, there, no_there, dat, dat_to, dat_do, refl, refl_only
# 0		1			2		3		4		5		6		7		8		9		10		11
