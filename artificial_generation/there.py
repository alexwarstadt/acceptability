"""
(320)
a. VERBS OF EXISTENCE (drawn from various subclasses): blaze, bubble, cling, coexist, correspond, decay, depend, drift, 
dwell, elapse, emanate, exist, fester, float, flow, fly, grow, hide, hover, live, loom, lurk, overspread, persist, 
predominate, prevail, project, protrude, remain, revolve, reside, rise, settle, shelter, smolder, spread, stream, survive, 
sweep, swing, tower, wind, writhe
b. VERBS OF SPATIAL CONFIGURATION: crouch, dangle, hang, kneel, lean, lie, perch, rest, sit, slouch, sprawl, squat, stand, straddle, stretch, swing
c. MEANDER VERBS: cascade, climb, crawl, cut, drop, go, meander, plunge, run, straggle, stretch, sweep, tumble, tum, twist, wander, weave, wind
d. VERBS OF APPEARANCE: accumulate, appear, arise, assemble, awake, awaken, begin, break, burst, dawn, derive, develop, 
emanate, emerge, ensue, evolve, exude, flow, follow, gush, happen, issue, materialize, occur, open, plop, rise, spill, 
steal, stem, supervene, surge
e. ? VERBS OF DISAPPEARANCE: die, disappear, vanish 
f. VERBS OF INHERENTLY DIRECTED MOTION: arrive, ascend, come, descend, drop, enter, fall, go, pass, rise

(321) 
a. A problem developed. b. There developed a problem.

(322)
a. A ship appeared on the horizon. 
b. There appeared a ship on the horizon.

(323)
Definiteness Effect: 
a. There appeared a ship on the horizon. 
b. * There appeared the ship on the horizon.

(324) VERBS OF MANNER OF MOTION (Run and Roll Verbs): amble, climb, crawl, creep, dance, dart, flee, float, fly, gallop, 
head, hobble, hop, hurtle, jump, leap, march, plod, prance, ride, roam, roll, run, rush, sail, shuffle, skip, speed, 
stagger, step, stray, stride, stroll, strut, swim, trot, trudge, walk

(325) Manner and direction of motion specified: 
a. A little boy darted into the room.
b.There darted into the room a little boy.
?? There darted a little boy into the room.
?? Into the room there darted a little boy.

(326) Manner of motion specified but direction of motion not specified: 
a. A little boy ran in the yard 
b. *There ran a little boy in the yard. 
c. ? There ran in the yard a little boy. 
d. ? In the yard there ran a little boy.

Potentially extended uses of certain verbs as verbs of existence: 
a. RuN VERBs: see above 
b. VERBS OF BODY-INTERNAL MOTION: flap, flutter 
c. VERBS OF SOUND EMISSION: beat, boom, chime, ring, rumble, shriek, tick
d. VERBS OF SoUND EXISTENCE: echo, resound, reverberate, sound 
e. VERBS OF LIGHT EMISSION: .flare, flash, flicker, gleam, glimmer, glisten, glitter, scintillate, shimmer, shine, sparkle, twinkle 
f. VERBS OF SUBSTANCE EMISSION: belch, puff, radiate
g. OTHER VERBS: chatter, doze, idle, labor, lounge, preside, reign, sing, sleep, toil, wait, work

(328)
TRANSITIVE VERBS USED IN THE PASSIVE: 
a. VERBS OF PERCEPTION: discern, discover, hear, see 
b. VERBS OF IMAGE CREATION: engrave, imprint, inscribe, paint, scrawl, stamp, tattoo, write
c. VERBS OF PUTriNG: hang, lay, mount, place, pile, stack, suspend, scatter
d. TAPE VERBS: glue, hook, pin, staple, strap e. OTHER VERBS: add, build, display, create, enact, find, show, understand, write

(329) (330)
a. An ancient treasure trove was found in this cave. 
b. There was found in this cave an ancient treasure trove.
TRANSITIVE VERBS (very few): await, confront, cross, enter, follow, reach, seize, take (place/shape), want 
a. Suddenly an ugly old man entered the hall. 
b. Suddenly there entered the hall an ugly old man.

(331) * CHANGE OF STATE VERBS: Given the size of the class, the members are not listed here.

(332)
a. A lot of snow melted on the streets of Chicago. 
b. *There melted a lot of snow on the streets of Chicago.
"""

import random
import numpy.random as np

there = [

# a. VERBS OF EXISTENCE (drawn from various subclasses): bubble, correspond, depend,
# elapse, emanate, fly, hide, lurk, overspread,
# predominate, revolve, rise, shelter, spread, stream, survive,
# sweep, swing, tower, wind, writhe

    {
        "verb": ["blazed",
                 "burned",
                 "crackled",
                 "smoldered"],
        "theme": ["a fire",
                  "a coal",
                  "a log"],
        "loc": ["in the fireplace",
                "in the oven",
                "on the stove"]
    },

    {
        "verb": ["grew",
                 "thrived",
                 "clung"],
        "theme": ["an orchid",
                  "a cactus",
                  "a tree"],
        "loc": ["at the edge of the cliff",
                "on the mountain"]
    },

    {
        "verb": ["decayed",
                 "festered",
                 "rotted",
                 "decomposed"],
        "theme": ["a dead rat",
                  "a forgotten apple",
                  "a strawberry",
                  "an old log"],
        "loc": ["in the gutter",
                "on the forest floor",
                "in the shadows"]
    },

    {
        "verb": ["existed",
                 "remained",
                 "persisted"],
        "theme": ["several concerns",
                  "some hope",
                  "several issues",
                  "some questions"],
        "loc": ["in our minds"]
    },

    # {
    #     "verb": ["drifted",
    #              "floated",
    #              "flowed"],
    #     "theme": ["a boat",
    #               "a leaf",
    #               "a duck"],
    #     "loc": ["on the water",
    #             "across the lake",
    #             "over the dam",
    #             "down the river"]
    # },

    {
        "verb": ["dwelled",
                 "lived",
                 "resided",
                 "settled",
                 "coexisted"],
        "theme": ["two families",
                  "a colony of ants",
                  "some bats",
                  "an indigenous group"],
        "loc": ["at the edge of the cliff",
                "in the house",
                "in the apartment",
                "in the town"]
    },

    {
        "verb": ["loomed",
                 "projected",
                 "protruded",
                 "hovered"],
        "theme": ["a boulder",
                  "a bridge",
                  "a branch"],
        "loc": ["over the edge of the cliff",
                "over the river",
                "above the walkway"]
    },

    # b. VERBS OF SPATIAL CONFIGURATION: lean, lie, slouch,
    # sprawl, straddle, stretch



    {
        "verb": ["crouched",
                 "kneeled",
                 "squatted",
                 "rested",
                 "sat",
                 "stood",
                 "perched"],
        "theme": ["a boy",
                  "a statue",
                  "a dog",
                  "a sparrow"],
        "loc": ["next to the church",
                "in the plaza",
                "on the stoop"]
    },

    {
        "verb": ["dangled",
                 "hung",
                 "swung"],
        "theme": ["a flag",
                  "a sheet",
                  "a clothes line"],
        "loc": ["over the yard",
                "from the tree"]
    },

    # c. MEANDER VERBS: cascade, climb, crawl, cut, drop, go, plunge, run, straggle, stretch,
    # sweep, tumble, tum, wind

    {
        "verb": ["meandered",
                 "twisted",
                 "weaved",
                 "wandered"],
        "theme": ["a path",
                  "a road",
                  "a telephone wire"],
        "loc": ["through the town",
                "up the mountain",
                "over the prairie"]
    },

    # d. VERBS OF APPEARANCE: assemble, awaken, break, dawn,
    # derive, develop, evolve,
    # open, plop, rise, steal, stem, supervene

    # {
    #     "verb": ["accumulated",
    #              "built up"
    #              "materialized"],
    #     "theme": ["a deposit",
    #               "a growth",
    #               "a lump"],
    #     "loc": ["on the ground",
    #             "below the patient's knee",
    #             "under the drain pipe"]
    # },

    {
        "verb": ["emanated",
                 "exuded",
                 "emerged",
                 "gushed",
                 "issued",
                 "spilled"],
        "theme": ["a sticky fluid",
                  "a putrid smell",
                  "a fine mist"],
        "loc": ["from the hole",
                "from the container",
                "from the cave",
                "from the oven"]
    },

    {
        "verb": ["appeared",
                 "burst",
                 "surged"],
        "theme": ["a bright light",
                  "a loud noise",
                  "a giant wave"],
        "loc": ["from the cave",
                "from the room",
                "from the hole"]
    },

    # {
    #     "verb": ["rose",
    #              "arose",
    #              "awoke"],
    #     "theme": ["a creature",
    #               "a small child",
    #               "a dog"],
    #     "loc": ["from under the covers",
    #             "out of a deep sleep"]
    # },

    {
        "verb": ["began",
                 "ensued",
                 "occurred",
                 "followed"],
        "theme": ["a series of mistakes",
                  "a long speech",
                  "an investigation"],
        "loc": ["after the meal",
                "in the next few months"]
    },


    # e. ? VERBS OF DISAPPEARANCE: die, disappear, vanish
    # f. VERBS OF INHERENTLY DIRECTED MOTION: arrive, ascend, come, descend, drop, enter, fall, go, pass, rise


]




no_there = [

    {
        "verb": ["graduated",
                 "flunked out"],
        "theme": ["a student",
                  "a senior"],
        "loc": ["from the university",
                "from college"]
    },

    {
        "verb": ["concluded",
                 "finished",
                 "ended"],
        "theme": ["a play",
                  "a lecture",
                  "a movie"],
        "loc": ["in the morning",
                "in the auditorium",
                "at the theater"
                ]
    },

    {
        "verb": ["cracked",
                 "chipped",
                 "shattered"],
        "theme": ["a glass",
                  "a bowl",
                  "a window"],
        "loc": ["in the yard",
                "all over the ground",
                "on john 's head"]
    },

    {
        "verb": ["closed",
                 "shut",
                 "opened"],
        "theme": ["a door",
                  "a window"],
        "loc": ["on the second floor",
                "down the street",
                "in the next room"]
    },

    {
        "verb": ["crashed",
                 "slammed",
                 "collided",
                 "bumped"],
        "theme": ["a car",
                  "a bicycle",
                  "a bird"],
        "loc": ["into the wall",
                "into the pole",
                "into the window"]
    },

    {
        "verb": ["melted",
                 "froze",
                 "solidified"],
        "theme": ["a glass of water",
                  "a puddle",
                  "a pile of snow"],
        "loc": ["on the floor",
                "on the table",
                "on the streets"]
    },

    {
        "verb": ["halted",
                 "slowed",
                 "accelerated"],
        "theme": ["a car",
                  "a bicycle",
                  "a bus"],
        "loc": ["at the intersection",
                "in front of the school",
                "near the pedestrians"]
    },

    # {
    #     "verb": ["bent",
    #              "folded",
    #              "wrinkled",
    #              "crumpled",
    #              "warped",
    #              "frayed",
    #              "stretched",
    #              "flattened",
    #              "smoothed"
    #              ],
    #     "theme": ["a piece of paper",
    #                 "a sheet",
    #                 "a poster",
    #                 "a sheet of foil"],
    #     "loc": ["in the bedroom",
    #             "on the floor",
    #             ]
    # },

    {
        "verb": ["heated",
                 "cooled",
                 "warmed",
                 "defrosted"
                 ],
        "theme": ["a glass of water",
                    "a plate of food",
                    "a casserole",
                    "a meal"],
        "loc": ["in the oven",
                "in the microwave",
                "on the table"]
    },

    {
        "verb": ["stomped",
                 "stepped",
                 "jumped"],
        "theme": ["a boy",
                  "an elephant",
                  "an exterminator"],
        "loc": ["on a bug",
                "on a button",
                "on a frog"]
    },

    {
        "verb": ["believed",
                 "had faith",
                 "trusted"],
        "theme": ["a cult",
                  "a preacher",
                  "a nice old lady"],
        "loc": ["in god",
                "in aliens",
                "in the goodness of people"]
    },

    {
        "verb": ["yelled",
                 "screamed",
                 "hollered"],
        "theme": ["a teacher",
                  "a parent",
                  "an old man"],
        "loc": ["at the children",
                "at the crowd",
                "at the pedestrians"]
    },

    {
        "verb": ["talked",
                 "lectured",
                 "spoke"],
        "theme": ["a teacher",
                  "a parent",
                  "an old man"],
        "loc": ["to the children",
                "to the crowd",
                "to the pedestrians"]
    },

    {
        "verb": ["played",
                 "juggled",
                 "hollered"],
        "theme": ["a teacher",
                  "a parent",
                  "an old man"],
        "loc": ["at the children",
                "at the crowd",
                "at the pedestrians"]
    },

    {
        "verb": ["agreed",
                 "dealt",
                 "disagreed"],
        "theme": ["a student",
                  "a protester",
                  "a committee"],
        "loc": ["with the politician",
                "with the police officer",
                "with the complaint"]
    },

    {
        "verb": ["looked",
                 "searched"],
        "theme": ["a little boy",
                  "an investigator",
                  "a scientist"],
        "loc": ["for evidence",
                "for treasure",
                "for the hidden children"]
    },

    {
        "verb": ["focused",
                 "concentrated"],
        "theme": ["a student",
                  "an editor"],
        "loc": ["on the book",
                "on the computer",
                "on the puzzle"]
    },

    {
        "verb": ["stuck",
                 "adhered"],
        "theme": ["a piece of paper",
                  "a fly"],
        "loc": ["to the glue",
                "to the wall"]
    },

]



# out_files = [open("../acceptability_corpus/artificial/verb_classes/there/train.tsv", "w"),
#        open("../acceptability_corpus/artificial/verb_classes/there/test.tsv", "w"),
#        open("../acceptability_corpus/artificial/verb_classes/there/dev.tsv", "w")]
#
# weights = [0.5, 0.5, 0]

out_files = [open("../acceptability_corpus/artificial/verb_classes/there/all.tsv", "w")]

weights = [1]

np.choice(out_files, p=weights)

for set in there:
    out = np.choice(out_files, p=weights)
    for v in set["verb"]:
        # if out is out_files[1] and random.random() >= 0.8:
        #         out2 = out_files[2]
        # else:
        out2 = out
        for t in set["theme"]:
            for l in set["loc"]:
                out2.write("there\t%s\t\t%s %s %s .\n" % (1, t, v, l))
                out2.write("there\t%s\t\tthere %s %s %s .\n" % (1, v, l, t))

for set in no_there:
    out = np.choice(out_files, p=weights)
    for v in set["verb"]:
        # if out is out_files[1] and random.random() >= 0.8:
        #         out2 = out_files[2]
        # else:
        out2 = out
        for t in set["theme"]:
            for l in set["loc"]:
                out2.write("there\t%s\t\t%s %s %s .\n" % (1, t, v, l))
                out2.write("there\t%s\t\tthere %s %s %s .\n" % (0, v, l, t))



# out = open("../acceptability_corpus/artificial/all_verbs/there.csv", "w")
#
# for set in there:
# 	for y in set["verb"]:
# 		out2.write(y + "," + ",".join([str(x) for x in ["x","x","x","x","x",1,0,0,0,0,0,0]]) + "\n")
#
# for set in no_there:
# 	for y in set["verb"]:
# 		out2.write(y + "," + ",".join([str(x) for x in ["x","x","x","x","x",0,1,0,0,0,0,0]]) + "\n")
#
#
# out.close()


# sl, sl_noloc, sl_nowith, inch, non_inch, there, no_there, dat, dat_to, dat_do, refl, refl_only
# 0		1			2		3		4		5		6		7		8		9		10		11