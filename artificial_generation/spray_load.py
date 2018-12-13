# Author: Adina Williams

import random
import numpy.random as np
import os

curpath = os.path.abspath(os.curdir)
print "Current path is: %s" % (curpath)

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

sprayload = [
	{
        "verb": ["crammed",
                 "loaded",
                 "stuffed",
                 "packed"
                 ],
        "patient": ["the fruit",
                 "the melons",
                 "the boxes",
                 "the packages",
                 "the cargo"
                 ],
        "location": ["the truck",
                 "the train",
                 "the boat",
                 "the trunk",
                 "the cabinet"
                 ],
        "preposition": ["into"]
	},
	{
        "verb": ["brushed", "dusted"],
        "patient": ["the sand",
                 "the dust"],
        "location": ["the box",
                 "the envelope"],
        "preposition": ["into"]
	},
	{
        "verb": ["injected"],
        "patient": ["the medicine"],
        "location": ["the patient"],
        "preposition": ["into"]
	},
	{
	    "verb": ["crowded",
                 "jammed"],
        "patient": ["the children",
                 "the students"
                 ],
        "location": ["the subway",
                 "the car",
                 "the airplane"
                 ],
        "preposition": ["onto"]
	},
	{
	    "verb": ["splashed",
                 "splattered",
                 "sprayed",
                 "spritzed",
                 "pumped",
                 "sprinkled",
                 "rubbed",
                 "squirted"
                 ],
        "patient": ["paint",
                 "water",
                 "perfume",
                 "alcohol",
                 "fire retardant",
                 "cleanser"
                 ],
        "location": ["the paper",
                 "the painting",
                 "the letter"
                 "the cloth"],
        "preposition": ["onto"]
	},
	{
	    "verb": ["smeared",
                 "slathered",
                 "dabbed",
                 "daubed",
                 "spreaded",
                 "plastered",
                 "smudged"
                 ],
        "patient": ["paint",
                 "putty",
                 "cement",
                 "stain",
                 "glue"
                 ],
        "location": ["the wall",
                 "the fence",
                 "the deck",
                 "the siding"
                 ],
        "preposition": ["onto"]
	},
	{
	    "verb": ["hung",
	    		"draped"
                 ],
        "patient": ["the blanket",
        		 "the towel",
        		 "the bedspread"
                 ],
        "location": ["the bed",
        		"the armchair",
        		 "the couch"
                 ],
        "preposition": ["over"]
	},
	{
	    "verb": ["sowed",
	    		"planted",
	    		"scattered"
                 ],
        "patient": ["corn",
        		 "seeds",
        		 "soybeans"
                 ],
        "location": ["the field",
        		"the yard"
                 ],
        "preposition": ["in"]
	}


]
# leftout verbs: stack, heaped, shower, drizzle, mound, heap, settle, seed, cultivate, strew, swab, stick, string, vest, wash
# note: some of these, heap, stack, seem to need a result ("heaped high")


sprayload_noLoc = [
	{
        "verb": ["anointed",
        		 "bathed",
        		 "smothered",
        		 "soaked",
        		 "coated",
        		 "covered"
        		 ],
        "patient": ["oil",
        		 "perfume"],
        "location": ["the child",
        		 "the patient",
        		 "the king"],
        "preposition": ["on"]
	},
	{
        "verb": ["adorned",
        		 "decorated",
        		 "embellished",
        		 "festooned",
        		 "garnished",
        		 "ornamented",
        		 "ringed",
        		 "wreathed",
        		 "garlanded",
        		 "cluttered",
        		 ],
        "patient": ["flowers",
        		 "gems",
        		 "decorations",
        		 "christmas lights",
        		 "streamers"],
        "location": ["the city",
        		 "the kingdom",
        		 "the palace",
        		 "the theater",
        		 "the staircase"],
        "preposition": ["on"]
	},
	{
        "verb": ["studded",
        		 "emblazoned",
        		 "encrusted",
        		 "dappled",
        		 "inlaid"],
        "patient": ["stars",
        		 "gems"],
        "location": ["the sky", "the heavens"],
        "preposition": ["onto"]
	},
	{
        "verb": ["covered",
        		 "robed",
        		 "smothered",
        		 "swaddled"],
        "patient": ["the blanket",
        		 "the quilt",
        		 "the cloak"],
        "location": ["the baby",
        		 "the queen"],
        "preposition": ["over"]
	},
		{
        "verb": ["shrouded",
        		 "veiled",
        		 "encircled",
        		 "surrounded",
        		 "masked",
        		 "suffused"],
        "patient": ["fog",
        		 "mist",
        		 "smog",
        		 "gloom",
        		 "clouds"],
        "location": ["the coastline",
        		 "the forest",
        		 "the town",
        		 "the village"],
        "preposition": ["over", "into"]
	},
	{
        "verb": ["carpeted",
        		 "blanketed",
        		 "littered"],
        "patient": ["pollen",
        		 "snow",
        		 "pine cones"],
        "location": ["the mountain",
        		 "the valley",
        		 "forest floor"],
        "preposition": ["over","on"]
	},
	{
        "verb": ["doused",
        		 "drenched",
        		 "flooded",
        		 "deluged"],
        "patient": ["water",
        		 "alcohol"],
        "location": ["the garage",
        		 "the basement"],
        "preposition": ["into","in"]
	},
	{
        "verb": ["contaminated",
        		 "dirtied",
        		 "polluted",
        		 "soiled",
        		 "stained",
        		 "tainted"],
        "patient": ["sewage",
        		 "mud",
        		 "waste",
        		 "muck"],
        "location": ["the hospital",
        		 "the clinic",
        		 "the operating room",
        		 "the bathroom"],
        "preposition": ["into"]
	},
	{
        "verb": ["clogged",
        		 "stopped up",
        		 "filled",
        		 "plugged",
        		 "dammed"
        		 ],
        "patient": ["sewage",
        		 "waste",
        		 "muck",
        		 "hair"],
        "location": ["the pipes",
        		 "the drain",
        		 "the spout"],
        "preposition": ["in"]
	},
		{
        "verb": ["bandaged",
        		 "bound",
        		 "choked",
        		 ],
        "patient": ["the rope",
        		 "the cloth"
        		 ],
        "location": ["the criminal",
        		 "the prisoner"
        		 ],
        "preposition": ["around"]
	},
	{
        "verb": ["innundated",
        		 "repopulated",
        		 "replenished",
        		 "saturated",
        		 "staffed"
        		 ],
        "patient": ["the settlers",
        		 "the employees",
        		 "the workers"
        		 ],
        "location": ["the colony",
        		 "the island",
        		 "the business"
        		 ],
        "preposition": ["into"]
	},
	{
        "verb": ["lined",
        		 "trimmed",
        		 "tiled",
        		 "paved",
        		 "edged",
        		 "framed",
        		 "dotted"
        		 ],
        "patient": ["the concrete",
        		 "the pebbles"
        		 "the bricks"
        		 ],
        "location": ["the courtyard",
        		 "the garden",
        		 "the driveway",
        		 "campsite"
        		 ],
        "preposition": ["in"]
	},
	{
        "verb": ["interspersed",
        		 "interleaved",
        		 "interweaved",
        		 "interlaced",
        		 "imbued"
        		 ],
        "patient": ["the lies",
        		 "the fables",
        		 "the falsehoods"
        		 ],
        "location": ["the truth",
        		 "the honesty"
        		 ],
        "preposition": ["into"]
	},
	{
        "verb": ["speckled",
        		 "mottled",
        		 "flecked",
        		 "speckled",
        		 "splotched",
        		 "blotted"
        		 ],
        "patient": ["the paint",
        		 "the dye",
        		 "the varnish",
        		 "the wax"
        		 ],
        "location": ["the art",
        		 "the walls",
        		 "the picture"
        		 ],
        "preposition": ["onto"]
	},
]

# leftout verbs:  bestrew, block, bombard, bound, cloak, deck, endow, enrich, entangle, face, frame, garland, impregnate, infect, inlay, interlard,
#  lard, lash,  pad, plate, riddle, ripple, season, spot, stipple, swathe, vein

sprayload_noWith = [
	{
        "verb": ["poured",
                 "ladled",
                 "dumped",
                 "funnelled",
                 "scooped",
                 "siphoned",
                 "spooned",
                 "poured",
                 "slopped",
                 "sloshed"
                 ],
        "patient": ["water",
                 "the soup",
                 "the broth",
                 "the stock",
                 "the ingredients",
                 "hot chocolate"
                 ],
        "location": ["the bowl",
                 "the bucket",
                 "the cup",
                 "the pot",
                 "the mug"
                 ],
         "preposition" :["into"]
	},
	{
        "verb": ["immersed",
        		 "situated"
                 ],
        "patient": ["the spy",
        		 "the immigrant"
                 ],
        "location": ["the culture",
                 "the language"
                 ],
         "preposition" :["in"]
	},
	{
        "verb": ["hoisted",
        		 "lifted",
        		 "dropped",
        		 "raised",
        		 "lowered",
        		 "placed",
        		 "set",
        		 "put",
                 ],
        "patient": ["the beam",
        		 "the window",
        		 "the curtain",
        		 "the couch",
        		 "the bedframe",
        		 "the materials"
                    ],
        "location": ["the foundation",
                 "the floor",
                 "the ground",
                 "the roof"
                 ],
         "preposition" :["onto"]
	},
	{
        "verb": ["rested",
        		 "stowed",
        		 "stashed",
        		 "pushed",
        		 "installed",
        		 "positioned",
        		 "sat",
        		 "leaned"
                 ],
        "patient": ["the ladder",
        		 "the bench",
        		 "the toolbox",
        		 "the fixture",
        		 "the BBQ"
                    ],
        "location": ["the ledge",
        		 "the roof",
        		 "the floor",
        		 "the deck",
        		 "the wall"
                 ],
         "preposition" :["on"]
	},
	{
        "verb": ["dangled",
        		 "slung",
        		 "suspended",
        		 "mounted",
                 ],
        "patient": ["the trophy",
        		 "the picture",
        		 "the painting",
        		 "the masterpiece"
                 ],
        "location": ["the doorframe",
                 "the floor",
                 "the ground",
                 "the roof",
                 "the stairwell"
                 ],
         "preposition" :["above"]
	},
	{
        "verb": ["hammered",
        		 "pounded",
        		 "rammed",
        		 "banged"
                 ],
        "patient": ["the nail",
        		 "the peg",
        		 "the pin"
                 ],
        "location": ["the drywall",
                 "the floor",
                 "the brick",
                 "the wood"
                 ],
         "preposition" :["into"]
	},
	# {
     #    "verb": ["shovelled",
     #    		 "shook",
     #    		 "scraped",
     #    		 "raked"
     #             ],
     #    "patient": ["the leaves",
     #    		 "the dirt",
     #    		 "the litter"
     #             ],
     #    "location": ["the box",
     #             "the bin",
     #             "the trashcan",
     #             "the bag"
     #             ],
     #     "preposition" :["into"]
	# },
		{
        "verb": ["dripped",
        		 "dribbled",
        		 "spilled"
                 ],
        "patient": ["the milk",
        		 "the soup"
                 ],
        "location": ["the spoon",
        		 "the cup",
        		 "the ladle",
        		 "the spatula"
                 ],
         "preposition" :["from"]
	},
	{
        "verb": ["spewed",
        		 "spurted"
                 ],
        "patient": ["the water",
        		 "the soup"
                 ],
        "location": ["the tube",
        		 "the hose"
                 ],
         "preposition" :["from"]
	},
	{
        "verb": ["shovelled",
        		 "shook",
        		 "scraped",
        		 "raked",
        		 "swept"
                 ],
        "patient": ["the leaves",
        		 "the dirt",
        		 "the litter"
                 ],
        "location": ["the box",
                 "the bin",
                 "the trashcan",
                 "the bag"
                 ],
         "preposition" :["into"]
	},
		{
        "verb": ["coiled",
        		 "curled",
        		 "looped",
        		 "twisted",
        		 "wound"
                 ],
        "patient": ["the yarn",
        		 "the thread",
        		 "the scarf",
        		 "the rope"
                 ],
        "location": ["the pole",
                 "the railing",
                 "the fence",
                 "the hydrant"
                 ],
         "preposition" :["around"]
	},
	{
        "verb": ["spun",
        		"twirled",
        		"whirled"
                 ],
        "patient": ["the baton",
        		 "the hula hoop",
        		 "the flag"
                 ],
        "location": ["the air",
        		 "the sky",
        		 "the breeze",
        		 "the wind"
                 ],
         "preposition" :["in"]
	},
	{
        "verb": ["arranged",
        		 "laid",
        		 "perched",
        		 "stood"
                 ],
        "patient": ["the gnomes",
        		 "the ornaments",
        		 "the pinwheels"
                 ],
        "location": ["the porch",
        		 "the lawn",
        		 "the grass"
                 ],
         "preposition" :["on"]
	},
]

# leftout verbs:

# lodge, perch, stand, channel, dip, squeeze, squish, squash, tuck, wad, wedge, wipe, wring, roll



out_files = [open("../acceptability_corpus/artificial/verb_classes/spray_load/train.tsv", "w"),
       open("../acceptability_corpus/artificial/verb_classes/spray_load/test.tsv", "w"),
       open("../acceptability_corpus/artificial/verb_classes/spray_load/dev.tsv", "w")]

weights = [0.5,0.5,0]

np.choice(out_files, p=weights)

for set in sprayload:
	out = np.choice(out_files, p=weights)
	for v in set["verb"]:
		if out is out_files[1] and random.random() >= 0.8:
				out2 = out_files[2]
		else:
			out2 = out
		for o in set["patient"]:
			for l in set["location"]:
				for p in set["preposition"]:
					s = random.choice(animate)
					out2.write("spray\t%s\t\t%s %s %s %s %s .\n" % (1, s, v, o, p, l))
					out2.write("spray\t%s\t\t%s %s %s %s %s .\n" % (1, s, v, l, "with", o))

for set in sprayload_noLoc:
	out = np.choice(out_files, p=weights)
	for v in set["verb"]:
		if out is out_files[1] and random.random() >= 0.8:
				out2 = out_files[2]
		else:
			out2 = out
		for o in set["patient"]:
			for l in set["location"]:
				for p in set["preposition"]:
					s = random.choice(animate)
					out2.write("spray\t%s\t\t%s %s %s %s %s .\n" % (0, s, v, o, p, l))
					out2.write("spray\t%s\t\t%s %s %s %s %s .\n" % (1, s, v, l, "with", o))

for set in sprayload_noWith:
	out = np.choice(out_files, p=weights)
	for v in set["verb"]:
		if out is out_files[1] and random.random() >= 0.8:
				out2 = out_files[2]
		else:
			out2 = out
		for o in set["patient"]:
			for l in set["location"]:
				for p in set["preposition"]:
					s = random.choice(animate)
					out2.write("spray\t%s\t\t%s %s %s %s %s .\n" % (1, s, v, o, p, l))
					out2.write("spray\t%s\t\t%s %s %s %s %s .\n" % (0, s, v, l, "with", o))

# out = open("../acceptability_corpus/artificial/all_verbs/spray_load.csv", "w")
#
# for set in sprayload:
# 	for y in set["verb"]:
# 		out2.write(y + "," + ",".join([str(x) for x in [1,0,0,"x","x","x","x",0,0,0,0,0]]) + "\n")
#
# for set in sprayload_noLoc:
# 	for y in set["verb"]:
# 		out2.write(y + "," + ",".join([str(x) for x in [0,1,0,"x","x","x","x",0,0,0,0,0]]) + "\n")
#
# for set in sprayload_noWith:
# 	for y in set["verb"]:
# 		out2.write(y + "," + ",".join([str(x) for x in [0,0,1,"x","x","x","x",0,0,0,0,0]]) + "\n")
#
# out.close()

# sl, sl_noloc, sl_nowith, inch, non_inch, there, no_there, dat, dat_to, dat_do, refl, refl_only
# 0		1			2		3		4		5		6		7		8		9		10		11













