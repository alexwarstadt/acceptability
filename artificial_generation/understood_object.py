# Author: Adina Williams

import random
import numpy.random as np
import os

curpath = os.path.abspath(os.curdir)
print "Current path is: %s" % (curpath)

animate = [{"name":"michael", "pronoun":"his", "refl":"himself"},
    {"name":"christopher", "pronoun":"his", "refl":"himself"},
    {"name":"jason", "pronoun":"his", "refl":"himself"},
    {"name":"david", "pronoun":"his", "refl":"himself"},
    {"name":"james", "pronoun":"his", "refl":"himself"},
    {"name":"nicole", "pronoun":"her", "refl":"herself"},
    {"name":"jessica", "pronoun":"her","refl":"herself"},
    {"name":"elizabeth", "pronoun":"her", "refl":"herself"},
    {"name":"rebecca", "pronoun":"her","refl":"herself"},
    {"name":"kelly", "pronoun":"her","refl":"herself"}
    ]

# Sorry that's an ugly dictionary, blech

bodyPart = [
	{
        "verb": ["blinked",
                 "squinted",
                 "winked"
                 ],
        "patient": ["eyes"
                 ]
	},
    {
        "verb": ["clapped",
                 "waved"
                 ],
        "patient": ["hands"
                 ]
    },
    {
        "verb": ["nodded"
                 ],
        "patient": ["head"
                 ]
    },
    {
        "verb": ["pointed"
                 ],
        "patient": ["fingers"
                 ]
    },
    {
        "verb": ["shrugged"
                 ],
        "patient": ["shoulders"
                 ]
    },
    {
        "verb": ["flexed"
                 ],
        "patient": ["muscles"
                 ]
    },
    {
        "verb": ["brushed",
                 "flossed"
                 ],
        "patient": ["teeth",
                 "molars"
                 ]
    },
    {
        "verb": ["shaved"
                 ],
        "patient": ["beard",
                 "legs",
                 "back",
                 "face",
                 "armpits"
                 ]
    },
    {
        "verb": ["washed"
                 ],
        "patient": ["hands",
                 "face",
                 "back",
                 "legs",
                 "arms"
                 ]
    }
]

bodyPartRequired = [
    {
        "verb": ["bared",
                 "gnashed",
                 "ground",
                 "flashed",
                 "showed",
                 "clenched"
                 ],
        "patient": ["teeth",
                 "molars",
                 "incisors"
                 ]
    },
    {
        "verb": ["closed",
                 "crossed",
                 "rolled",
                 "rubbed",
                 "opened"
                 ],
        "patient": ["eyes"
                 ]
    },
    {
        "verb": ["puckered",
                 "pursed",
                 "smacked"
                 ],
        "patient": ["lips"
                 ]
    },
    {
        "verb": ["shuffled"
                 ],
        "patient": ["feet"
                 ]
    },
    {
        "verb": ["snapped"
                 ],
        "patient": ["fingers"
                 ]
    },
    {
        "verb": ["trimmed"
                 ],
        "patient": ["beard",
                 "bangs"
                 ]
    },
    {
        "verb": ["arched",
                 "craned",
                 "flexed",
                 "rubbed"
                 ],
        "patient": ["neck"
                 ]
    },
    {
        "verb": ["batted",
                 "fluttered",
                 "twitched"
                 ],
        "patient": ["eyelashes"
                 ]
    },
    {
        "verb": ["crooked",
                 "wagged",
                 "rubbed"
                 ],
        "patient": ["finger"
                 ]
    },
    {
        "verb": ["wagged"
                 ],
        "patient": ["tail"
                 ]
    },
    {
        "verb": ["blew",
                 "twitched",
                 "wiggled",
                 "wrinkled"
                 ],
        "patient": ["nose"
                 ]
    },
    {
        "verb": ["hung",
                 "raised",
                 "rubbed",
                 "shook",
                 "cocked"
                 ],
        "patient": ["head"
                 ]
    },
    {
        "verb": ["opened",
                 "raised",
                 "rubbed",
                 "shook",
                 "wrung"
                 ],
        "patient": ["hands"
                 ]
    },
    {
        "verb": ["twitched",
                 "waggled",
                 "wiggled",
                 "flapped"
                 ],
        "patient": ["ears",
                 "wings"
                 ]
    },
    {
        "verb": ["crossed",
                 "flapped",
                 "stretched",
                 "raised",
                 "shook",
                 "folded"
                 ],
        "patient": ["arms",
                 "legs"
                 ]
    },
    {
        "verb": ["raised",
                 "knit"
                 ],
        "patient": ["eyebrows"
                 ]
    },
    {
        "verb": ["clenched",
                 "shook"
                 ],
        "patient": ["fist"
                 ]
    },
    {
        "verb": ["arched"
                 ],
        "patient": ["back",
                 "spine"
                 ]
    },
    {
        "verb": ["clicked"
                 ],
        "patient": ["heels",
                 "tongue"
                 ]
    },
    {
        "verb": ["bobbed",
                 "braided",
                 "combed",
                 "conditioned",
                 "crimped",
                 "cropped",
                 "curled",
                 "cut",
                 "dyed",
                 "lathered",
                 "parted",
                 "permed",
                 "plaited",
                 "rinsed",
                 "set",
                 "shampooed",
                 "teased",
                 "trimmed",
                 "waved",
                 "plucked"
                 ],
        "patient": ["hair"
                 ]
    },
    {
        "verb": ["clipped",
                 "manicured",
                 "trimmed",
                 "filed"
                 ],
        "patient": ["nails",
                 "toenails"
                 ]
    },
    {
        "verb": ["coldcreamed",
                 "powdered",
                 "rinsed",
                 "towelled",
                 "rouged"
                 ],
        "patient": ["face",
                 "cheeks"
                 ]
    },
    {
        "verb": ["plucked",
                 "shaped"
                 ],
        "patient": ["eyebrows"
                 ]
    },
    {
        "verb": ["soaped",
                 "towelled",
                 "lathered"
                 ],
        "patient": ["face"
                 "hands"
                 ]
    },
    {
        "verb": ["bumped",
                 "burned",
                 "broke",
                 "bruised",
                 "cut",
                 "fractured",
                 "hurt",
                 "injured",
                 "strained",
                 "scalded"
                 ],
        "patient": ["hand",
                 "foot",
                 "leg",
                 "arm",
                 "shoulder",
                 "finger"
                 ]
    },
    {
        "verb": ["ruptured"
                 ],
        "patient": ["spleen",
                 "appendix"
                 ]
    },
    {
        "verb": ["bit",
                 "split"
                 ],
        "patient": ["lip"
                 ]
    },
    {
        "verb": ["sprained",
                 "turned",
                 "twisted"
                 ],
        "patient": ["ankle"
                 ]
    },
    {
        "verb": ["chipped",
                 "broke",
                 "pulled"
                 ],
        "patient": ["tooth"
                 ]
    },
    {
        "verb": ["nicked"
                 ],
        "patient": ["chin",
                 "leg"
                 ]
    },
    {
        "verb": ["pricked"
                 ],
        "patient": ["finger"
                 ]
    },
    {
        "verb": ["pulled",
                 "strained"
                 ],
        "patient": ["muscle"
                 ]
    }
]

# both dressed and dressed herself are good
bodypartReflexive= [
    {
        "verb": ["dressed",
                 "bathed",
                 "changed",
                 "dress",
                 "shaved",
                 "undressed",
                 "stripped",
                 "washed",
                 "rinsed"
                 ]
    },
]

# *cut but cut herself is good
bodypartReflexiveOnly= [
    {
        "verb": ["cut",
                "sliced",
                "powdered",
                "bit",
                "bumped",
                "burned",
                "hurt",
                "injured",
                "scalded"
        ]
    }

]
# disrobed is good but *disrobed herself
bodypartNoReflexive= [
    {
        "verb": ["disrobed",
                 "exercised",
                 "preened",
                 "primped",
                 "groomed",
                 "flossed"
        ]
     }
]

# *brushed and *brushed herself
bodypartReflexiveNoNothing= [
    {
        "verb": ["brushed",
                 "bobbed",
                 "braided",
                 "clipped",
                 "brushed",
                 "coldcreamed",
                 "combed",
                 "crimped",
                 "cropped",
                 "dyed",
                 "filed",
                 "manicured",
                 "parted",
                 "permed",
                 "plaited",
                 "plucked",
                 "rouged",
                 "set",
                 "soaped",
                 "towelled",
                 "trimmed",
                 "waved",
                 ]
    },
]

# x and y verbed, x verbed y both ok
ReciprocalPlurSubj=[
    {
        "verb": ["courted",
                 "cuddled",
                 "dated",
                 "divorced",
                 "embraced",
                 "hugged",
                 "kissed",
                 "married",
                 "nuzzled",
                 "passed",
                 "petted",
                 "battled",
                 "boxed",
                 "fought",
                 "consulted",
                 "debated",
                 "met",
                 "bantered",
                 "bargained",
                 "bickered",
                 "touch"
        ]
    }
]

# x and y verbed, *x verbed y

NoReciprocalPlurSubj=[
    {
        "verb": ["agreed",
                 "argued",
                 "bantered",
                 "bargained",
                 "bickered",
                 "brawled",
                 "clashed",
                 "coexisted",
                 "collaborated",
                 "collided",
                 "commiserated",
                 "communicated",
                 "competed",
                 "concurred",
                 "confabulated",
                 "conflicted",
                 "consorted",
                 "cooperated",
                 "corresponded",
                 "differed",
                 "disagreed",
                 "dissented",
                 "duelled",
                 "eloped",
                 "feuded",
                 "flirted",
                 "haggled",
                 "hobnobbed",
                 "joked",
                 "jousted",
                 "mated",
                 "necked",
                 "negotiated",
                 "paired off",
                 "plotted",
                 "quarreled",
                 "quibbled",
                 "rendezvoused",
                 "scuffled",
                 "sparred",
                 "spooned",
                 "squabbled",
                 "struggled",
                 "tussled",
                 "wrangled",
                 "wrestled",
                 "spoke",
                 "talked",
                 "argued",
                 "chatted",
                 "chattered",
                 "chitchatted",
                 "conferred",
                 "conversed",
                 "gabbed",
                 "gossip",
                 "rapped",
                 "schmoozed",
                 "yakked"
        ]
    }
]

# *x and y verbed, x verbed y is ok
NoReciprocalPlur=[
    {
        "verb": ["hit",
                 "missed"
        ]
    }
]

# leftout verbs: beat (feet), drum (finger), flick (finger), hunch (shoulders), kick, stamp (foot), toss (mane), tum (head), twiddle (thumbs),
# wiggle (hips), wrinkle (forehead), talc (body), powder (nose), rinse (mouth), trim (beard)



out_files = [open("../acceptability_corpus/artificial/verb_classes/understood/train.tsv", "w"),
       open("../acceptability_corpus/artificial/verb_classes/understood/test.tsv", "w"),
       open("../acceptability_corpus/artificial/verb_classes/understood/dev.tsv", "w")]


weights = [0.5,0.5,0]
#np.choice(out_files, p=weights)

for set in bodyPart:
    out = np.choice(out_files, p=weights)
    for v in set["verb"]:
        if out is out_files[1] and random.random() >= 0.8:
                out2 = out_files[2]
        else:
            out2 = out
        for o in set["patient"]:
            # for s in animate:
		     s = random.choice(animate)
		     out2.write("u_obj\t%s\t\t%s %s %s %s .\n" % (1, s["name"], v, s["pronoun"], o))
		     out2.write("u_obj\t%s\t\t%s %s .\n" % (1, s["name"], v))

for set in bodyPartRequired:
    out = np.choice(out_files, p=weights)
    for v in set["verb"]:
        if out is out_files[1] and random.random() >= 0.8:
                out2 = out_files[2]
        else:
            out2 = out
        for o in set["patient"]:
            # for s in animate:
             s = random.choice(animate)
             out2.write("u_obj\t%s\t\t%s %s %s %s .\n" % (1, s["name"], v, s["pronoun"], o))
             out2.write("u_obj\t%s\t\t%s %s .\n" % (0, s["name"], v))

for set in bodypartReflexive:
    for v in set["verb"]:
        out = np.choice(out_files, p=weights)
        if out is out_files[1] and random.random() >= 0.8:
                out2 = out_files[2]
        else:
            out2 = out
        s = random.choice(animate)
        out2.write("u_obj\t%s\t\t%s %s %s .\n" % (1, s["name"], v, s["refl"]))
        out2.write("u_obj\t%s\t\t%s %s .\n" % (1, s["name"], v))

for set in bodypartReflexiveOnly:
    for v in set["verb"]:
        out = np.choice(out_files, p=weights)
        if out is out_files[1] and random.random() >= 0.8:
                out2 = out_files[2]
        else:
            out2 = out
        s = random.choice(animate)
        out2.write("u_obj\t%s\t\t%s %s %s .\n" % (1, s["name"], v, s["refl"]))
        out2.write("u_obj\t%s\t\t%s %s .\n" % (0, s["name"], v))


for set in bodypartNoReflexive:
    for v in set["verb"]:
        out = np.choice(out_files, p=weights)
        if out is out_files[1] and random.random() >= 0.8:
                out2 = out_files[2]
        else:
            out2 = out
        s = random.choice(animate)
        out2.write("u_obj\t%s\t\t%s %s %s .\n" % (0, s["name"], v, s["refl"]))
        out2.write("u_obj\t%s\t\t%s %s .\n" % (1, s["name"], v))


for set in bodypartReflexiveNoNothing:
    for v in set["verb"]:
        out = np.choice(out_files, p=weights)

        if out is out_files[1] and random.random() >= 0.8:
                out2 = out_files[2]
        else:
            out2 = out
        s = random.choice(animate)
        out2.write("u_obj\t%s\t\t%s %s %s .\n" % (0, s["name"], v, s["refl"]))
        out2.write("u_obj\t%s\t\t%s %s .\n" % (0, s["name"], v))

for set in ReciprocalPlurSubj:
    for v in set["verb"]:
        out = np.choice(out_files, p=weights)
        if out is out_files[1] and random.random() >= 0.8:
                out2 = out_files[2]
        else:
            out2 = out
        s = np.choice(animate,2, replace=False)
        out2.write("u_obj\t%s\t\t%s %s %s .\n" % (1, s[1]["name"], v, s[0]["name"]))
        out2.write("u_obj\t%s\t\t%s %s %s %s .\n" % (1, s[1]["name"], "and", s[0]["name"], v))

for set in NoReciprocalPlurSubj:
    for v in set["verb"]:
        out = np.choice(out_files, p=weights)
        if out is out_files[1] and random.random() >= 0.8:
                out2 = out_files[2]
        else:
            out2 = out
        s = np.choice(animate,2, replace=False)
        out2.write("u_obj\t%s\t\t%s %s %s .\n" % (0, s[1]["name"], v, s[0]["name"]))
        out2.write("u_obj\t%s\t\t%s %s %s %s .\n" % (1, s[1]["name"], "and", s[0]["name"], v))


for set in NoReciprocalPlur:
    for v in set["verb"]:
        out = np.choice(out_files, p=weights)
        if out is out_files[1] and random.random() >= 0.8:
                out2 = out_files[2]
        else:
            out2 = out
        s = np.choice(animate,2, replace=False)
        out2.write("u_obj\t%s\t\t%s %s %s .\n" % (1, s[1]["name"], v, s[0]["name"]))
        out2.write("u_obj\t%s\t\t%s %s %s %s .\n" % (0, s[1]["name"], "and", s[0]["name"], v))



# out = open("../acceptability_corpus/artificial/all_verbs/understood_object.csv", "w")
#
# for x in [bodyPart]:
#     for set in x:
#         for y in set["verb"]:
#             out2.write(y + "," + ",".join([str(x) for x in [0,0,0,"x","x","x","x",0,"x",0,1,0]]) + "\n")
#
# for x in [bodyPartRequired]:
#     for set in x:
#         for y in set["verb"]:
#             out2.write(y + "," + ",".join([str(x) for x in [0,0,0,"x","x","x","x",0,"x",0,0,1]]) + "\n")
#
# out.close()

# sl, sl_noloc, sl_nowith, inch, non_inch, there, no_there, dat, dat_to, dat_do, refl, refl_only
# 0		1			2		3		4		5		6		7		8		9		10		11
