import random
import numpy.random as np

"""
(115) Alternating Verbs (characterizations in quotes are from Gropen et al. (1989)): 
a. GIVE VERBS ("verbs that inherently signify acts of giving"): 
    feed, give, lease, lend, loan, pass, pay, peddle, refund, render, rent, repay, sell, serve, trade
b. VERBS OF FUTURE HAVING ("commitments that a person will have something at some later point"): 
    advance, allocate, allot, assign, award, bequeath, cede, concede, extend, grant, guarantee, issue, leave, offer, 
    owe, promise, vote, will, yield
c. BRING AND TAKE ("verbs of continuous causation of accompanied motion in a deictically specified direction"): 
    bring, take
d. SEND VERBS ("verbs of sending"): 
    forward, hand, mail, post, send, ship, slip, smuggle, sneak
e. SLIDE VERBS: 
    bounce, float, roll, slide 
f. CARRY VERBS ("verbs of continuous causation of accompanied motion in some manner"): 
    carry, drag, haul, heave, heft, hoist, kick, lug, pull, push, schlep, shove, tote, tow, tug
g. ? DRIVE VERBS: 
    barge, bus, cart, drive, ferry, fly, row, shuttle, truck, wheel, wire (money)
h. VERBS OF THROWING ("instantaneously causing ballistic motion"; most): 
    bash, bat, bunt, catapult, chuck, flick, fling, flip, hit, hurl, kick, lob, pass, pitch, punt, shoot, shove, 
    slam, slap, sling, throw, tip, toss
i. VERBS OF TRANSFER OF A MESSAGE ("verbs of type of communicated message [differentiated by something like 'illocutionary force']"): 
    ask, cite, ?pose, preach, quote, read, relay, show, teach, tell, write
j. VERBS OF INSTRUMENT OF COMMUNICATION: 
    cable, e-mail, fax, modem, netmail, phone, radio, relay, satellite, semaphore, sign, signal, telephone, telecast, 
    telegraph, telex, wire, wireless

(116)
a. Bill sold a car to Tom. 
b. Bill sold Tom a car.

(117) "Animacy" restriction on double object construction: 
a. Bill sent a package to Tom/London. 
b. Bill sent Tom/* London a package.




"""



both = [

    {
        "verb": ["fed",
                 "served"],
        "patient": ["the casserole",
                    "the meal"],
        "recipient": ["the guests",
                      "the children",
                      "the family"]
    },

    {
        "verb": ["gave",
                 "left",
                 "brought"],
        "patient": ["a gift",
                    "a surprise",
                    "a puppy",
                    "a book",
                    "a plate of food"],
        "recipient": ["the little boy",
                      "his wife",
                      "the family"],
        "subject": ["john"]
    },

    {
        "verb": ["passed",
                 "sent"],
        "patient": ["the salt",
                    "a letter"],
        "recipient": ["the person across the table",
                      "the lawyer"]
    },

    {
        "verb": ["sold",
                 "traded",
                 "peddled",
                 "offered"],
        "patient": ["a book",
                    "a computer",
                    "a car"],
        "recipient": ["the sales clerk",
                      "the customer",
                      "the mother"]
    },

    {
        "verb": ["rented",
                 "leased",
                 "loaned"],
        "patient": ["a car",
                    "an apartment"],
        "recipient": ["the tenant",
                      "the defendant"]
    },

    {
        "verb": ["assigned",
                 "awarded"],
        "patient": ["a project",
                    "a gold star",
                    "a task"],
        "recipient": ["the student",
                      "the employee"]
    },

    {
        "verb": ["allotted",
                 "bequeathed",
                 "ceded",
                 "granted",
                 "left",
                 "promised",
                 "willed",
                 "allocated"],
        "patient": ["a plot of land",
                    "the property",
                    "the art collection",
                    "the mansion"],
        "recipient": ["the client",
                      "the children",
                      "the family lawer"]
    },

    {
        "verb": ["forwarded",
                 "mailed"],
        "patient": ["a letter",
                    "a package",
                    "an email"],
        "recipient": ["the client",
                      "the recipient"]
    },

    {
        "verb": ["shipped",
                 "posted"],
        "patient": ["a letter",
                    "a package"],
        "recipient": ["the client",
                      "the recipient"]
    },

    {
        "verb": ["carried",
                 "schlepped",
                 "lugged",
                 "dragged",
                 "hauled"],
        "patient": ["a package",
                    "the sofa",
                    "the bags"],
        "recipient": ["the family",
                      "the new tenants"]
    },

    {
        "verb": ["chucked",
                 "flung",
                 "tossed",
                 "threw",
                 "tossed",
                 "kicked"],
        "patient": ["the ball",
                    "the stick",
                    "the frisbee"],
        "recipient": ["the goalie",
                      "the other team",
                      "the most valuable player"]
    },

    {
        "verb": ["read"],
        "patient": ["a story",
                    "a book"],
        "recipient": ["the class",
                      "the students"]
    },

    {
        "verb": ["told"],
        "patient": ["a secret",
                    "a story"],
        "recipient": ["the little girl",
                      "the ambassador"]
    },

    {
        "verb": ["taught"],
        "patient": ["a new technique",
                    "a lesson"],
        "recipient": ["the student",
                      "the class"]
    },

    {
        "verb": ["showed"],
        "patient": ["a photo",
                    "a new technique"],
        "recipient": ["the audience",
                      "the attendees"]
    },



]







"""
(118) Non-Alternating to Only: 
a. *Primarily Latinate verbs belonging to some of the semantically plausible classes listed above: 
    address, administer, broadcast, convey, contribute, delegate, deliver, denounce, demonstrate, describe, dictate, 
    dispatch, display, distribute, donate, elucidate, exhibit, express, explain, explicate, forfeit, illustrate, 
    introduce, narrate, portray, proffer, recite, recommend, refer, reimburse, remit, restore, return, sacrifice, 
    submit, surrender, transfer, transport
b. *SAY VERBS ("verbs of communication of propositions and propositional attitudes"): 
    admit, allege, announce, articulate, assert, communicate, confess, convey, declare, mention, propose, recount, 
    repeat, report, reveal, say, state
c. *VERBS OF MANNER OF SPEAKING: 
    babble, bark bawl, bellow, bleat, boom, bray, burble, cackle, call, carol, chant, 
    chatter, chirp, cluck, coo, croak, croon, crow, cry, drawl, drone, gabble, gibber, groan, growl, grumble, grunt, 
    hiss, holler, hoot, howl, jabber, lilt, lisp, moan, mumble, murmur, mutter, purr, rage, rasp, roar, rumble, scream, 
    screech, shout, shriek, sing, snap, snarl, snuffle, splutter, squall, squawk, squeak, squeal, stammer, stutter, 
    thunder, tisk, trill, trumpet, twitter, wail, warble, wheeze, whimper, whine, whisper, whistle, whoop, yammer, yap, 
    yell, yelp, yodel
d. *VERBS OF PUTTING WITH A SPECIFIED DIRECTION: 
    drop, hoist, lift, lower, raise
e. *VERBS OF FULFILLING ("X gives something to Y that Y deserves, needs, or is worthy of"): 
    credit, entrust, furnish, issue, leave, present, provide, serve, supply, trust

"""



dative_only = [

    {
        "verb": ["addressed",
                 "recited"],
        "patient": ["a letter",
                    "a speech"],
        "recipient": ["the audience",
                      "the recipient"]
    },

    {
        "verb": ["explained",
                 "illustrated",
                 "demonstrated"],
        "patient": ["the issue",
                    "the solution"],
        "recipient": ["the client",
                      "the team",
                      "the mechanic"]
    },

    {
        "verb": ["donated",
                 "contributed"],
        "patient": ["a statue",
                    "a hundred dollars"],
        "recipient": ["the museum",
                      "the university"]
    },

    {
        "verb": ["delivered",
                 "conveyed"],
        "patient": ["the message",
                    "the letter"],
        "recipient": ["the client",
                      "the recipient"]
    },

    {
        "verb": ["restored",
                 "returned"],
        "patient": ["the painting",
                    "the car"],
        "recipient": ["its owner",
                      "its place"]
    },

    {
        "verb": ["announced",
                 "declared",
                 "revealed",
                 "reported"],
        "patient": ["the committee 's decision",
                    "the result"],
        "recipient": ["the shareholders",
                      "the investors"]
    },

    {
        "verb": ["admitted",
                 "confessed"],
        "patient": ["the crime",
                    "the error"],
        "recipient": ["the authorities"]
    },

    {
        "verb": ["mentioned",
                 "suggested",
                 "proposed"],
        "patient": ["a solution",
                    "an idea"]
    },

    {
        "verb": ["distributed",
                 "administered"],
        "patient": ["the medications",
                    "the tests"],
        "recipient": ["the children"]
    },

    {
        "verb": ["expressed",
                 "broadcasted"],
        "patient": ["the concern",
                    "the news"]
    }

]








"""
(119) Non-Alternating Double Object Only: 
a. *MISC
    accord, ask, bear, begrudge, bode, cost, deny, envy, flash (a glance), forbid, forgive, guarantee, 
    issue (ticket, passport), refuse, save, spare, strike (a blow), vouchsafe, wish, write (check)
b. *BILL VERBS: 
    bet, bill, charge, fine, mulct, overcharge, save, spare, tax, tip, undercharge, wager
c. *APPOINT VERBS: 
    acknowledge, adopt, appoint, consider, crown, deem, designate, elect, esteem, imagine, mark, nominate, ordain, 
    proclaim, rate, reckon, report, want
d. *DuB VERBS: 
    anoint, baptize, brand, call, christen, consecrate, crown, decree, dub, label, make, name, nickname, pronounce, 
    rule, stamp, style, term, vote
e. * DECLARE VERBS: 
    adjudge, adjudicate, assume, avow, believe, confess, declare, fancy, find, judge, presume, profess, prove, 
    suppose, think, warrant
"""


double_obj_only = [

    {
        "verb": ["cost"],
        "patient": ["a hundred dollars",
                    "an arm and a leg"],
        "recipient": ["John"],
        "subject": ["the television",
                    "the car"]
    },

    {
        "verb": ["envied",
                 "begrudged",
                 "forgave"],
        "patient": ["her success",
                    "her beauty",
                    "her privilege"],
        "recipient": ["Alice"]
    },

    {
        "verb": ["wished"],
        "patient": ["a safe trip",
                    "the best"]
    },

    {
        "verb": ["guaranteed",
                 "accorded",
                 "refused"],
        "patient": ["respect",
                    "a generous reward",
                    "access to the archives"]
    },

    {
        "verb": ["saved"],
        "patient": ["a piece of pie",
                    "a spot",
                    "a thousand dollars"]
    },

    {
        "verb": ["bore"],
        "patient": ["a child",
                    "good tidings"]
    },

    {
        "verb": ["charged",
                 "overcharged",
                 "bet",
                 "tipped",
                 "spared",
                 "fined",
                 "taxed"],
        "patient": ["a thousand dollars",
                    "a large sum",
                    "a week 's salary",
                    "20 pounds"]
    },

    {
        "verb": ["appointed",
                 "elected",
                 "crowned",
                 "pronounced",
                 "made",
                 "dubbed",
                 "named"],
        "patient": ["king",
                    "the leader",
                    "the winner",
                    "the most valuable player",
                    "the CEO"]
    },

    {
        "verb": ["considered",
                 "deemed",
                 "fancied",
                 "judged",
                 "found",
                 "proclaimed",
                 "thought",
                 "labeled",
                 "ruled"],
        "patient": ["a hero",
                    "a loser",
                    "the greatest athlete",
                    "a criminal",
                    "the best",
                    "the most valuable player"]
    },
]

subj_names = ["michael",
    "christopher",
    "jason",
    "nicole",
    "jessica",
    "elizabeth"]

recip_names = ["david",
    "james",
    "john",
    "rebecca",
    "kelly",
    "susan"]


# out_files = [open("../acceptability_corpus/artificial/verb_classes/dative/train.tsv", "w"),
#        open("../acceptability_corpus/artificial/verb_classes/dative/test.tsv", "w"),
#        open("../acceptability_corpus/artificial/verb_classes/dative/dev.tsv", "w")]
#
# weights = [0.5,0.5,0]

out_files = [open("../acceptability_corpus/artificial/verb_classes/dative/all.tsv", "w")]

weights = [1]

np.choice(out_files, p=weights)

for set in both:
    out = np.choice(out_files, p=weights)
    for v in set["verb"]:
        # if out is out_files[1] and random.random() >= 0.8:
        #         out2 = out_files[2]
        # else:
        out2 = out
        for o in set["patient"]:
            if "recipient" not in set:
                r = random.choice(recip_names)
            else:
                r = random.choice(set["recipient"])
            if "subject" not in set:
                s = random.choice(subj_names)
            else:
                s = random.choice(set["subject"])
            out2.write("dat\t%s\t\t%s %s %s to %s .\n" % (1, s, v, o, r))
            out2.write("dat\t%s\t\t%s %s %s %s .\n" % (1, s, v, r, o))

for set in dative_only:
    out = np.choice(out_files, p=weights)
    for v in set["verb"]:
        # if out is out_files[1] and random.random() >= 0.8:
        #         out2 = out_files[2]
        # else:
        out2 = out
        for o in set["patient"]:
            if "recipient" not in set:
                r = random.choice(recip_names)
            else:
                r = random.choice(set["recipient"])
            if "subject" not in set:
                s = random.choice(subj_names)
            else:
                s = random.choice(set["subject"])
            out2.write("dat\t%s\t\t%s %s %s to %s .\n" % (1, s, v, o, r))
            out2.write("dat\t%s\t\t%s %s %s %s .\n" % (0, s, v, r, o))

for set in double_obj_only:
    out = np.choice(out_files, p=weights)
    for v in set["verb"]:
        # if out is out_files[1] and random.random() >= 0.8:
        #         out2 = out_files[2]
        # else:
        out2 = out
        for o in set["patient"]:
            if "recipient" not in set:
                r = random.choice(recip_names)
            else:
                r = random.choice(set["recipient"])
            if "subject" not in set:
                s = random.choice(subj_names)
            else:
                s = random.choice(set["subject"])
            out2.write("dat\t%s\t\t%s %s %s to %s .\n" % (0, s, v, o, r))
            out2.write("dat\t%s\t\t%s %s %s %s .\n" % (1, s, v, r, o))

#
# out = open("../acceptability_corpus/artificial/all_verbs/dative.csv", "w")
#
# for set in both:
# 	for y in set["verb"]:
# 		out2.write(y + "," + ",".join([str(x) for x in [0,0,0,0,"x",0,"x",1,0,0,0,0]]) + "\n")
#
# for set in dative_only:
# 	for y in set["verb"]:
# 		out2.write(y + "," + ",".join([str(x) for x in [0,0,0,0,"x",0,"x",0,1,0,0,0]]) + "\n")
#
# for set in double_obj_only:
# 	for y in set["verb"]:
# 		out2.write(y + "," + ",".join([str(x) for x in [0,0,0,0,"x",0,"x",0,0,1,0,0]]) + "\n")
#
#
# out.close()


# sl, sl_noloc, sl_nowith, inch, non_inch, there, no_there, dat, dat_to, dat_do, refl, refl_only
# 0		1			2		3		4		5		6		7		8		9		10		11
