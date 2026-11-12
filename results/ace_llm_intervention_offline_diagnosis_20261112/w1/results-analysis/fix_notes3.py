p = 'assemble_notes.py'
s = open(p).read()
reps = [
("frozen (observe only) is the mirror image: smallest q0 error, largest error at X = -1 in chains.",
 "frozen (observe only) is the mirror image: the smallest q0 error in chains and colliders, and in chains more than four times dpo's error at X = -1 (10.4 against 2.4)."),
("The excess is in the X = 0 and X = +1 queries (7.9 against 1.9), which need the slope that data at X = -1 cannot give.",
 "The excess is in the X = 0 and X = +1 queries (7.9 against 2.0), which need the slope that data at X = -1 cannot give."),
]
for a, b in reps:
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)
open(p, 'w').write(s)
