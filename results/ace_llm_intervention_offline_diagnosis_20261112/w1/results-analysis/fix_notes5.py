p = 'assemble_notes.py'
s = open(p).read()
reps = [
("The four learned arms are not separated from random or from each other on any endpoint. Target risk and local information cannot be ranked.",
 "The ranker and the three LLM arms are not separated from random or from each other beyond chance (1 of 70 pairwise tests over seven endpoints is below 0.05). Target risk and local information cannot be ranked."),
("4. **Single runs are dominated by measurement-noise replicate variance.** Arms do not share noise.",
 "4. **Single runs carry large measurement-noise replicate variance.** Arms do not share noise."),
("\"DPO was worst\" is a statement about the mean of one endpoint, not a robust ordering among the four learned arms and random.",
 "\"DPO was worst\" is a statement about the mean of one endpoint, not a robust ordering among the ranker, the three LLM arms and random. Over all ten pairs among those five arms and seven endpoints (script `a26_weakpairs.py`), 1 of 70 sign-flip tests is below 0.05 (sft has lower error than frozen at cost 12, p 0.043) and 2 are below 0.10, which is what chance alone would give."),
("(`common.py`, `a01_sanity.py` ... `a24_sign2.py`, each with `*_out.txt`;",
 "(`common.py`, `a01_sanity.py` ... `a26_weakpairs.py`, each with `*_out.txt`; table generators `make_tables.py`, `make_tables2.py`, `make_tables3.py` and `assemble_notes.py`;"),
]
for a, b in reps:
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)
open(p, 'w').write(s)
