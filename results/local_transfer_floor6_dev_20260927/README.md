# Transfer allocation floor: six samples per node

Post hoc development diagnostic paired with the five-sample floor. The first four samples per node nominate mechanisms; every node receives two more samples, leaving 20 of the exact 200 target examples for nomination-prioritized allocation. This protects broad coverage at the expense of concentrated acquisition. It uses the same 12 reused systems, 72 settings, 2,560-example source library, and held-out test panel as the four-sample and five-sample diagnostics.

For family changes k=1/3/10, changed-node adaptive-warm error versus uniform warm is 0.220/0.188/0.684; untouched-node ratio is 1.006/1.033/1.063. The protected switch versus adaptive warm gives changed-node ratios 0.985/0.436/0.585. The k=1 switch still fails its ≤0.8 gate; the k=10 untouched-node ratio remains above 1.05. Neither candidate satisfies the complete development gate. The broader floor is a useful damage-control clue, not a promoted architecture.

The 2,160 node rows and 12 summary rows passed complete-receipt SHA-256 and exact 200-example budget validation. The default four-sample implementation reproduced the archived metrics byte for byte. No CURC job or model API call was used.
