# Scientific CPU campaign monitor R1 — focused independent recheck

**R1 closed; 0 required findings in this delta.** Corrected monitor SHA256 `cb63691494e51fd200b1d40105794126fc7cd1986e7521a1d667836225fd0012` matches the frozen producer's literal output path. Scope is this path correction only; no builder/launcher core review was repeated and no execution or acceptance is authorized.

The monitor differs from predecessor `354cab58ecf5416293ee8464f133906d4cd05e6f080e47ff5a8160a0e93e0b5e` by exactly one byte-string replacement on line51: `cpu/worker/artifacts/campaign/bundle.json` → `cpu/worker/artifacts/cpu-phase/campaign-bundle.json`. The corrected marker occurs exactly once and the obsolete marker is absent. Builder, launcher and argument map are byte-identical to their preserved predecessors.

The frozen source chain is literal: coordinator build_packet.py:348 supplies `cpu_output=ROOT+'/cpu'`; inherited prepare_llm_intervention_campaign_cpu.py:189–193 derives `worker/artifacts/cpu-phase` from that output; scientific prepare_llm_intervention_scientific_campaign_cpu.py:329 sets `out=loc['phase']`, and line359 saves `out/'campaign-bundle.json'`. Line525 reads that same path. Producer bytes match committed sourcea9346022f145cc29439325b6f291597b2e6ec037, respectively `850ca3d4d7ac4799f68d74bd2d0b1196d10809751a4bdf878633122fc6a9c2c4` and `ba439d819f49f6832c97181e4ff727c4813f29031e7b67f8085670f845b0f893`. No producer callable was imported or executed.

Read the author's one NEW focused AST regression and original stdout/stderr/receipt without rerunning it. Receipt `9af1ae617602cb638ec2d3693d8bc3fa031a5b6117bb835e1219ab71816ce46f` reports return0, one method,0.0465385s wall/0.042706s childCPU; retained test and monitor snapshots match the handed-off files. Test source pin is `5c0aa54d0fdddc0d158081509f7ce80afa429522ff703c3a8113bae450e60d69`.

Initial Godel review `94dee1c33ea357baac5d77d47f17e454e41340b2f52e510befecb6da497babae` and original sources remain preserved and the full predecessor preservation map matches. The current original review already contains a focused R1 closure; it was left unchanged. This separate requested recheck adds no broader component findings or scientific claims.

Evidence snapshots: `/tmp/ace-cpu-campaign-monitor-r1-recheck-a09k1com`. Independent recheck receipt SHA256 `ceab7a80f6f99ca95c736e1758c3ac06dad1b19657c0e2f9f39a825fdc65213b`. Static byte/AST/read-only Git inspection only; no tests, project imports, remote/model work, packet build, source edit or commit. Concrete packet and actual-result gates remain separate.
