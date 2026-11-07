# CPU threading observations and custody review

Independent reviewer Darwin (`01a1b7f6-2aa8-7403-82f4-481096ab9fdb`) reviewed saved evidence without sourceexecution, models, tests or remotecommands. Required findings: **0**.

All175rawfiles/5519042bytes match hashes and exclusiveexternalcustody. All179files includingmetadata matchbetweencopies. The31transferpins,28diagnosticinputs,25unchangedpredecessorinputs and19committedsourcefilesagree.

Parentimports set C MKL_THREADING_LAYER=INTEL whilePythonmappingremainsunset. Originalhandoffpreservesit. Copied-runtimechild exitsduringTorchimport withstderrbyte-identical toGPU11. ExplicitGNUfreshchildrecordsGNUinbothviews and completesTorch2.9.1+cu128/NumPy2.2.5/SciPy1.15.3 andCPUarithmeticfromcopiedruntime, forceflagabsent. Bothownedgroupsaregonewithoutcleanup errors.

SlurmCOMPLETED0:0,2CPUs×91seconds=182allocatedCPU and0GPU. Closedtotals6852CPU/886GPU, admission10452CPUwith3600preparation. BatchMaxRSS6290032KiB is1424KiBbelow6GiB, notcomfortableheadroom.

The reproduction identifies the mechanism in thisdiagnostic and supports a prospective correction. It doesnotestablishGPU11's unrecorded environment or qualify modelserving/GPUwork/scientificexecution. InterpretationSHA25689142537f2924fef63d13a15b730341f2d2e14b4942c5783cd8b3b690024bc06.
