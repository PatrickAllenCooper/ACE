# CPU service preflight02 independent results review

November 7, 2026. Reviewer Socrates (01a1b7d8-0671-7fc1-a98d-0d08307d2631). Zero required findings. Accepted scope is completed CPU import and tokenization preflight only.

All194rawfiles/7,665,919bytes match the fresh remote inventory. All198custodyfiles/7,747,749bytes, including collector records, match exclusive external originals. All28inputpins and17committedsourcebindings reconcile implementationcddb7eb73da0ec110f13412e1587cfa8ed271147 and launchb4f77fda4c72e3bc40d7bbfc892c32b8d8ec4d8b.

Both fresh sessions match the pinned stage/archive receipts, six provenance sources, captured project closure and observed runtime paths. ActualPython3.10.19, Torch2.9.1+cu128,NumPy2.2.5,SciPy1.15.3,Transformers4.57.3,safetensors0.7.0,huggingface_hub0.36.0,tokenizers0.22.1 andJinja2 3.1.6 match. NumPy metadata default2.2.6 remains distinct from imported2.2.5. Historical cache differences remain unexplained.

Both sessions preserve identical complete token records for three public fixtures, with613/4341/7343prompttokens and3/9/9legalcompletions. Input hashes, counts and context limits reconcile. Children returned successfully and owned groups were gone after cleanup. No weights, adapter deserialization, explicitCUDA calls, training updates or environment responses occurred.

SlurmCOMPLETED0:0,1CPU/3840MiB/86seconds/zeroGPU onucb736_asc2. ClosedCPU6470 plus unchanged3600preparation gives10070/57600CPU admission. GPU836/7200unchanged. BatchMaxRSS3929884KiB is only2276KiB below requested memory. Its aggregate/cache attribution and comfortable headroom are not established. OtherCPU/RSS/elapsed measures overlap.

This acceptance excludes GPU inference, adapter execution, complete model-service qualification, scientific execution and intervention-efficiency claims. Review used saved records only, with no project-source execution, runtime imports, model loads, test/replay reruns, remote calls, archive hashing or edits.
