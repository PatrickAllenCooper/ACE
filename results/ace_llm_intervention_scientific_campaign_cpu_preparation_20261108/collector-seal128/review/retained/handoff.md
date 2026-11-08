# CPU campaign collector upstream seal limit delta

Ready for Leibniz independent review. Standalone source `/private/tmp/ace-cpu-campaign-collector-seal128-th31ah9v/pull_terminal.py`, SHA `3a04ebbe8d20a8954d6dfd749cc224c80067579f290c74e4b7cf991b0a11a721`. Predecessor8904b03d and independent review preserved; GPU41603ce1 reference preserved. Core/child/source freeze/history/CLI unchanged.

Only a dedicated128MiB complete upstream custody-seal reader and two call replacements: accepted training capsule seal and nested failed TOKEN01 capsule seal. Mandatory basename custody-inventory.json, independently supplied64hexpin, sole-link regular file, no symlink, streamed fingerprint, bounded128MiB+1 read, exact size/digest agreement. Ordinary read_metadata32MiB is unchanged. Complete inventory/raw/copy checks following both calls remain unchanged. Entire predecessor AST matches after removing new reader/constant and undoing precisely two call identifiers. No claim that the user-reported unsealed actual training capsule is accepted.

Observed size class72,434,765bytes from user-reported seal4ad14f3985e31c9d3b7bb087055f7477d4db3da47aafaa2f71747c947248f616 motivated this bound; actual seal/capsule was not read. Future larger-than128MiB seals still fail closed.

Three NEW methods passed once (unittest0.284s): exact observed-size fabricatedJSON accepted with matching pin and rejected by ordinary32MiB;128MiB+1 rejected before second buffered read; small legitimate ordinary metadata and wrong pin/basename/symlink/hardlink rejection. No old tests repeated. Exact source snapshots, stdout/stderr, command/cwd, fixture bytes/hashes, intentional tiny alias-fixture metadata and receipt retained at attempt01/. No failed test attempt occurred.

Test source SHA `d1ba79cc743a41e4f666a6a4f98089cb67714d36fd3f08efdea66843603dd68c`. Receipt SHA `f33a33ef0f8a8a16ac89eb3c6799db4859c5cd9075b104e2744559dd774e67d8`. Cost 0.332734childCPU/0.350052167elapsed seconds; peakRSS245432320bytes. These are local fabricated test costs, no allocation/scientific execution.

Large original fixtures72,434,765 and134,217,729bytes remain in attempt01/fixtures with exact inventory in receipt. Nothing was staged or written to repo. A future repository custody package may use a lossless compressed archive with original filehash/size binding; do not add the>100MiB raw fixture to Git or delete originals. No actual collector/build/remote/model/scientific data, no core edits or commit. Resource/capacity and actual acceptance remain separately pending.
