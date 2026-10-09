# Captured-input descriptor transport review — October9UTC

Volta reviewed only the new post-E2BIG transport and single-attempt operational gate. Earlier five runtime/source findings remain closed; no completed scientific or component review repeated. No executions/edits/messages by reviewer.

Final narrow review: zero required defects. Parent captures inputs into <=16MiB Linux anonymous memfd and applies write/grow/shrink/seal protections before handoff. Child checks regular kind, size, required seals and independent digest before parsing; output/source/manifest pins remain. Extra descriptor travels through pass_fds and closes in parent and child. Large data are removed from argv. Requeue=0/Restarts=0 now required; proposed launcher must use --no-requeue.

18 focused methods pass from unrelated cwd, including fabricated3MBmanifest (larger than real2628435byte manifest), changed digest and missing-seal rejection before qualification, descriptor/bootstrap ordering and existing cleanup cases. Darwin emulates /proc pathname projection and mocks Linux sealing. Actual descriptor transport runs; real Linux seals, target runtime and full scientific replay are NOT qualified. No allocation, installation, target import, checkpoint, fit or response was executed for this preparation.

Source/resource/input proposal is frozen privately and independently hashed. New output is unclaimed, new launcher/freeze identity remains dependent on post-approval claim. User's previous one additional replay was consumed by failed33625983; proposal is unauthorized/unsubmitted. One new15minute1CPU3GiB replay adds.25 reserved core-hour to88.618333/150 with unchanged pins, no installation or fitted rescue. Failure or mismatch is terminal; no implied retries.
