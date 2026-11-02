# Finite intervention grammar review

Independent reviewer: Descartes. Static review of the pure single-prompt implementation, six initial artificial methods and scope document. Zero required defects. No tests or models rerun by the reviewer.

The implementation includes every affordable public action andstop, binds the complete prompt, returns only next tokens of compatible commands, permits EOS only after a complete command, rejects partial or unknown output, and copies its token inputs. Exact round trips preserve command text. Menu ordering does not choose an action. Actual tokenizer, callback, batching and model integration remain open.

Reviewed module SHA256 `d06df6e0afe975d56c101ac57e70e35865b7ad45ccb03f00ebc1ef213220c1ae`. The module is unchanged after review.

One optional suggestion asked for equality with an independent allowed-token-set oracle at every prefix, beyond reachability inclusion. A new focused method implements that oracle using fixed command text. It passed once. The original six methods were not repeated. Preparation02 preserves the added test and measured command:0.035408childCPU seconds,0.041646209elapsed seconds,17219584peakchildRSS bytes. Initial preparation remains preserved with unmetered local command cost. This addition changes test source only and is not a new actual-tokenizer or model qualification.

No scientific intervention utility or automatic safety guarantee is inferred. The deterministic prefix property depends on correct callback integration and a generation length cap at least as long as the longest legal completion.
