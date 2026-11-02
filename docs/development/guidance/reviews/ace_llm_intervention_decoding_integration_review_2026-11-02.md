# Decoder integration review

Reviewer: Descartes. Scope: the new finite-menu worker, CPU qualifier and diagnostic changes. No tests, models or earlier reviews repeated by the reviewer.

The initial review found one required issue. Importing the Transformers AutoModel factory does not import the actual Qwen3 implementation. Class resolution can occur lazily during weight loading. CPU success would therefore not qualify that import, and the proposed load timing would still combine class import and weights. The review used the pinned Transformers4.57.3 AutoModel implementation to distinguish these operations.

Both workers now explicitly import Qwen3ForCausalLM before model loading, record its class and module origin, and give the import its own timing interval. The CPU qualifier checks the pinned config's architecture and model type without instantiation. The GPU worker retains the original AutoModel weight-loading call and requires its returned type to be the CPU-qualified Qwen3 class before transfer.

No other required defect was found in the initial delta. The review covered project-source closure, authenticated isolated imports, prompt binding, no fallback or preferred-action injection, pre-generation adapter/check persistence, output pins, cleanup and deadlines. The training-loop AST is unchanged.

New local evidence comprises five integration methods, one affected freeze-schema method, a separate isolated-entrypoint rejection fixture and a separate actual Qwen3-class import. The class-import check uses local Transformers4.51.3 andTorch2.14.1. It instantiated zero models and reported CUDA uninitialized. It is not target-runtime qualification. Preparations01–03 preserve exact sources, outputs, costs and scope. No old grammar or process-group family was repeated.

The narrow static recheck closed the lazy-import finding with zero required issues remaining. It verified the CPU config check, GPU exact-class check and unchanged training/masking/recipe. Actual CURC tokenizer/processor/import qualification and actual GPU generation remain open.
