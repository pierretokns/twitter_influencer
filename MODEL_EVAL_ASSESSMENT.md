# Local Model Evaluation Assessment

Last updated: 2026-05-26

## Scope

This report summarizes the current local CPU model evaluation for the Hetzner VM
running the Brandon news-summary migration. It is based on:

- `output_data/model_bench_quality_v2_summary.json`
- `output_data/model_bench_expanded_v1_summary.json`
- `output_data/model_bench_quality_nemotron_summary.json`
- `output_data/model_bench_expanded_nemotron_summary.json`
- VM smoke tests for Mistral Small 3.2

## Current Per-Role Choices

| Role | Current best base model(s) | Fine-tune needed? | Notes |
| --- | --- | --- | --- |
| General brief generation | `LiquidAI/LFM2-2.6B`, `gemma-4-E2B` | Not initially | Base models pass expanded brief cases, but prompts should enforce citations and second-order insight. |
| Finance synthesis | `gemma-4-E2B`, fallback `LFM2-1.2B` | Not initially | Needs retrieval to surface finance context. Add more finance evals before training. |
| Simple missing-source refusal | `LFM2-2.6B`, `gemma-4-E2B`, `Nemotron 3 Nano` | Not initially | Base models can refuse absent facts. |
| Nuanced insufficient-proof refusal | No current base model passes | Yes, or deterministic verifier | All tested models failed the conflicting/insufficient-proof case. Build hard negatives and verifier rules. |
| Structured JSON/XML boundaries | `Nemotron 3 Nano`, then `Phi-4-mini` | Not first | Use constrained decoding/schema validation/retry before fine-tuning. |
| Citation discipline | `Nemotron 3 Nano`, `LFM2-1.2B`, `gemma-4-E2B` | Not for format | Keep deterministic citation verifier; fine-tune only if verifier-guided prompting remains weak. |
| RAG/webchat over retrieved sources | `LiquidAI/LFM2-2.6B` | Not initially | Latest VM RAG run: 100.0%, passed 3/3 gated cases. Gemma E2B scored 76.7%; NVIDIA Nano scored 66.7%. |

## Comparative Slice Scores

Expanded model-only benchmark:

| Slice | Model | Score | Pass | Avg sec | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| Brief | `LFM2-2.6B` | 80.0% | 2/2 | 52.1 | Tied with Gemma; faster. |
| Brief | `gemma-4-E2B` | 80.0% | 2/2 | 61.8 | Tied; weaker second-order/source details in one case. |
| Brief | `LFM2-1.2B` | 70.0% | 2/2 | 43.7 | Cheaper fallback. |
| Finance | `gemma-4-E2B` | 90.0% | 2/2 | 52.2 | Best model-only finance synthesis. |
| Finance | `LFM2-1.2B` | 80.0% | 2/2 | 23.8 | Fast fallback. |
| Finance | `Phi-4-mini` | 80.0% | 2/2 | 54.4 | Structured-capable fallback. |
| Refusal | `LFM2-2.6B` | 80.0% | 1/2 | 14.6 | All models failed nuanced conflicting-source case. |
| Refusal | `gemma-4-E2B` | 80.0% | 1/2 | 31.4 | Same refusal limitation. |
| Refusal | `gemma-4-E4B` | 80.0% | 1/2 | 51.0 | Same refusal limitation. |
| Structured unconstrained | `NVIDIA Nano` | 75.0% | 2/2 | 98.9 | Best prompt-only structured candidate, but still needs schema constraints. |
| Structured unconstrained | `Phi-4-mini` | 55.0% | 1/2 | 65.8 | JSON fallback only. |
| Structured unconstrained | `Qwen3.6` | 55.0% | 1/2 | 284.2 | Too slow for this VM unless quality is uniquely needed. |
| Structured constrained JSON | `Phi-4-mini` | 94.4% | 2/2 | 54.8 | Best under llama.cpp `--json-schema-file`; one semantic miss. |
| Structured constrained JSON/chat | `FunctionGemma 270M` | 94.4% | 2/2 | 18.9 | Same score as Phi, much faster; strong tiny parser/control-plane candidate. |
| Structured constrained JSON/chat | `Phi-4-mini` | 94.4% | 2/2 | 90.8 | Strong but slower in chat-template mode. |
| Production-shaped structured JSON | `Phi-4-mini` | 85.7% | 3/4 | 45.8 | Current leader for app-like QA, citation verifier, retrieval gate, and route contracts. |
| Production-shaped structured JSON | `FunctionGemma 270M` | 71.4% | 2/4 | 12.8 | Fast, useful for simple routing/gating, weaker for QA/citation verifier contracts. |
| Production-shaped structured JSON, strict rescore + prompt fix | `Phi-4-mini` | 97.1% | 3/4 | 24.7 | Prompt/schema clarity fixed QA and retrieval gate; remaining miss is citation verifier marking weak evidence as verified. |
| Production-shaped structured JSON, strict rescore + prompt fix | `FunctionGemma 270M` | 74.3% | 1/4 | 10.9 | Still too weak for hard verifier/route semantics; useful only for very simple gates with validation. |
| Production-shaped structured JSON, verifier few-shot | `Phi-4-mini` | 100.0% | 4/4 | 26.5 | Few-shot hard negatives fixed the citation-verifier semantic miss. Current structured/control-plane leader. |
| Structured constrained JSON | `NVIDIA Nano` | 22.2% | 0/2 | 8.0 | llama.cpp sampler initialization error, not a normal model-quality failure. |
| Structured constrained JSON | `gemma-4-E2B` | 22.2% | 0/2 | 9.5 | Same sampler initialization error. |
| Structured constrained JSON | `LFM2-2.6B` | 22.2% | 0/2 | 5.3 | Same sampler initialization error. |
| Structured constrained JSON/chat | `LFM2.5 Nova Function Calling` | 11.1% | 0/2 | 5.2 | Sampler initialization error under JSON-schema path; still needs native function-call-template test. |
| Structured constrained JSON/chat | `Qwen3.5 xLAM Function Calling` | 11.1% | 0/2 | 9.7 | Same sampler error; not counted out for native function calling yet. |
| Citation | `Qwen3.6` | 95.0% | 2/2 | 207.8 | Quality reference, removed from cache. |
| Citation | `NVIDIA Nano` | 95.0% | 2/2 | 83.3 | Best practical model-only citation candidate. |
| Citation | `LFM2-1.2B` | 85.0% | 2/2 | 16.4 | Fast citation-format fallback. |

RAG/webchat benchmark after retrieval fixes:

| Slice | Model | Score | Pass | Avg/sec or case sec | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| Overall RAG | `LFM2-2.6B` | 100.0% | 3/3 | 48.3 | Current webchat leader. |
| Overall RAG | `gemma-4-E2B` | 76.7% | 2/3 | 124.0 | Failed unsupported exact claim; malformed `[N]`. |
| Overall RAG | `NVIDIA Nano` | 66.7% | 2/3 | 80.7 | Weak inline citations in this pipeline. |
| Finance RAG | `LFM2-2.6B` | 10/10 | pass | 62.8 | Tied with Gemma. |
| Finance RAG | `gemma-4-E2B` | 10/10 | pass | 155.9 | Good but slower. |
| Refusal RAG | `LFM2-2.6B` | 10/10 | pass | 28.5 | Best abstention behavior in this run. |
| Citation RAG | `LFM2-2.6B` | 10/10 | pass | 53.6 | Best inline-citation behavior in this run. |

RAG/webchat citation-support rescore:

- Added lightweight citation-support precision to `tools/rag_webchat_bench.py`.
- Rescored existing VM outputs into `output_data/model_bench/rag_webchat_focus_k10_support_rescore_v2`.
- Scores did not change materially: `LFM2-2.6B` remains 100.0% pass 3/3, `gemma-4-E2B` remains 80.0% pass 2/3, NVIDIA Nano remains 70.0% pass 2/3.
- This is still weaker than gold sentence/source-span attribution; use it as an interim gate until gold spans and reranker/citation-verifier scoring are added.

RAG/webchat reranked service-shaped benchmark:

- Updated `tools/rag_webchat_bench.py` to use the same service path as `ChatAgent`: retrieve 15, optionally rerank with `BAAI/bge-reranker-v2-m3`, compress to top 3 context sources, then run local GGUF generation.
- Fixed a benchmark prompt ambiguity from placeholder `[N]` to numeric citations such as `[1]` or `[2]`; this materially improved citation-format behavior, so `[N]` failures should not be counted as model-limit failures before this prompt fix.
- Adjusted unsupported-refusal scoring so a clean refusal is not penalized for lacking inline citations.
- VM artifact: `output_data/model_bench/rag_webchat_reranked_top3_numeric_citations_rescored/rag_webchat_summary.json`.

| Model | Score | Pass | Avg sec | Notes |
| --- | ---: | ---: | ---: | --- |
| `LFM2-2.6B` | 100.0% | 5/5 | 29.0 | Current service-shaped RAG leader across finance, local models, curation, refusal, and citation slices. Only residual note is weak citation-support precision on curation. |
| `Phi-4-mini` | 92.0% | 5/5 | 39.4 | Strong backup; still has occasional missing citations and weaker answer-term coverage. |
| `NVIDIA Nano` | 82.0% | 4/5 | 36.3 | Viable for some slices, but failed local-model RAG due length/citation/coverage issues. |

Slice leaders after reranking and numeric-citation prompt:

| Slice | #1 | #2 | #3 |
| --- | --- | --- | --- |
| Finance | `LFM2-2.6B` 10/10 | `NVIDIA Nano` 9/10 | `Phi-4-mini` 9/10 |
| Local models | `LFM2-2.6B` 10/10 | `Phi-4-mini` 8/10 | `NVIDIA Nano` 5/10 fail |
| Curation/data flywheel | `LFM2-2.6B` 10/10 | `NVIDIA Nano` 10/10 | `Phi-4-mini` 10/10 |
| Refusal | `LFM2-2.6B` 10/10 | `Phi-4-mini` 10/10 | `NVIDIA Nano` 9/10 |
| Citation | `LFM2-2.6B` 10/10 | `Phi-4-mini` 9/10 | `NVIDIA Nano` 8/10 |

Interpretation: for this service-shaped RAG slice, base models plus retrieval/reranking/prompt/scorer fixes are good enough for the current seed benchmark. Do not fine-tune this slice yet; expand the gold set and add sentence/span citation attribution first.

Production-seed gold benchmark:

- Added `tools/build_production_gold_eval.py` and `tools/production_gold_bench.py`.
- The VM production DB was copied locally to `output_data/ai_news.vm.db`; the older host `output_data/ai_news.db` remains the March 29 MB fixture.
- Seed dataset: `output_data/gold_eval/production_seed_v1.jsonl`, 8 cases from 34 unique VM DB sources.
- Case mix: 3 retrieval, 3 provided-source RAG generation, 2 citation-pair checks.
- OpenAI-style seed data for later review: `output_data/gold_eval/production_seed_v1_openai_messages.jsonl`.

Current production-seed cheap gates:

| Gate | Result | Notes |
| --- | ---: | --- |
| Retrieval | 3/3 pass | Scored with parent-document recall because production webchat returns article IDs, not paragraph IDs. `max_sources=10` is the minimum passing setting; `max_sources=15` improves finance/local-model recall. |
| Citation pair | 2/2 pass | Negative cross-source claim and positive Phoenix tracing/eval pair both pass heuristic support scoring. |

Retrieval sweep:

| Setting | Pass | Notes |
| --- | ---: | --- |
| `max_sources=5` | 1/3 | Too narrow; finance and local-model coverage fail. |
| `max_sources=10` | 3/3 | Minimum passing setting: finance parent recall 0.5, curation 0.667, local-model 0.3. |
| `max_sources=15` | 3/3 | Safer setting: finance parent recall 0.6 and local-model recall 0.5. |

Interpretation: retrieve at least 10-15 candidates, then compress/rerank to the top 3 sources for local generation. This is a system fix, not a fine-tuning need.

Service-path implementation status:

- `agents/chat_agent.py` now defaults to retrieving 15 candidates (`CHAT_RETRIEVAL_MAX_SOURCES`) and compresses prompt context to 3 sources (`CHAT_CONTEXT_MAX_SOURCES`) with 250 characters per source (`CHAT_CONTEXT_SOURCE_CHARS`).
- Optional CrossEncoder reranking is now inserted before source compression. Enable with `CHAT_ENABLE_RERANKER=1`; default model is `BAAI/bge-reranker-v2-m3` via `CHAT_RERANKER_MODEL`.
- Citation extraction now uses the compressed context sources, keeping `[N]` citation numbering aligned with the prompt.
- The `sources` event includes `retrieved_source_count`, reranker metadata, and returns the compressed context sources that the model actually sees.
- Tests added in `test_chat_agent.py` cover context source selection, clipping, reranker ordering, and reranker fallback.
- Next step is to run the VM production-seed harness with `CHAT_ENABLE_RERANKER=1` and compare recall/citation support against the source-compression baseline.

Current production-seed generation smoke:

| Case | Model | Result | Elapsed | Notes |
| --- | --- | ---: | ---: | --- |
| `rag_curation_before_finetune_v1` | `LFM2-2.6B` | pass | 21.2s | After fixing the harness with `--single-turn`. |
| `rag_curation_before_finetune_v1` | `gemma-4-E2B` | raw pass, strict UI fail | 59.6s | Emits `[Start thinking]`; demote for user-facing webchat unless a reasoning-off template fixes this. |
| `rag_curation_before_finetune_v1` | `Phi-4-mini` | pass | 38.0s | Clean output. |

Important harness finding:

- The earlier ~246s generation timeouts were a benchmark bug: `llama-cli` stayed in interactive mode after answering.
- Adding `--single-turn` made the same micro LFM prompt pass in 12.6s with a 509-character prompt.
- Keep `--single-turn` or a long-lived `llama-server` in all future llama.cpp generation benchmarks.

Production-seed top-three RAG generation, source-compressed scoring:

| Model | Pass | Avg sec | Strict user-facing notes |
| --- | ---: | ---: | --- |
| `LFM2-2.6B` | 3/3 | 21.3 | Current leader for provided-source RAG generation. |
| `Phi-4-mini` | 3/3 | 36.8 | Clean second choice; slower than LFM but strong refusal behavior. |
| `gemma-4-E2B` | 2/3 raw, 0/3 strict UI | 61.5 | Leaks thinking on all three cases and fails unsupported-source refusal; demote for webchat/refusal unless template-specific prompting fixes it. |

Interpretation: base models are good enough to continue without fine-tuning for this seed RAG generation slice if the system uses LFM2-2.6B or Phi-4-mini with source compression. Gemma should remain a writing-quality candidate only after solving thinking leakage; do not use it for source-grounded refusal yet.

Reranker component smoke:

| Model | Retrieval MRR | Hit@1 | Citation Pairwise | Cold sec | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `BAAI/bge-reranker-v2-m3` | 1.0 | 1.0 | 1.0 | 61.0 | Best first production candidate: strong, Apache 2.0, aligned with current BGE stack. |
| `zeroentropy/zerank-1-small-reranker` | 1.0 | 1.0 | 1.0 | 201.7 | Strong synthetic result, but slower; test VM resource use before production. |
| `Qwen/Qwen3-Reranker-0.6B` | 0.723 | 0.625 | 0.417 | 45.4 | Generic CrossEncoder loader warned about uninitialized head, so this is not a fair result. Retest with Qwen method. |

No LiquidAI, NVIDIA, Google/Gemma, or Microsoft/Phi official rerankers were found in the current HF search. Qwen, BAAI, Mixedbread, and ZeroEntropy are the realistic reranker families to test.

Service integration status: `agents/chat_agent.py` now has a lazy-loaded reranker hook. It is off by default to avoid accidental model downloads/loads, honors reranked order during context compression when applied, and falls back to base retrieval order with a warning if the reranker cannot load.

Production-seed retrieval-context comparison on the VM:

| Setting | Candidate Retrieval Pass | Compressed Context Pass | Avg Context Recall | Avg Context MRR | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `max_sources=15`, top-3 heuristic compression | 3/3 | 2/3 | 0.256 | 0.5 | Finance top-3 missed all gold parent sources. |
| `max_sources=15`, `BAAI/bge-reranker-v2-m3`, top-3 context | 3/3 | 3/3 | 0.322 | 1.0 | Finance gets 1 supporting source, curation gets 2, local-model case gets 2. |

Interpretation: keep BGE reranking for the service-path RAG benchmark before deciding on fine-tuning. The improvement is in the evidence pack the generator sees, not in wide retrieval recall, so this should be measured before rerunning the top generator models.

Reranked service-shaped RAG generation on the VM:

| Model | Score | Pass | Avg sec | Decision |
| --- | ---: | ---: | ---: | --- |
| `LFM2-2.6B` | 100.0% | 5/5 | 29.0 | Primary webchat/RAG generator candidate. |
| `Phi-4-mini` | 92.0% | 5/5 | 39.4 | Backup generator candidate. |
| `NVIDIA Nano` | 82.0% | 4/5 | 36.3 | Keep as structured/citation candidate, but not primary RAG generator. |

Expanded production-shaped benchmark, compact top-3 provided-source RAG:

- Added `production_expanded_v2`, a 20-case gold set from 53 unique VM `ai_news.db` sources. V2 keeps the same cases as v1 but orders provided RAG sources by expected-term coverage so compact top-3 prompts preserve important evidence.
- Case mix: 6 retrieval, 8 RAG generation, 6 citation-pair support cases.
- The OpenAI-format review/training seed file has 8 RAG rows and still requires human review before SFT use.

Cheap retrieval/citation gates:

| Gate | Result | Notes |
| --- | ---: | --- |
| Candidate retrieval | 6/6 | BGE-M3 + `BAAI/bge-reranker-v2-m3`, `max_sources=15`, query-focused source expansion. |
| Top-3 context pack | 5/6 | Structured-output/control-plane retrieval misses gold support after compression; selected sources have relevant terms, but gold spans need human review. |
| Citation pairs | 6/6 | Includes finance/local-model entity mismatches and Mastercard/Balyasny/J.P. Morgan positives. |

Expanded generation leaderboard:

| Model | Pass | Rate | Avg sec | Decision |
| --- | ---: | ---: | ---: | --- |
| `LFM2-2.6B` | 8/8 | 100.0% | 28.5 | Primary RAG/news generator. |
| `Phi-4-mini` | 6/8 | 75.0% | 54.4 | Structured-control-plane primary; RAG backup only after citation prompt/few-shot optimization. |
| `NVIDIA Nano` | 0/8 | 0.0% | 95.1 | Count out for user-facing RAG; it leaks thinking traces on every case. |

Phi citation retry follow-up:

| Model/method | Pass | Rate | Avg sec | Notes |
| --- | ---: | ---: | ---: | --- |
| `Phi-4-mini` + numeric citation prompt + missing-citation retry | 8/8 | 100.0% | 61.9 | Only one case needed retry; no fine-tune needed for this backup path yet. |

The v1 shared LFM/Phi failure was `rag_local_model_deployment_tradeoffs_v1`; v2 fixed the benchmark source ordering so quantization and CPU evidence enter the compact prompt. LFM now passes all expanded RAG cases. Phi's remaining plain-run failures have adequate term coverage but no inline citations; the validation-driven retry closes them without fine-tuning. Nemotron Nano remains a user-facing RAG failure because hidden thinking leaks on every expanded case.

Prompt/scoring fixes tested before fine-tuning: numeric citation wording instead of `[N]` placeholders, UTF-8 replacement for llama.cpp output capture, and unsupported-refusal scoring that does not require citations when the correct answer is abstention.

Structured-output prompt optimization finding:

- Strict rescoring showed the earlier production-shaped structured pass counts were too lenient: schema errors and critical semantic misses can no longer pass.
- A small prompt/schema clarity change improved `Phi-4-mini` from one strict true pass to 3/4 passes and 97.1%, showing DSPy/GEPA-style prompt optimization can help when failures are ambiguous instructions or schema semantics.
- The remaining Phi failure is the citation-verifier contract: it still marks a cited source as `verified` even when the source supports only Gemma GGUF and not the J.P. Morgan analyst-assistant claim. Treat this as a semantic verifier failure, not a syntax issue.
- `FunctionGemma 270M` remains fast but only passed 1/4 under strict criteria after prompt fixes; keep it for simple routing/gating experiments, not citation verification or QA review.

Few-shot verifier result:

- Added `--verifier-few-shot` to `tools/constrained_output_bench.py`, appending three hard-negative/positive citation examples only to the citation-verification contract.
- VM artifact: `output_data/model_bench/constrained_json_production_contracts_phi_fewshot_v1/constrained_summary.json`.
- `Phi-4-mini` reached 100.0%, 4/4, avg 26.5s.
- Parsed citation-verifier output correctly set `status: invalid`, empty `entity_overlap`, and reasoned that the source supports Gemma local model availability but not J.P. Morgan or analyst assistants.
- Interpretation: DSPy/GEPA-style prompt/example optimization is sufficient for the current structured seed contracts. Do not fine-tune this slice until expanded production traces show failures after constrained decoding, validation-error retry, and few-shot hard negatives.

Expanded structured/control-plane contracts:

- Added three production-shaped contracts to `tools/constrained_output_bench.py`: finance relevance classification, citation correction action, and final delivery payload.
- Added `--validation-retry`, which retries failed constrained outputs once with schema/semantic validation errors included in the prompt.
- VM artifacts:
  - `output_data/model_bench/constrained_json_production_contracts_v2_phi_retry/constrained_summary.json`
  - `output_data/model_bench/constrained_json_production_contracts_v2_functiongemma_retry/constrained_summary.json`

| Model/method | Pass | Score | Avg sec | Decision |
| --- | ---: | ---: | ---: | --- |
| `Phi-4-mini` + constrained JSON + verifier few-shot + validation retry | 7/7 | 100.0% | 30.2 | Primary structured/control-plane model. |
| `FunctionGemma 270M` + constrained JSON + verifier few-shot + validation retry | 4/7 | 86.6% | 22.7 | Fast but not reliable for citation verification, route decisions, or delivery payloads. |

Interpretation: structured/control-plane also does not need fine-tuning before first local deployment. Phi has enough base capability when constrained decoding, few-shot hard negatives, and validation retry are used. FunctionGemma remains a narrow/simple-gate candidate only when strict validation can reject bad outputs.

Held-out production structured traces:

- Added `tools/build_heldout_production_traces.py`.
- Generated `output_data/gold_eval/heldout_production_traces_v1` from the expanded v2 real-source gold set.
- Trace mix: 16 held-out structured contracts:
  - 2 finance relevance traces.
  - 2 delivery payload traces.
  - 3 unsupported-source route traces.
  - 3 retrieval gate traces.
  - 6 citation verifier traces.
- These rows are marked `heldout_only`; do not train on them unless a future explicit split moves rows out of holdout.
- Added `--cases-jsonl` support to `tools/constrained_output_bench.py` so production-derived contracts can be run without hard-coding new cases.
- VM held-out top-three structured run:

| Model | Held-out Pass | Score | Avg sec | Decision |
| --- | ---: | ---: | ---: | --- |
| `Phi-4-mini` + constrained JSON + validation retry | 16/16 | 100.0% | 61.7 | Primary structured/control-plane model remains good enough without fine-tuning. |
| `FunctionGemma 270M` | 7/16 | 82.1% | 32.0 | Too weak for production control-plane; keep only for trivial routes behind strict validation. |
| `LFM2-2.6B` | 0/16 | 10.6% | 4.5 | Counted out for llama.cpp JSON-schema structured control-plane; sampler initialization errors on every held-out trace. |

Interpretation: the held-out structured/control-plane gate now supports the same conclusion as the seed contracts. Use Phi with constrained decoding, hard-negative examples, and validation retry. Do not fine-tune this slice before first local deployment; collect human-reviewed failures and add traces when schemas change.

Held-out human-review packet:

- Added `tools/build_human_review_packets.py`.
- Generated `output_data/gold_eval/human_review_v1` from the expanded v2 gold set plus current LFM/Phi/Nemotron generation artifacts.
- Packet size:
  - 24 answer-level rows.
  - 167 sentence-level citation/support rows.
  - 62 rows already have citations.
  - 105 supported-case rows are uncited and tagged `missing_citation_candidate`.
- This is not training data yet. It is a review queue for answer usefulness, factual consistency, and sentence/source support spans. After labeling, split into held-out regression evals and only then decide whether any rows are safe to use for SFT or preference tuning.
- Added `tools/build_review_label_splits.py` to enforce that separation.
- Current split artifact: `output_data/gold_eval/human_review_v1_splits/split_summary.json`.
- Current split status:
  - Reviewed rows: 0.
  - Approved training candidates: 0.
  - Unlabeled or invalid rows: 191.
- Interpretation: we do not yet have approved fine-tuning data from the human-review packet. The dataset is useful as a labeling queue and future regression/training source, but all rows remain excluded from training until labels are completed and candidate rows are explicitly approved.

Local deployment wiring:

- `ChatAgent` now has a switchable local backend via `CHAT_BACKEND=llama_cpp`.
- The local path uses the evaluated service shape: retrieve candidates, optional BGE rerank, compress to top 3 sources, run `llama-cli`, extract citations, and retry missing-citation supported answers with stricter numeric citation instructions.
- VM smoke artifact: `output_data/model_bench/local_chat_backend_smoke/local_chat_backend_smoke.json`.
- VM smoke with LFM2-2.6B Q4_K_M produced a clean cited finance answer in 63.5s CPU time with BGE reranking enabled, `citations_count=4`, and no llama.cpp timing text in the answer.
- This verifies the main Flask/webchat path can run without Bedrock when configured. AgentCore standalone deployed agents now return disabled responses by default unless explicitly re-enabled with `ALLOW_LEGACY_AGENTCORE_RUNTIME=1`.

Hosted migration audit:

- Added `tools/audit_hosted_model_paths.py`.
- Artifact: `output_data/model_bench/hosted_model_migration_audit/hosted_model_migration_audit.json`.
- Result: 12 files still contain hosted/stale references, but current statuses are resolved as local or guarded:
  - `local_backend_available`: 2 files.
  - `legacy_runtime_guarded`: 7 AgentCore runtime files.
  - `hosted_deployment_guarded`: 1 deploy script.
  - `local_docs_updated`: 2 docs files.
- Decision: the main Flask/webchat and shared LLM client can run locally. The AgentCore tournament stack remains disabled by default and should stay that way unless rewritten to call Hetzner-local llama.cpp endpoints.
- Shared `agents/llm_client.py` now has a local default backend:
  - Prose/default model: `LFM2-2.6B`.
  - JSON/control-plane model: `Phi-4-mini`.
  - Bedrock is still available only when `LLM_BACKEND=bedrock`.
  - VM smoke artifact: `output_data/model_bench/llm_client_local_smoke/llm_client_local_smoke.json`.
  - VM smoke passed: `call_llm_json()` returned validated route JSON through local llama.cpp/Phi.

Regression gate:

- Added `tools/model_regression_gate.py`.
- Gate result on local and VM artifacts: `deployment_gate_passed=true`.
- Fine-tune readiness: `fine_tune_ready=false`; the only current warning is `fine_tune_labels_available` because human-review labels and approved train candidates are still absent.
- The gate emits a `model_roster` with per-task keep/count-out status, which is now surfaced in the production decision report.
- Counted out by gate:
  - `NVIDIA Nano`: user-facing RAG generation.
  - `FunctionGemma 270M`: general structured/control-plane use.
  - `LFM2-2.6B`: llama.cpp JSON-schema control-plane path.

Benchmark coverage audit:

- Added `tools/benchmark_coverage_audit.py`.
- Current result: `coverage_gate_passed=false`, `human_label_ready=false`.
- The only hard slice coverage failure is `data_curation_eval`; it needs more gold cases before making a strong per-slice fine-tuning decision.
- This does not block first local deployment because the model regression gate passes, but it does block marking the overall model-evaluation goal complete.

## Counted Out For Current VM

- `Qwen3.5-4B`: dominated in v2 and removed from VM cache.
- `Mistral-Small-3.2-24B-Instruct-2506-UD-IQ1_S`: real smoke with `--ctx 2048` timed out at 900 seconds; removed from VM cache.
- `gpt-oss-20b-mxfp4`: failed to load on the 8 GB no-swap VM because llama.cpp could not allocate a roughly 9.7 GB CPU repack buffer; removed from VM cache.
- `DeepSeek-V3.2` and `Kimi-K2.6`: dry-run sizes are far beyond this VM.
- Prior tiny/poor models already pruned: Qwen3.5 0.8B/2B, Phi-4-mini-reasoning, Llama 3.2 1B/3B, Gemma 3 QAT, LiquidAI 350M/thinking variants.

## Keep For Now

- `gemma-4-E2B`: best all-around current model.
- `LiquidAI/LFM2-2.6B`: strong brief/refusal candidate.
- `LiquidAI/LFM2-1.2B`: fast, useful fallback and citation-format candidate.
- `nvidia/NVIDIA-Nemotron-3-Nano-4B`: best structured/citation production candidate so far.
- `Phi-4-mini-instruct`: still useful JSON fallback.
- `gemma-4-E4B`: citation candidate, but weaker than expected on structured output.
- `LiquidAI/LFM2-8B-A1B`: passed but has not beaten smaller LiquidAI models enough yet.
- `Qwen3.6-35B-A3B`: quality reference only after Nemotron matched citation/structure with much lower cost. Removed from VM cache to make room for GPT-OSS testing.

## Untested Latest-Family Gap

`gpt-oss-20b` has now been tested enough to count out for this VM.

- `ggml-org/gpt-oss-20b-GGUF:gpt-oss-20b-mxfp4.gguf` downloaded and attempted.
- llama.cpp failed during model load with insufficient memory for a roughly 9.7 GB CPU repack buffer.
- Revisit only with a larger RAM VM or explicit storage/swap provisioning.

## Literature-Aligned Interpretation

- Structured output: current failures match JSONSchemaBench-style findings: prompt-only JSON/XML is brittle, and constrained decoding is the right first fix for syntax/schema validity. This does not prove XML is generally better than JSON; XML/tagged output should stay a local empirical fallback, while production structured boundaries should prefer grammar/schema-constrained JSON plus validation/retry. Fine-tune only if semantic field selection remains poor after constraints.
- Refusal/grounding: current failures match RAGTruth/FaithEval-style findings: RAG systems can still produce unsupported or contradictory claims even with retrieved context. Simple unanswerable refusal is easy; nuanced insufficient-proof cases need hard negatives, claim-support verification, and possibly preference tuning.
- Retrieval: the small component benchmark supports keeping BGE-M3 over MiniLM for this repo, and that choice is consistent with BGE-M3/BEIR-style retrieval literature. It does not prove the webchat pipeline is reliable; end-to-end RAG evaluation must separately measure recall@k, MRR/nDCG, answer faithfulness, citation precision/recall, and abstention.
- Citation: current results align with CiteFix/VeriCite/CiteGuard-style work: base models can learn citation format, but citation correctness improves when generation is paired with evidence selection, NLI/semantic verification, and post-processing correction. Keep deterministic source/citation verification in production.
- Finance slice: FinanceBench supports treating finance as a domain-specific retrieval-and-reasoning slice, not as a generic prose preference. Add evidence-backed finance questions with named entities, metrics, and source spans.

Relevant primary anchors:

- IFEval, arXiv 2311.07911: exact-rule instruction checks.
- RAGTruth, arXiv 2401.00396, and FaithEval, arXiv 2410.03727: unsupported/context-conflicting RAG outputs.
- CRAG, NeurIPS 2024: end-to-end RAG evaluation with dynamic retrieval context.
- JSONSchemaBench, arXiv 2501.10868: constrained decoding and JSON schema reliability.
- CiteFix, arXiv 2504.15629; VeriCite, arXiv 2510.11394; CiteGuard, arXiv 2510.17853: citation correction/verification.
- BEIR, arXiv 2104.08663; BGE-M3, arXiv 2402.03216; NV-Embed, arXiv 2405.17428: retrieval/embedding evaluation context.
- FinanceBench, arXiv 2311.11944: finance-domain open-book QA.

## Next Evaluation Step

Before declaring the model selection complete:

1. Improve production-seed curation/eval retrieval recall with reranking and entity/query expansion.
2. Add source compression and reranked top-2/top-3 context before rerunning production-seed generation.
3. Run local generation through `llama-server` or another long-lived process to avoid per-call load/prompt overhead, then rerun the top-three provided-source RAG cases.
4. Add constrained-decoding tests for structured output.
5. Keep Phi as the structured/control-plane default after it passed 16/16 held-out production-derived contracts; do not use LFM2-2.6B for llama.cpp JSON-schema control-plane, and use FunctionGemma only for trivial validated routes.
6. Add parser/router/function-calling candidates before fine-tuning only if they can be tested through their native tool/function templates: Qwen3.5 xLAM function-calling GGUF and LFM2.5 Nova Function Calling GGUF remain native-template gaps.
7. Run prompt/code-fix variants one variable at a time and compare against the same golden dataset. Consider DSPy/GEPA after metrics are stable enough for automated prompt optimization.
8. Decide whether the current VM should keep `LFM2-8B-A1B` and `gemma-4-E4B`, since neither is currently a clear per-role winner.
9. Fine-tune only after base models plus prompt/code/constrained-decoding/retrieval fixes fail a slice-specific good-enough bar.
