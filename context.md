# Local Model Migration Context

Last updated: 2026-05-26

## Goal

Migrate the Hetzner-hosted system away from Claude/Anthropic usage toward free local CPU inference. The product direction is no longer "Twitter influencer"; it should become a background agent that produces a useful news summary for Brandon with interesting insights, citations, and practical takeaways.

Current model-selection convergence criteria:

- Evaluate models by workflow task, not only by one aggregate score.
- Keep the top 3 viable models for each role so the final system can use a constellation of small local models.
- Define "best" and "good enough" per slice:
  - Best means highest verified score for that workflow role under the current benchmark, with failure modes understood.
  - Good enough means the smallest/cheapest local model that meets the slice's success threshold after allowed prompt/code/constrained-decoding fixes, not necessarily the highest-scoring model.
  - Use Pareto comparisons for quality versus latency/disk/RAM. Discard dominated models unless they provide a unique fallback role.
- Treat evals like product tests:
  - Keep a golden dataset per slice with representative inputs and expected/acceptable outputs.
  - Run repeated trials for nondeterministic generation where temperature or sampling is nonzero.
  - Track code-based metrics first: JSON validity, schema validity, citation-range validity, source support, length, banned social/influencer language, finance entity coverage, p50/p95 latency, disk/RAM.
  - Use LLM-as-judge or human review only for subjective equivalence: usefulness, insight quality, factual consistency beyond deterministic checks.
  - Promote failed production traces into regression evals before changing prompts/models.
- Always evaluate the latest available model family and largest viable size/quant for that family before settling for older variants. Older models are only baselines or fallbacks when the latest family cannot load, lacks usable quants, or fails quality/convergence.
- GPT-OSS was initially missed and must be included as a latest-family candidate:
  - `unsloth/gpt-oss-20b-GGUF` and `ggml-org/gpt-oss-20b-GGUF` are the realistic VM stress-test options.
  - Smallest observed Unsloth files are about 11.5 GB; ggml-org `gpt-oss-20b-mxfp4.gguf` is about 12.1 GB.
  - Treat `gpt-oss-120b` as larger-machine/reference territory.
  - `ggml-org/gpt-oss-20b-GGUF:gpt-oss-20b-mxfp4.gguf` was tested and failed to load on the current 8 GB no-swap VM because llama.cpp could not allocate a roughly 9.7 GB CPU repack buffer. Count out for this VM; keep as larger-machine candidate.
- Other large OSS provider families to track:
  - Mistral: `unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF`; smallest observed UD file is about 5.6 GB, with more practical tiny quants around 6-9 GB. This is a plausible one-at-a-time stress candidate on the current VM.
  - DeepSeek: `unsloth/DeepSeek-V3.2-GGUF`; even the smallest split quant dry-run requires many roughly 50 GB shards / hundreds of GB total. Exclude from this VM and treat as larger-machine territory.
  - Kimi: `unsloth/Kimi-K2.6-GGUF`; split quant dry-run requires hundreds of GB. Exclude from this VM and treat as larger-machine territory.
  - Qwen Coder: `unsloth/Qwen3-Coder-Next-GGUF`; consider for code/tool-planning slices, not necessarily Brandon prose.
  - Gemma stress tiers: Gemma 4 26B-A4B and 31B remain quality-reference/stress candidates; only retain if they materially beat E2B/E4B because disk and RAM pressure are high.
- Roles to rank separately:
  - Brandon news brief generation.
  - Finance-domain relevance for Brandon's new role.
  - Source-grounded refusal / anti-hallucination.
- Structured outputs only where downstream code needs deterministic parsing.
  - Test both JSON and XML/tagged structured output for those boundaries. Small local models may emit XML/tags more reliably than strict JSON, while JSON remains simpler for code validation. If XML wins, parse/validate XML and convert it into typed internal objects.
  - Citation discipline and source attribution.
- Brandon's new role makes finance-domain relevance important, but it is a benchmark slice rather than the entire benchmark. That slice should highlight AI items that reference hedge funds, financial services, fintech, companies/services, OSS models, domain-specific models, and workflows.
- Priority finance company contexts include J.P. Morgan, Acadian Asset Management, Balyasny, Arrowstreet Capital, Citadel, Mastercard, Visa, and similar regulated finance companies.
- Count out models that fail tiny prompts, leak thinking traces unexpectedly, cannot load reliably, or are dominated across all roles.
- Slowness is not a primary disqualifier for autonomous background work if the model converges and output quality is materially better. Runtime should be tracked for scheduling/capacity, but prune for non-convergence, bad outputs, memory/load failure, or quality domination.
- After the first-pass benchmark, run an expanded benchmark on the per-role top candidates.
- Before fine-tuning, run the full "engineering ladder" on the top candidates:
  1. Prompt variants with exactly one variable changed at a time, including few-shot examples where small models can imitate format.
  2. Deterministic code fixes where cheaper than inference: truncation, citation-range stripping, schema validation, retry, source support filters.
  3. llama.cpp constrained decoding with `--json-schema-file` / grammar where the boundary truly needs JSON.
  4. Parser/router candidates such as FunctionGemma 270M, Qwen3.5 xLAM function-calling GGUFs, and LFM2.5 Nova Function Calling GGUF.
  5. Retrieval improvements such as LightRAG-style entity/relation sidecar and reranking for finance/tooling/citation slices.
  6. Only then decide whether SFT/QLoRA or preference tuning is needed.
- Decide base-model sufficiency versus fine-tuning separately for each role after prompt/code/constrained-decoding/retrieval fixes have been tested.
- Before committing to our own fine-tune, search community OSS for already tuned variants/adapters/distills that target the same failure mode or workflow. This includes RAG, tool-use, extraction, citation, refusal/grounding, finance-domain, and structured-output variants for the selected model families.
- Add community/function-calling fine-tunes to the candidate set when they match a workflow role, even if the base model family already failed as a general summarizer. They should be judged on the narrow role they were trained for.
- For each shortlisted model family, inspect official and credible community examples before counting it out or fine-tuning:
  - LiquidAI / LFM: Liquid4All cookbook examples, model-specific llama.cpp/server settings, prompt/tool-call formats, structured extraction examples, and LFM2.5/LFM2 MoE guidance.
  - Gemma / FunctionGemma: Google/Unsloth examples, chat templates, function-calling formats, schema/JSON extraction examples.
  - Qwen / Qwopus: Qwen docs, Unsloth quant notes, xLAM/function-calling fine-tunes, reasoning-off/template requirements, structured-output guidance.
  - Phi: Microsoft/Unsloth examples, Phi-4-mini JSON/tool-use prompting and constrained decoding behavior.
  - NVIDIA Nemotron: NVIDIA docs/examples for Nemotron text, function/tool use, embedding/reranking, and NeMo Curator/Data Flywheel recipes.
  - Retrieval models: BGE, NVIDIA embed/NV-Embed, EmbeddingGemma, zembed/ZeroEntropy examples for indexing, reranking, and evaluation.
- Capture per-family operational notes in the context before interpreting benchmark failures. A failure with the wrong template, missing server mode, unsupported JSON-schema path, or ignored cookbook guidance is a harness failure until retested with the recommended method.

Benchmark layering:

- Keep `tools/model_quality_bench.py` as an isolated generation benchmark for local chat models.
- Add a separate retrieval-agent/webchat benchmark for the deployed webchat and inline-citation pipeline. Inline citation quality depends on retrieval, chunking, embedding model, reranking, prompt assembly, generator behavior, and citation verification, so failures should be attributable to the pipeline stage rather than hidden inside a pure model score.
- The retrieval-agent benchmark should test end-to-end questions against known source sets and report:
  - retrieval recall@k / MRR against gold supporting documents,
  - reranker lift versus base retrieval,
  - citation precision/recall at sentence level,
  - unsupported-claim refusal,
  - answer usefulness and concision,
  - latency/resource use as scheduling data, not primary pruning criteria.
- Inline citations should be a required slice for the retrieval-agent benchmark because the current deployed webchat uses Gemini Flash for that behavior and may be swapped to a local model.
- Candidate retrieval components to test:
  - current BGE-M3 hybrid retrieval baseline,
  - EmbeddingGemma 300M for efficient local multilingual/on-device-style embeddings,
  - NVIDIA NV-Embed / newer NVIDIA embedding models, including NVIDIA Omni embedding models, if they fit the VM or can run in a separate indexing job,
  - `nvidia/NV-Embed-v2` as a strong quality reference, subject to license and resource constraints,
  - `nvidia/llama-nemotron-embed-1b-v2` as a more practical NVIDIA retriever candidate,
  - zembed / ZeroEntropy embedding and reranking options,
  - Qwen3 and BGE rerankers if CPU runtime is acceptable.

Benchmark validity references:

- Instruction following and formatting constraints:
  - IFEval (`google-research` / arXiv 2311.07911) is the closest analogue for verifiable instructions such as required citations, banned terms, bullet counts, and output format constraints.
  - Use this to justify exact-rule checks, but avoid overfitting to fixed prompts by perturbing wording in the expanded pass.
- Source-grounded refusal and hallucination:
  - RAGTruth (arXiv 2401.00396) is relevant for unsupported or contradictory claims in RAG outputs and hallucinated spans.
  - FaithEval (arXiv 2410.03727) is especially relevant to the conflicting-source/insufficient-proof failure mode because it tests whether models stay faithful to provided context under unanswerable, inconsistent, and counterfactual contexts.
  - RefusalBench-style dynamic tests are relevant for unanswerable questions and hard negatives; fixed refusal prompts are too easy to memorize.
- End-to-end retrieval/webchat:
  - CRAG / Comprehensive RAG Benchmark (NeurIPS 2024) is relevant for factual QA over dynamic retrieved context, including retrieval quality and answer correctness.
  - Retrieval-agent benchmark should report recall@k, MRR, nDCG@k, citation precision/recall, and unsupported-answer refusal.
- Structured boundaries:
  - StructuredRAG-style work is relevant for JSON reliability in complex RAG systems.
  - There is not enough academic support for a blanket claim that XML is better than JSON. Treat XML/tags as a local empirical fallback only. JSON remains the better production interchange format when grammar/schema constrained, validated, and retried.
  - Provider guidance generally favors native structured modes/tool calling where available. For local llama.cpp, prefer grammar-constrained decoding (`--grammar`, JSON schema to GBNF, or LLGuidance-enabled builds) over unconstrained prompting for strict syntax.
  - Constrained decoding fixes syntax/shape validity; it does not guarantee semantic correctness. Keep validators for required fields, source IDs, enum values, and business rules.
  - Remediation order for failed structured output:
    1. Use constrained decoding / JSON schema / GBNF where possible.
    2. Extract and repair deterministic wrappers such as markdown fences and prompt echo.
    3. Validate with schema and retry with explicit validation errors.
    4. Fine-tune only if syntax is valid but semantic field selection, role planning, or tool-choice quality remains poor.
  - Relevant literature/tools:
    - OpenAI Structured Outputs and Mistral/Qwen structured output docs for provider-native JSON/schema modes.
    - llama.cpp grammar / JSON schema to GBNF and LLGuidance for local constrained decoding.
    - JSONSchemaBench / structured-output benchmark papers for measuring schema coverage and constrained-decoding overhead.
    - Small-model QLoRA/tool-use work, including recent Gemma/Qwen tool-planning fine-tunes, for cases where semantic planning is the failure rather than syntax.
- Citation attribution:
  - CiteFix (arXiv 2504.15629), VeriCite (arXiv 2510.11394), and CiteGuard (arXiv 2510.17853) support the current direction: generate citations, then verify/correct evidence attribution instead of trusting citation markers from the generator.
  - Sentence-level and sub-sentence attribution work suggests citation-format pass rates are not enough. The benchmark should score whether each cited source actually supports the local claim.
- Retrieval / embeddings:
  - BEIR (arXiv 2104.08663) supports evaluating retrievers across heterogeneous tasks and using recall/MRR/nDCG rather than only qualitative answer checks.
  - BGE-M3 (arXiv 2402.03216) supports the current dense+sparse hybrid baseline; our small component benchmark only shows it works on our synthetic slices, not that the deployed webchat is correct.
  - NV-Embed (arXiv 2405.17428) supports testing NVIDIA embedding families as quality references, but VM failures/OOM still count for production viability.
- Finance-domain slice:
  - Use finance-specific RAG/eval patterns where possible, such as FinanceBench-style document QA and domain-specific entity coverage, but keep this as one slice rather than the whole benchmark.
  - FinanceBench (arXiv 2311.11944) is the closest benchmark anchor: finance questions should require evidence spans, named entities, metric/date disambiguation, and multi-document retrieval where possible.

## Hetzner VM

- Server: `linkedin-influencer`
- IPv4: `157.90.125.102`
- SSH port: `49222`
- User: `appuser`
- Type: Hetzner `cx33`
- CPU: 4 shared x86 vCPU, Intel Skylake, AVX2 + AVX-512 available
- RAM: 8 GB total
- Disk: 80 GB total, about 44 GB free after model/cache work
- OS: Ubuntu 24.04

Firewall update completed:

- Added current VPN IP `23.234.114.196/32` for inbound TCP `49222`.
- VPN changed on 2026-05-28; added current VPN IP `23.234.110.196/32` for inbound TCP `49222`, `5001`, and `5002`.
- Storage cleanup on 2026-05-28 removed large non-finalist HF caches from the VM:
  - `unsloth/Qwen3.6-35B-A3B-GGUF` (about 11 GB): quality-reference only, not a practical fine-tuning base on the 8 GB CPU VM.
  - `unsloth/gemma-4-E4B-it-GGUF` (about 8.9 GB): no longer a per-role production leader after LFM2-2.6B, Phi-4-mini, Gemma E2B, and guarded Nemotron Nano results.
  - Free disk improved from about 7.7 GB to about 27 GB.
  - Kept LFM2-2.6B, Phi-4-mini, Gemma E2B, BGE-M3, MiniLM, BGE reranker, and small parser/function-calling candidates for continued benchmark/fine-tune assessment.

Services stopped during model work:

- `linkedin-ui.service`
- `linkedin-feed.service`

Stopping these frees several GB of RAM. With them stopped the VM has about 6.5 GB available and no swap.

## Claude Data Distillation

Use the VM Claude folder, not the local laptop Claude folder:

- Source: `/home/appuser/.claude`
- Size: about 532 MB total
- Useful project JSONL files: 9,055
- Useful project JSONL text size: about 2.7 MB
- Event counts observed: 14,332 user events, 3,544 assistant events

Created a local redacting/distillation script:

- Repo file: `tools/claude_distill.py`
- VM file: `/home/appuser/twitter_influencer/tools/claude_distill.py`

Output on VM:

- Directory: `/home/appuser/twitter_influencer/output_data/claude_distill`
- `sft_train.jsonl`: 1,147 rows, about 7.9 MB
- `sft_validation.jsonl`: 128 rows, about 897 KB
- `bench_prompts.jsonl`: 104 rows, about 691 KB
- `manifest.json`

Copied to host:

- Host directory: `output_data/claude_distill`
- OpenAI chat format directory: `output_data/claude_distill/openai`
- `openai/train.jsonl`: 1,147 valid OpenAI chat rows
- `openai/validation.jsonl`: 128 valid OpenAI chat rows
- Format: `{"messages":[{"role":"system"},{"role":"user"},{"role":"assistant"}],"metadata":{...}}`

The curation done so far is deterministic and local:

- Parse Claude Code project JSONL.
- Pair user messages with following assistant messages.
- Redact common secrets/tokens/private keys.
- Split into SFT train/validation.
- Build benchmark prompts with references and a rubric.

NVIDIA NeMo Curator / Data Flywheel has not yet been run. It should be considered for the next curation pass, especially for deduplication, quality filtering, PII/secrets filtering, synthetic augmentation, and eval-set construction.

Features to incorporate from related tools:

- Arize Phoenix:
  - Trace every agent run as spans: source collection, ranking, summarization, citation verification, and delivery.
  - Record model id, quant, prompt version, runtime, selected sources, output, verifier result, and failure labels.
  - Promote failed or interesting traces into datasets for repeatable experiments.
  - Use deterministic evaluators first: citation validity, JSON parse, unsupported-source refusal, banned social language, finance-entity coverage.
  - Compare model/prompt changes on the same datasets before deployment.
- NVIDIA Data Flywheel / NeMo Curator:
  - Treat production traces as the raw material for curation, eval construction, and later fine-tuning.
  - Deduplicate similar examples and filter low-quality or secret-bearing samples.
  - Segment curated data by role: brief, finance relevance, citation verification, JSON planning, unsupported-source refusal.
  - Keep eval holdouts separate from SFT/preference data.
  - Generate synthetic hard cases for conflicting sources, unsupported claims, finance-company mentions, stale news, and citation distractors.
- LLaMA-Factory:
  - Use for training, not VM inference.
  - Start with LoRA/QLoRA SFT for style, schema, Brandon brief format, and finance relevance.
  - Use DPO/KTO/ORPO-style preference training for refusal, citation discipline, and avoiding influencer copy if SFT is insufficient.
  - Merge/quantize trained adapters to GGUF for llama.cpp inference after validation.

Refusal fine-tuning notes:

- Unsupported-source refusal is a viable fine-tuning target when the base model has enough capacity and already follows instructions reasonably.
- Use SFT records where the answer explicitly says the requested fact is not in the sources and states what is missing.
- Use preference records where the chosen answer stays source-grounded and the rejected answer invents a plausible finance/company claim, over-cites, or adds social/influencer language.
- Do not expect fine-tuning to fix tiny-model reasoning limits, incoherent outputs, weak multi-document synthesis, or poor entity disambiguation in sub-capacity models; those are model-selection failures.
- Useful external dataset/eval patterns to inspect before building our own:
  - `NovachronoAI/RAG-Grounded-QA-188k` for answerable/unanswerable RAG examples.
  - RAGTruth for hallucinated span and incorrect-refusal annotations.
  - RefusalBench-style dynamic/hard-negative refusal tests so the model cannot memorize fixed prompts.

Guardrail principle:

- Train models to reduce failure rates, but still enforce citation validity, JSON schema validity, source-grounded refusal, and delivery constraints with deterministic validators and retry loops.

Assessment of whether VM `.claude` data is enough for better benchmarks:

- Current distilled artifacts:
  - `sft_train.jsonl`: 1,147 rows.
  - `sft_validation.jsonl`: 128 rows.
  - `bench_prompts.jsonl`: 104 rows.
  - OpenAI chat format mirrors the same 1,147 / 128 rows.
- Term coverage in `sft_train.jsonl`:
  - `citation`: 610 rows.
  - `citations`: 528 rows.
  - `rag`: 772 rows.
  - `arize`: 352 rows.
  - `finance`: 150 rows.
  - `retrieval`: 40 rows.
  - `curator`: 9 rows.
  - `schema`: 3 rows.
- Term coverage in `bench_prompts.jsonl`:
  - `citation`: 58 rows.
  - `citations`: 47 rows.
  - `rag`: 45 rows.
  - `finance`: 8 rows.
  - `retrieval`: 1 row.

Conclusion:

- The `.claude` distill is enough to seed style/workflow benchmarks, prompt optimization traces, citation-policy examples, and regression cases derived from previous failures.
- It is not enough by itself for deployment-grade gold evals because it lacks:
  - reliable gold source IDs and sentence-level support spans;
  - enough finance-company/current-role cases;
  - enough structured-schema/control-plane examples;
  - enough retrieval cases with gold recall@k/MRR/nDCG labels;
  - clean held-out splits for every workflow role.
- We should generate and curate additional data, but generate it from our actual workload artifacts:
  1. sample real `ai_news.db` retrieved source sets;
  2. label gold source IDs/spans and finance relevance;
  3. derive negative/hard cases from missing evidence, conflicting sources, stale dates, and invalid citations;
4. synthesize structured JSON examples for actual production contracts;

Expanded production gold benchmark status:

- Added `production_expanded_v2` from real VM `ai_news.db` rows. V2 keeps the same case mix as v1 but orders provided RAG sources by expected-term coverage so compact top-3 prompts preserve key evidence such as quantization/CPU/local-model tradeoffs.
- Artifacts:
  - `output_data/gold_eval/production_expanded_v2.jsonl`
  - `output_data/gold_eval/production_expanded_v2_openai_messages.jsonl`
  - `output_data/gold_eval/production_expanded_v2_summary.json`
- Case mix: 20 cases from 53 unique sources:
  - 6 retrieval cases.
  - 8 provided-source RAG generation cases.
  - 6 citation-pair support cases.
- Slices covered:
  - finance-domain signal, including regulated payments and hedge-fund/asset-management research;
  - local-model operations;
  - pre-fine-tune curation/prompt optimization;
  - source-grounded refusal;
  - structured-control-plane retrieval;
  - inline citation support/mismatch cases.
- Cheap gate with BGE-M3 retrieval + `BAAI/bge-reranker-v2-m3` + query-focused source expansion:
  - Candidate retrieval: 6/6.
  - Top-3 compressed context pack: 5/6.
  - Citation pairs: 6/6.
  - Remaining retrieval gap: structured-output/control-plane query finds relevant candidates but the top-3 compressed context misses gold support. Fix with better query/entity expansion or role-specific context packing, not model fine-tuning.
- Expanded compact RAG generation on the VM, top 3 candidates, 8 cases each:
  - `LiquidAI/LFM2-2.6B-GGUF:LFM2-2.6B-Q4_K_M.gguf`: 8/8, 100.0%, avg 28.5s.
  - `unsloth/Phi-4-mini-instruct-GGUF:Phi-4-mini-instruct-Q3_K_M.gguf`: 6/8, 75.0%, avg 54.4s.
  - `nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF:NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf`: 0/8, avg 95.1s, counted out for user-facing RAG because it leaks thinking traces on every case.
- Phi backup prompt/retry result:
  - Removed the ambiguous `[N]` placeholder from the benchmark prompt and replaced it with numeric citation examples such as `[1]` and `[2]`.
  - Added validation-driven retry only when a supported answer has enough content coverage but no valid citations.
  - Artifact: `output_data/gold_eval/production_expanded_v2_generation_phi_citation_retry`.
  - Result: `Phi-4-mini` 8/8, 100.0%, avg 61.9s. Only one case needed the stricter retry.
- Interpretation:
  - LFM2-2.6B remains the primary RAG/news generator.
  - Phi-4-mini remains the structured-control-plane primary and is a viable RAG backup when validation retries missing-citation outputs with the stricter citation prompt.
  - NVIDIA Nano should not be used as the user-facing RAG generator unless a model-specific no-thinking template fixes the leakage; keep it only for non-user-facing experiments if needed.
  - No RAG fine-tune is justified yet. The next improvements are retrieval/context packing, prompt/example optimization, deterministic citation verification, and human-reviewed gold spans.
  5. keep generated train data separate from human/trace-derived validation and test sets.
- Use the `.claude` data as seed/teacher traces, not as the sole benchmark authority.

## Local Runtime

`llama.cpp` is the correct CPU-only runtime for this VM because it supports GGUF, CPU inference, and Unsloth/UD quantized files.

Installed on VM:

- Path: `/home/appuser/opt/llama.cpp`
- Build: `b9310`
- Binaries: `llama-cli`, `llama-server`

Important runtime detail:

- Qwen reasoning models must be run with `--reasoning off --single-turn` for direct benchmark output.
- Without this, Qwen3 entered thinking mode and did not finalize quickly.

Created benchmark harness:

- Repo file: `tools/local_model_bench.py`
- VM file: `/home/appuser/twitter_influencer/tools/local_model_bench.py`
- Outputs: `/home/appuser/twitter_influencer/output_data/model_bench`

Current benchmark status:

- Early Qwen3 runs produced empty result files because they were killed while debugging reasoning mode.
- A direct tiny Qwen3-0.6B prompt completed correctly after adding `--reasoning off --single-turn`.
- The benchmark runner has been patched to include those flags.
- Smoke-test runner added: `tools/smoke_local_models.py`.
- Task-specific quality runner added: `tools/model_quality_bench.py`.
- Exact GGUF file repos require `--hf-repo` + `--hf-file`; `-hf repo:file` failed for these repos.
- Smoke tests completed successfully for five accessible candidates:
  - `unsloth/Qwen3.5-0.8B-GGUF:Qwen3.5-0.8B-UD-IQ2_XXS.gguf` loaded/exited in 19.57s, but output quality was bad on the tiny prompt.
  - `LiquidAI/LFM2.5-350M-GGUF:LFM2.5-350M-Q4_K_M.gguf` loaded/exited in 6.81s with a clean short answer.
  - `LiquidAI/LFM2.5-1.2B-Thinking-GGUF:LFM2.5-1.2B-Thinking-Q4_K_M.gguf` loaded/exited in 9.65s but emitted `<think>`, so needs thinking suppression/template work.
  - `unsloth/Qwen3.5-2B-GGUF:Qwen3.5-2B-UD-IQ2_XXS.gguf` loaded/exited in 12.07s with a generic answer.
  - `unsloth/Qwen3.5-4B-GGUF:Qwen3.5-4B-UD-IQ2_XXS.gguf` loaded/exited in 24.30s with a generic answer.
- These are smoke results only. They prove load/generate/exit, not quality.
- After copying the host HF token to the VM, gated model access works via `HF_TOKEN`.
- Additional smoke tests:
  - `unsloth/Phi-4-mini-instruct-GGUF:Phi-4-mini-instruct-Q3_K_M.gguf`: passed, 48.99s.
  - `unsloth/Phi-4-mini-reasoning-GGUF:Phi-4-mini-reasoning-UD-IQ2_XXS.gguf`: passed, 71.16s, but output was nonsensical on the tiny prompt.
  - `unsloth/Llama-3.2-1B-Instruct-GGUF:Llama-3.2-1B-Instruct-UD-IQ3_XXS.gguf`: passed, 25.68s.
  - `unsloth/Llama-3.2-3B-Instruct-GGUF:Llama-3.2-3B-Instruct-UD-IQ2_XXS.gguf`: passed, 55.32s.
  - `unsloth/Qwen3.6-35B-A3B-GGUF:Qwen3.6-35B-A3B-UD-IQ1_M.gguf`: passed, 116.03s, about 6.7 GB RSS, no swap, services stopped.
  - `unsloth/gemma-4-E2B-it-GGUF:gemma-4-E2B-it-UD-IQ2_M.gguf`: passed, 58.83s.
  - `unsloth/gemma-4-E4B-it-GGUF:gemma-4-E4B-it-UD-IQ2_M.gguf`: passed, 34.09s.
  - `unsloth/gemma-4-26B-A4B-it-GGUF:gemma-4-26B-A4B-it-UD-IQ2_XXS.gguf`: passed, 180.30s, about 6.8 GB RSS, no swap, services stopped.
  - `google/gemma-3-1b-it-qat-q4_0-gguf` and `google/gemma-3-4b-it-qat-q4_0-gguf`: failed quickly; use Unsloth Gemma 4 instead.
- VM model cache grew to about 44 GB and root disk reached 92% used / 6 GB free. Do not download more large models before pruning cache or increasing disk.
- Pruning performed after first quality pass:
  - Removed old Qwen3 0.6B/1.7B test caches.
  - Removed Qwen3.5-0.8B due bad tiny-prompt output.
  - Removed Phi-4-mini-reasoning due nonsensical tiny-prompt output.
  - Removed failed Gemma 3 QAT caches.
  - Removed LFM2.5-350M, LFM2.5-1.2B-Thinking, Llama 3.2 1B/3B, Qwen3.5-2B, and Gemma 4 26B-A4B as dominated/poor/too slow for this VM.
  - Cache after pruning: about 28 GB. Root disk: about 50 GB used / 22 GB free.

LFM larger-model result:

- `LiquidAI/LFM2-1.2B-GGUF:LFM2-1.2B-Q4_K_M.gguf`: passed smoke, 21.46s.
- `LiquidAI/LFM2-2.6B-GGUF:LFM2-2.6B-Q4_K_M.gguf`: passed smoke, 49.19s.
- Newly identified larger LiquidAI MoE candidates:
  - `LiquidAI/LFM2-8B-A1B-GGUF`, latest updated Mar 30, 2026.
  - Exact files include `LFM2-8B-A1B-Q4_0.gguf` at about 4.7 GB and `LFM2-8B-A1B-Q4_K_M.gguf` at about 5.0 GB.
  - This is the next LiquidAI model to test because it activates about 1B parameters per token and should be plausible on the 8 GB VM.
  - `LiquidAI/LFM2-24B-A2B-GGUF`, latest updated Mar 30, 2026.
  - Exact files include `LFM2-24B-A2B-Q4_0.gguf` at about 13.5 GB and `LFM2-24B-A2B-Q4_K_M.gguf` at about 14.4 GB.
  - Treat 24B-A2B as a stress-test candidate only; it may not load reliably without swap or a larger VM.

First real-context quality prompt:

- Prompt described migration from Claude/Twitter influencer tooling to Brandon local CPU news briefs with citations and second-order insights.
- LFM2-2.6B gave the best LiquidAI response among tested LFMs.
- Qwen3.5-4B gave a relevant 4-bullet style answer and beat Qwen3.5-2B.
- Phi-4-mini-instruct gave relevant bullets around summarization and citations.
- Gemma 4 E2B/E4B both produced relevant structured answers.
- Tiny prompt failures/poor outputs are now treated as exclusions.

Current retained model cache shortlist:

- `LiquidAI/LFM2-1.2B-GGUF`
- `LiquidAI/LFM2-2.6B-GGUF`
- `LiquidAI/LFM2-8B-A1B-GGUF`
- `nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF`
- `unsloth/Phi-4-mini-instruct-GGUF`
- `unsloth/gemma-4-E2B-it-GGUF`
- `unsloth/gemma-4-E4B-it-GGUF`
- `unsloth/Qwen3.6-35B-A3B-GGUF`
- Existing embedding model: `BAAI/bge-m3`

Hugging Face private dataset upload:

- Host has `hf`, but stored token is invalid.
- VM does not have `hf` installed.
- Private push is blocked until a fresh Hugging Face token/login is available.

## Candidate Models

Prefer latest practical models, and prefer Unsloth GGUF / UD Dynamic quants where available.

### Qwen

Use Qwen3.5 for small local models, not older Qwen3.

Candidate order:

1. `unsloth/Qwen3.5-0.8B-GGUF`
2. `unsloth/Qwen3.5-2B-GGUF`
3. `unsloth/Qwen3.5-4B-GGUF`
4. `unsloth/Qwen3.5-9B-GGUF` only if testing with swap or bigger VM

Qwen3.6 is latest overall, but the relevant open model is much larger:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- Smallest main GGUF found: `Qwen3.6-35B-A3B-UD-IQ1_M.gguf`, 9.36 GiB
- Next: `UD-IQ2_XXS`, 10.02 GiB
- This will not fit cleanly in 8 GB RAM without swap. Disk space is enough, RAM is the blocker.

Also checked:

- `unsloth/Qwen3.5-35B-A3B-GGUF`
- Smallest main GGUF found: `Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf`, 9.93 GiB

Qwen3.6 dense sibling:

- Model: `Qwen3.6-27B`
- Unsloth GGUF repo found: `unsloth/Qwen3.6-27B-GGUF`
- Other GGUF repo found: `batiai/Qwen3.6-27B-GGUF`
- It is a dense 27B model, so all 27B parameters are active each forward pass.
- This is likely harder for the 8 GB CPU VM than `Qwen3.6-35B-A3B`, despite the smaller total parameter count, because the MoE model only activates about 3B parameters per token.
- Public reports suggest Qwen3.6-27B tiny quants need about 12-18 GB RAM depending on quant/context. Treat this as larger-VM territory unless testing with a temporary swap file.
- If tested anyway, try the smallest Unsloth Dynamic/UD quant available, with low context and `--reasoning off --single-turn`.

### Gemma

Use Gemma 4 if GGUF/llama.cpp support is available, not Gemma 3.

Candidate order:

1. Gemma 4 E2B GGUF / Unsloth GGUF if available
2. Gemma 4 E4B GGUF / Unsloth GGUF if available and memory holds
3. Fall back to Gemma 3 1B/4B only if Gemma 4 packaging/runtime is not practical

### Liquid AI

Use LFM2.5 for small CPU candidates, not older LFM2.

Candidate order:

1. LFM2.5-350M GGUF
2. LFM2.5-1.2B-Thinking GGUF
3. LFM2.5 2B-class model if available and memory holds

Need to handle thinking mode similarly to Qwen if the model emits reasoning traces.

### Phi

Use Phi-4-mini family, not Phi-3/Phi-3.5.

Candidate:

- Phi-4-mini / Phi-4-mini-reasoning GGUF if available

### Llama

Latest practical small Llama for this VM remains Llama 3.2 1B/3B.

Llama 4 Scout/Maverick are newer overall but too large for this VM class.

### Qwopus / Opus Distills

Interesting as teacher/eval candidates, especially Qwen3.5/3.6 Opus-distilled models. Most are likely too large for the 8 GB VM, but check Unsloth/community GGUF quants before ruling out.

## 35B Tiny Quant Test

Smallest known Qwen3.6 35B-A3B main quant is 9.36 GiB (`UD-IQ1_M`). This is larger than physical RAM. A test may be possible only with:

- services stopped,
- a temporary 12-20 GB swap file,
- low context,
- low prediction count,
- `--reasoning off --single-turn`,
- expectation of very slow generation.

Do not treat success under swap as production-ready. If it works, it only proves emergency feasibility. A reliable 35B-A3B path needs a larger VM, likely 16-32 GB RAM minimum.

## Model Pass Criteria

The current benchmark pass criteria need to become stricter before model selection.

Minimum checks:

- Completes benchmark prompts without hanging.
- Does not emit thinking traces unless requested.
- Follows the concrete user request.
- Preserves constraints from the prompt.
- Avoids invented facts.
- Produces direct, useful engineering prose.

First-pass role benchmark tasks:

- `brandon_brief`: 5-bullet Brandon news brief, source citations, second-order insight, no social/influencer language.
- `finance_relevance`: finance-facing slice for Brandon's new role, including hedge funds, asset managers, fintech/payments, regulated finance firms, OSS/domain-specific models, and workflows.
- `source_refusal`: answer only from sources and refuse unsupported Anthropic pricing details.
- `workflow_json`: strict JSON for local CPU news-agent workflow. This is a system-boundary test, not a requirement for all model outputs. Use it where downstream code must parse tool/workflow plans, API/webchat payloads with citations, trace/eval records, or retryable verifier outputs.
- `workflow_xml`: XML/tagged alternative for the same structured workflow boundary, to see whether local models converge more reliably than with JSON.
- `citation_discipline`: compare model candidates with correct source citations and avoid using a false source to support a local Phi claim.

## Verified Quality Benchmark v2

Run completed on the VM at `output_data/model_bench/quality_v2` using the corrected harness with prompt/banners stripped, finance slice included, JSON and XML structured-output slices, and per-task leaderboards.

Models tested:

- `LiquidAI/LFM2-1.2B-GGUF:LFM2-1.2B-Q4_K_M.gguf`
- `LiquidAI/LFM2-2.6B-GGUF:LFM2-2.6B-Q4_K_M.gguf`
- `LiquidAI/LFM2-8B-A1B-GGUF:LFM2-8B-A1B-Q4_0.gguf`
- `unsloth/Phi-4-mini-instruct-GGUF:Phi-4-mini-instruct-Q3_K_M.gguf`
- `unsloth/Qwen3.5-4B-GGUF:Qwen3.5-4B-UD-IQ2_XXS.gguf`
- `unsloth/gemma-4-E2B-it-GGUF:gemma-4-E2B-it-UD-IQ2_M.gguf`
- `unsloth/gemma-4-E4B-it-GGUF:gemma-4-E4B-it-UD-IQ2_M.gguf`
- `unsloth/Qwen3.6-35B-A3B-GGUF:Qwen3.6-35B-A3B-UD-IQ1_M.gguf`

Additional NVIDIA generation candidate tested separately:

- Run path: `output_data/model_bench/quality_nvidia_nano`
- Model: `nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF:NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf`
- Result: 55/60, 91.7%, passed 6/6.
- Clean passes: finance relevance, source refusal, JSON structured boundary.
- Minor failures: weak Brandon/source framing heuristic, XML role/workflow coverage, missing core model citations heuristic.
- Interpretation: this is now a top-tier local generation candidate alongside `gemma-4-E2B` and `Qwen3.6-35B-A3B`; it is much faster than Qwen3.6 and better structured than the LiquidAI models in this pass.

Combined generation summary including NVIDIA Nano:

1. `gemma-4-E2B-it` - 93.3%, passed 6/6.
2. `NVIDIA-Nemotron-3-Nano-4B` - 91.7%, passed 6/6.
3. `Qwen3.6-35B-A3B` - 88.3%, passed 5/6; very slow, failed JSON but passed XML/citation.
4. `Phi-4-mini-instruct` - 80.0%, passed 5/6.
5. `gemma-4-E4B-it` - 78.3%, passed 4/6.
6. `LFM2-2.6B` - 75.0%, passed 4/6.
7. `LFM2-1.2B` - 71.7%, passed 4/6.
8. `LFM2-8B-A1B` - 68.3%, passed 4/6.
9. `Qwen3.5-4B` - 66.7%, passed 3/6; dominated and removed from VM cache.

Combined top role candidates:

- Brandon brief: `LFM2-1.2B`, `LFM2-8B-A1B`, `LFM2-2.6B` by current heuristic; manually compare against `gemma-4-E2B` and NVIDIA Nano in expanded eval because the heuristic underweights insight quality.
- Finance relevance: `LFM2-1.2B`, `LFM2-2.6B`, `Phi-4-mini-instruct`; NVIDIA Nano and `gemma-4-E2B` also scored 10/10 and should remain in the expanded finance slice.
- Source refusal: `LFM2-2.6B`, NVIDIA Nano, `gemma-4-E2B`.
- JSON structured boundary: NVIDIA Nano, `Phi-4-mini-instruct`, `gemma-4-E2B`.
- XML/tagged structured boundary: `Qwen3.6-35B-A3B`, `gemma-4-E2B`, NVIDIA Nano.
- Citation discipline: `gemma-4-E4B`, `Qwen3.6-35B-A3B`, then the 8-point acceptable group; NVIDIA Nano is in the acceptable group but not best by current citation heuristic.

Overall v2 ranking after rescoring with the 10-point cap:

1. `gemma-4-E2B-it` - 93.3%, passed 6/6.
2. `Qwen3.6-35B-A3B` - 88.3%, passed 5/6; very slow, failed JSON but passed XML/citation.
3. `Phi-4-mini-instruct` - 80.0%, passed 5/6; strongest JSON boundary candidate.
4. `gemma-4-E4B-it` - 78.3%, passed 4/6; strong citation, weaker structured stability than E2B in this run.
5. `LFM2-2.6B` - 75.0%, passed 4/6; strong finance/refusal, weak structure.
6. `LFM2-1.2B` - 71.7%, passed 4/6; fastest strong prose/finance candidate, weak structure.
7. `LFM2-8B-A1B` - 68.3%, passed 4/6; passed prose/finance/refusal/citation but did not beat smaller Liquid models enough yet.
8. `Qwen3.5-4B` - 66.7%, passed 3/6; dominated in this run and removed from VM cache.

Top role candidates from v2:

- Brandon brief: `LFM2-1.2B`, `LFM2-8B-A1B`, `LFM2-2.6B`.
- Finance relevance: `LFM2-1.2B`, `LFM2-2.6B`, `Phi-4-mini-instruct` with `gemma-4-E2B` also tied on score.
- Source refusal: `LFM2-2.6B`, `gemma-4-E2B`, `gemma-4-E4B`, `Qwen3.6`; `Qwen3.5-4B` scored well here but is removed because it was dominated overall.
- JSON structured boundary: `Phi-4-mini-instruct`, `gemma-4-E2B`; other current candidates failed unconstrained JSON.
- XML structured boundary: `Qwen3.6-35B-A3B`, `gemma-4-E2B`.
- Citation discipline: `gemma-4-E4B`, `Qwen3.6-35B-A3B`, then LiquidAI models as acceptable but weaker.

Current pruning:

- Removed `unsloth/Qwen3.5-4B-GGUF` from VM cache after v2 because it was dominated across roles and used about 2.1 GB.
- Removed `nvidia/llama-nemotron-embed-1b-v2` cache after it failed the retrieval component test with process exit 137 on the 8 GB/no-swap VM.
- Removed `LiquidAI/LFM2-8B-A1B-GGUF` after expanded v1 because it did not beat `LFM2-2.6B`, `gemma-4-E2B`, or NVIDIA Nano on the harder role benchmark and consumed several GB.
- Removed `nvidia/omni-embed-nemotron-3b` cache after the current project dependencies could not load it (`sentence_transformers.base` missing). It was only a tiny partial cache, but it is not a viable VM candidate yet.
- Do not prune `Qwen3.6-35B-A3B` yet despite slowness because it is top-2 for XML and citation; treat as a reference/quality candidate pending expanded pass.
- Re-restored `unsloth/Qwen3.6-35B-A3B-GGUF:Qwen3.6-35B-A3B-UD-IQ1_M.gguf` after a final cache audit showed it missing; it loaded and answered a 2-token smoke prompt successfully.

## Retrieval Component Benchmark v1

Added local script:

- `tools/retrieval_component_bench.py`

Run on VM:

- Path: `output_data/model_bench/retrieval_components_v1`
- Synthetic slices: finance company retrieval, local model retrieval, curation/data-flywheel retrieval, Phoenix/RAG eval retrieval, unsupported refusal retrieval, citation-verification retrieval.

Results:

1. `BAAI/bge-m3`: ok, hit@1 1.0, hit@3 1.0, MRR 1.0.
2. `sentence-transformers/all-MiniLM-L6-v2`: ok, hit@1 0.875, hit@3 1.0, MRR 0.938; missed the curation query at rank 1.
3. `google/embeddinggemma-300m`: failed, gated repo access denied for current HF account on VM.
4. `nvidia/omni-embed-nemotron-3b`: failed under current environment with `ModuleNotFoundError: No module named 'sentence_transformers.base'`; it appears to require a newer/dev Sentence Transformers API than the project currently has.
5. `nvidia/llama-nemotron-embed-1b-v2`: downloaded but process exited 137 during load/encode on the 8 GB/no-swap VM, likely OOM for this runtime.

Retrieval interpretation:

- Keep BGE-M3 as the retrieval baseline for now; it is already cached and won the small retrieval component benchmark.
- MiniLM remains a lightweight fallback, but it loses some curation nuance.
- NVIDIA Omni/Nemotron embeddings are not production-usable on this VM without dependency/runtime work or more memory.
- EmbeddingGemma cannot be evaluated until the HF account has accepted gated access.
- The proper webchat/inline-citation benchmark still needs to run end-to-end against the actual retriever, context packer, generator, and citation verifier. This component benchmark only tests embedding retrieval.

## RAG/Webchat Benchmark v1

Local retrieval-only run:

- Command: `uv run python tools/rag_webchat_bench.py --retrieval-only --out-dir output_data/model_bench/rag_webchat_local_retrieval_v1`
- Output: `output_data/model_bench/rag_webchat_local_retrieval_v1/rag_webchat_summary.json`
- BGE-M3 cold-load first query took about 144 seconds locally; subsequent queries were sub-second.
- Average required-term hit rate: 0.416.
- Coverage gates passed: 3/5.

Per-slice retrieval gates:

- `finance_firms`: failed, hit rate 0.091. Only `payments` matched; no J.P. Morgan/Citadel/Mastercard/Visa/Balyasny/Arrowstreet/Acadian/hedge fund/asset-manager hits in top sources. This means the current DB/retriever cannot yet support Brandon's finance-company news slice reliably.
- `local_models`: passed, hit rate 0.778. Found llama.cpp/Qwen/Gemma/Liquid/NVIDIA/local/CPU; missed GGUF and Phi.
- `data_flywheel_eval`: failed, hit rate 0.111. Only fine-tuning matched; Phoenix/Arize/NeMo Curator/Data Flywheel/LLaMA-Factory/eval/trace were not retrieved. This means the current source corpus is missing or failing to retrieve the tooling/curation evidence the workflow now cares about.
- `unsupported_exact_claim`: passed its retrieval gate because refusal cases do not require high positive coverage, but the top sources still matched generic subscription/price/tier terms without Anthropic/terms/clause support. This is the intended abstention setup.
- `citation_webchat`: passed, hit rate 0.6. Found model/source/eval but missed retrieval/citation terms.

Interpretation:

- The retrieval-agent benchmark should gate generation on retrieval coverage. Generating a finance or data-flywheel answer from weakly retrieved context would conflate retriever failure with model failure.
- Before running local model generation for the failed slices, either backfill/source fresh finance and tooling documents or adjust benchmark cases to current DB contents. For product convergence, prefer backfilling because those slices are explicit product requirements.
- End-to-end generation still needs to run on the VM for the passed slices and after finance/tooling retrieval is improved.

VM RAG/webchat generation run after retrieval fixes:

- Source: `output_data/model_bench/rag_webchat_focus_k10_rescored_citations/rag_webchat_summary.json` on the VM.
- Cases: finance_firms, unsupported_exact_claim, citation_webchat.
- Retrieval gates: 3/3 passed.
  - Finance hit rate improved to 0.727, with J.P. Morgan / Citadel / Mastercard / Visa / Balyasny / Arrowstreet / hedge fund covered; still missing Acadian, asset manager, and payments.
  - Unsupported exact Anthropic claim hit rate 0.167 by design; it only retrieved Anthropic without the exact subscription/terms/price/tier/clause support, so the correct model behavior is abstention.
  - Citation webchat hit rate 0.2, just at the current threshold.
- Generation leaderboard:
  1. `LiquidAI/LFM2-2.6B-GGUF:LFM2-2.6B-Q4_K_M.gguf`: 100.0%, passed 3/3, avg 48.292s.
  2. `unsloth/gemma-4-E2B-it-GGUF:gemma-4-E2B-it-UD-IQ2_M.gguf`: 76.7%, passed 2/3, avg 124.036s. Failed unsupported exact claim due no valid inline citation / `[N]` malformed citation / did not refuse unsupported question; citation case had weak answer term coverage.
  3. `nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF:NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf`: 66.7%, passed 2/3, avg 80.741s. Failed or weakened on citations by outputting `[N]` placeholders and weak answer term coverage.
- Per-slice:
  - Finance: LFM2-2.6B 10/10, Gemma E2B 10/10, NVIDIA Nano 8/10.
  - Refusal: LFM2-2.6B 10/10, NVIDIA Nano 7/10, Gemma E2B 5/10.
  - Citation: LFM2-2.6B 10/10, Gemma E2B 8/10, NVIDIA Nano 5/10.
- Interpretation: for this RAG/webchat slice, LFM2-2.6B is the current production leader. This is a different result than the model-only structured/citation benchmark, which reinforces that each workflow role needs its own leaderboard.

RAG/webchat citation-support scoring update:

- `tools/rag_webchat_bench.py` now includes a lightweight citation-support heuristic in addition to marker/range checks.
- It extracts finance/model/retrieval/citation entities and proper nouns from cited sentences and cited source text, then reports support precision.
- Unsupported/refusal cases are not penalized for citation support when the correct behavior is abstention.
- Existing VM RAG outputs were rescored locally:
  - Input: `output_data/model_bench/rag_webchat_focus_k10_rescored_citations`.
  - Output: `output_data/model_bench/rag_webchat_focus_k10_support_rescore_v2`.
  - LFM2-2.6B remains 100.0%, passed 3/3.
  - Gemma E2B remains 80.0%, passed 2/3.
  - NVIDIA Nano remains 70.0%, passed 2/3.
- Caveat: this is still a heuristic support check. The final benchmark should use gold source IDs/spans and/or the hybrid citation verifier/reranker pair scoring.

## GraphRAG / LightRAG Notes

LightRAG (`HKUDS/LightRAG`, arXiv 2410.05779, EMNLP 2025) is relevant because our failing or brittle retrieval slices are entity/relation-heavy:

- Brandon finance slice depends on relations between AI news, companies, roles, compliance/risk, and named finance firms.
- Citation verification depends on claim-to-source attribution, not just semantic chunk similarity.
- Curation/data-flywheel slice depends on tool relationships: traces -> eval datasets -> fine-tuning -> validators -> deployment.

Ideas to adopt without replacing the whole stack immediately:

- Add an entity/relation sidecar index over `tweets`, `web_articles`, `youtube_segments`, and curated Claude logs.
- Extract entities and relationships for people, companies, models, tools, workflows, finance domains, and citation claims.
- Use graph expansion before vector retrieval for named-entity queries such as "Citadel", "Balyasny", "J.P. Morgan", "Phoenix", "NeMo Curator", "LLaMA-Factory", "GGUF", and "citation verifier".
- Keep BGE-M3 vector retrieval as the text retriever, but combine it with graph-neighborhood candidates and rerank the merged set.
- Return retrieved contexts and graph edges in benchmark artifacts so we can measure context precision/recall and citation support.
- Do not adopt LightRAG blindly on the 8 GB VM. Graph extraction itself needs an LLM and can create noisy edges. Start with deterministic/entity-regex extraction plus optional local-model relation extraction, then benchmark against the current RAG gates.

Citation-specific LightRAG takeaways:

- LightRAG added citation functionality in March 2025 for source attribution and document traceability.
- LightRAG also added returning retrieved contexts for RAGAS/context-precision evaluation in November 2025.
- These are useful for our inline-citation webchat because they make source provenance and context precision first-class benchmark artifacts.
- However, LightRAG citation support should not replace our citation verifier. It can improve source selection and traceability, but claim-level citation correctness still needs sentence/claim-to-source verification, entity overlap, semantic/NLI checks, and invalid-citation stripping.
- Adoptable citation design:
  - Store source/chunk IDs and graph entity/relation IDs with each retrieved context.
  - Require answer citations to reference retrieved source IDs, not arbitrary `[N]` placeholders.
  - Score citation precision/recall using returned contexts plus our existing `verify_citations_hybrid()` logic.
  - Add a graph-backed citation candidate step: for each claim sentence, retrieve candidate sources from both vector similarity and entity/relation neighborhood, then verify.

Implementation sequence:

1. Add a lightweight `source_entities` / `source_relations` SQLite sidecar.
2. Backfill finance and tooling entities from existing DB rows plus curated Claude/context docs.
3. Modify retrieval to union keyword, vector, and graph-neighbor candidates before final ranking.
4. Add retrieval benchmark columns for entity recall and relation recall.
5. Only then consider a full LightRAG deployment if the sidecar shows measurable lift.

## Reranker / Tournament Notes

Rerankers can help this repo, but should not directly replace Elo/debate ranking yet.

Current tournament path:

- `linkedin_autopilot.py` orchestrates source selection, variant generation, citation parse/verify/correct, QE scoring, proximity deduplication, evolution, Elo tournament, and persistence.
- `agents/news_selector.py` uses recency/engagement plus MMR diversity and BGE-M3/MiniLM embeddings.
- `agents/variant_generator.py` handles source-aware citation prompts and hybrid citation verification.
- `agents/qe_agent.py` uses LLM JSON scoring.
- `agents/elo_ranker.py` runs random pairings and confidence-weighted Elo updates.
- `agents/debate_agent.py` uses an LLM debate prompt for pairwise preference.

Best reranker integration points:

1. Source selection before generation:
   - retrieve a larger candidate pool, rerank against Brandon/news/finance relevance, then MMR the top candidates.
2. Citation verification:
   - score `(claim sentence, source chunk)` with a reranker/cross-encoder in addition to entity overlap and BGE-M3 similarity.
3. Chat/RAG retrieval:
   - rerank the merged keyword + vector + chunk candidates before selecting final `max_sources`.
4. Candidate pruning before Elo:
   - use reranker/rubric scores as a cheap prefilter or tie-breaker before expensive LLM debates.
5. Debate augmentation:
   - pass source-support, novelty, and finance-relevance scores into debate prompts. Do not let the reranker decide winners alone.

Poor fit:

- Direct Elo replacement. Generic rerankers optimize relevance, not hook quality, originality, authenticity, or Brandon usefulness.

Candidate OSS rerankers to test:

- `BAAI/bge-reranker-v2-m3`: Apache 2.0, about 568M, best first candidate because the repo already uses BGE-M3 ideas.
- `Qwen/Qwen3-Reranker-0.6B`: Apache 2.0, instruction-aware, likely viable for small pools.
- `mixedbread-ai/mxbai-rerank-base-v2`: Apache 2.0, practical commercial-safe candidate.
- `zeroentropy/zerank-1-small-reranker`: Apache 2.0, heavier at about 1.7B but worth offline testing if latency budget allows.
- `zeroentropy/zembed-1-embedding`, `zeroentropy/zerank-2-reranker`, and `jinaai/jina-reranker-v2-base-multilingual` are interesting but have license/resource caveats for production.

Provider reranker availability checked:

- LiquidAI/LFM: no official LiquidAI reranker found on Hugging Face. Use Liquid models for generation/tool roles, not reranking, unless community examples appear.
- Qwen: official rerankers exist: `Qwen/Qwen3-Reranker-0.6B`, `4B`, `8B`, and newer VL rerankers. The 0.6B text reranker is the practical CPU candidate.
- NVIDIA/Nemotron: no obvious official text reranker found in the current HF search. NVIDIA has strong embeddings, but reranking may need other providers for now.
- Google/Gemma: no obvious official Google Gemma reranker found. BAAI has Gemma-based reranker variants, but license and runtime must be checked.
- Microsoft/Phi: no obvious official Phi reranker found.
- BAAI: multiple BGE rerankers; `bge-reranker-v2-m3` is the current best first candidate.
- ZeroEntropy: `zerank-1-small-reranker` is Apache 2.0 and should be tested; `zerank-2-reranker` and `zerank-1-reranker` have non-commercial license constraints.

Initial reranker component result:

- Added `tools/reranker_component_bench.py`.
- Local run with `BAAI/bge-reranker-v2-m3`:
  - retrieval MRR: 1.0
  - retrieval hit@1: 1.0
  - citation pairwise accuracy: 1.0
  - total cold run: 61.039s
- Local run with `zeroentropy/zerank-1-small-reranker`:
  - retrieval MRR: 1.0
  - retrieval hit@1: 1.0
  - citation pairwise accuracy: 1.0
  - total cold run: 201.682s
  - interpretation: quality looks strong on the small synthetic bench, but it is much slower than BGE and needs resource testing on the VM before production use.
- Local run with `Qwen/Qwen3-Reranker-0.6B` through generic `CrossEncoder`:
  - retrieval MRR: 0.723
  - retrieval hit@1: 0.625
  - citation pairwise accuracy: 0.417
  - warning: classification head was newly initialized, so this is likely a harness/loader misuse result, not a fair Qwen reranker result. Retest with Qwen's recommended reranker scoring method before counting it out.
- This remains a synthetic component bench, not production proof. Next reranker step is a production-shaped reranker eval over real retrieved source/citation pairs.

Required evals before integration:

- Extend `tools/retrieval_component_bench.py` to support cross-encoder rerankers: retrieve top N with current hybrid search, rerank, report nDCG@5/10, MRR, hit@1/3, and latency.
- Add citation-support pair evals: positives from verified citations, hard negatives from weak/misattributed citations.
- Add tournament replay eval: compare reranker top picks to Elo winners, QE winners, and eventually published/liked outcomes.
- Add shadow-mode telemetry: log reranker scores without changing winners, then inspect disagreements.
- Track load time, per-pair latency, memory, total tournament latency, and disagreement rate.

Caveat:

- The reranker should augment entity-overlap checks, not replace them. Otherwise the old brand/entity misattribution issues can return.

## Community Fine-Tunes / Prior Art To Test

Structured-output and tool-calling candidates found online:

- `google/functiongemma-270m-it`: tiny function-calling model focused on structured tool calls. Useful as a parser/router candidate, not as the main summarizer.
- `unsloth/functiongemma-270m-it` and `lmstudio-community/functiongemma-270m-it-GGUF`: practical packaging candidates for local testing.
- NVIDIA NeMo AutoModel FunctionGemma tutorial: useful recipe for fine-tuning FunctionGemma on xLAM-style function calling.
- `ermiaazarkhalili/Qwen3.5-2B-Function-Calling-xLAM-Unsloth-GGUF` and `ermiaazarkhalili/Qwen3.5-0.8B-Function-Calling-xLAM-Unsloth-GGUF`: testable GGUFs trained on Salesforce xLAM function-calling data.
- `NovachronoAI/LFM2.5-1.2B-Nova-Function-Calling-GGUF`: testable LiquidAI function-calling GGUF, potentially relevant because LFM2-2.6B currently leads RAG/webchat.
- `bastienp/Gemma-2-2B-Instruct-structured-output`: Gemma adapter trained on `paraloq/json_data_extraction`.
- `paraloq/json_data_extraction`: useful benchmark/fine-tuning dataset for restricted JSON extraction.
- Smaller/community JSON extraction repos exist for Qwen/Gemma, but many are older, low-download, narrow-domain, or adapter-only. Treat them as benchmark ideas unless they have GGUFs or easy merge paths.

Takeaway:

- For parser/router/control-plane roles, test FunctionGemma, Qwen3.5 xLAM, and LFM2.5 Nova Function Calling before training our own adapter.
- For our own fine-tune, use community data patterns but train/evaluate on our workflow-specific schemas: Brandon brief metadata, citation decision, retrieval/citation trace, source-grounded refusal, and finance relevance.

## Prompt Optimization / DSPy / GEPA

Use DSPy/GEPA concepts before fine-tuning:

- DSPy is useful as a way to express the workflow as modules/signatures and evaluate each module against metrics.
- GEPA is relevant because it evolves prompts from structured execution traces and textual feedback, while maintaining a Pareto frontier instead of one global prompt.
- This maps well to our slices:
  - Brief prompt should optimize for usefulness, no influencer language, citation density, and second-order insight.
  - Refusal prompt should optimize for abstaining on insufficient evidence.
  - Citation prompt should optimize for valid inline citations and source support.
  - Structured prompt should optimize for schema semantic correctness after constrained decoding handles syntax.
- Do not let automated prompt optimization hide uncontrolled variables. Each prompt experiment must keep model, retrieval set, decoding settings, and eval dataset fixed.
- Phoenix traces should be the source of GEPA/DSPy feedback: failed parse messages, invalid citation IDs, missing finance entities, unsupported claim labels, latency, and human-review notes.
- Use Pareto frontier selection like the SAGE post: discard dominated model/prompt pairs, keep tradeoff candidates where one is meaningfully faster/cheaper and another is meaningfully higher quality.
- Current structured-output evidence supports this: a manual DSPy/GEPA-style prompt/schema clarity pass improved `Phi-4-mini` production contracts from one strict true pass to 3/4 passes and 97.1%.
- DSPy/GEPA can still help with tiny models when the failure is ambiguous instructions, missing examples, unclear schema semantics, or poor prompt wording. It cannot fix llama.cpp JSON-schema sampler errors, missing model capacity, or a model consistently choosing the wrong semantic label after clear instruction.
- A targeted few-shot hard-negative pass fixed the remaining Phi citation-verifier failure on the seed production contracts. This supports using prompt/example optimization before fine-tuning for structured-output semantics.

Implementation sequence:

1. Encode current prompts as named versions.
2. Run baseline traces for top models on the golden dataset.
3. Try manual one-variable prompt variants first, especially few-shot examples for small models.
4. Add DSPy/GEPA only after the metrics are stable enough to optimize automatically.
5. Promote the best prompt per model/slice, not one universal prompt for all roles.

## Constrained JSON Benchmark v1

Added script:

- `tools/constrained_output_bench.py`

VM run:

- Output: `output_data/model_bench/constrained_json_v1`.
- Method: llama.cpp `--json-schema-file`, two cases: workflow plan and citation decision.

Results:

1. `unsloth/Phi-4-mini-instruct-GGUF:Phi-4-mini-instruct-Q3_K_M.gguf`: 94.4%, passed 2/2, avg 54.83s. One semantic miss on `primary_model_role`.
2. `LiquidAI/LFM2-2.6B-GGUF:LFM2-2.6B-Q4_K_M.gguf`: 22.2%, passed 0/2.
3. `nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF:NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf`: 22.2%, passed 0/2.
4. `unsloth/gemma-4-E2B-it-GGUF:gemma-4-E2B-it-UD-IQ2_M.gguf`: 22.2%, passed 0/2.

Important caveat:

- The failed models emitted `Error: Failed to initialize samplers: std::exception` with return code 0. That suggests a llama.cpp JSON-schema/sampler compatibility issue for those models or schemas, not ordinary model refusal.
- Phi-4-mini is therefore the current constrained-JSON leader for this exact llama.cpp method, but the next check should try simpler schemas, a GBNF grammar, and/or a newer llama.cpp build before permanently counting out Nemotron/Gemma/LFM for constrained decoding.

Template-aware function-calling rerun:

- Script supports `--template-mode chat` now, using llama.cpp conversation mode with the model's chat template while keeping `--json-schema-file`.
- Run: `output_data/model_bench/constrained_json_function_calling_chat_v1` on the VM.
- Results:
  1. `lmstudio-community/functiongemma-270m-it-GGUF:functiongemma-270m-it-F16.gguf`: 94.4%, passed 2/2, avg 18.889s.
  2. `unsloth/Phi-4-mini-instruct-GGUF:Phi-4-mini-instruct-Q3_K_M.gguf`: 94.4%, passed 2/2, avg 90.791s.
  3. `NovachronoAI/LFM2.5-1.2B-Nova-Function-Calling-GGUF:LFM2.5-1.2B-Nova-Function-Calling.Q4_K_M.gguf`: 11.1%, passed 0/2, llama.cpp sampler initialization error.
  4. `ermiaazarkhalili/Qwen3.5-2B-Function-Calling-xLAM-Unsloth-GGUF:Qwen3.5-2B-Function-Calling-xLAM-Unsloth.Q4_K_M.gguf`: 11.1%, passed 0/2, llama.cpp sampler initialization error.
- Interpretation: FunctionGemma is now a strong tiny parser/control-plane candidate for constrained JSON. Phi remains strong but much slower. LFM2.5 Nova and Qwen3.5 xLAM are not counted out for function calling yet because they failed the llama.cpp JSON-schema sampler path; they still need native function-call-template or unconstrained JSON/tool-call tests.

Benchmark validity review:

- The current model-only brief/finance/refusal/citation benchmarks are screening tests, not deployment gates.
- The RAG/webchat benchmark is closest to the target boundary because it uses `ChatAgent._retrieve_sources()`, but local artifacts must include VM summaries or the local review evidence is incomplete.
- Current structured schemas are toy contracts. Replace them with actual workflow contracts before using them for fine-tuning decisions:
  - QE review JSON from `agents/qe_agent.py`: score, breakdown, feedback, strengths, issues.
  - Debate result JSON from `agents/debate_agent.py`: arguments, winner, reasoning, confidence.
  - LinkedIn QA review JSON from `linkedin_autopilot.py`: approved, score, issues, summary.
  - Citation verification result from `agents/variant_generator.py`: citation, sentence, source, similarity, entity_overlap, status, reason.
  - Proposed route decision for local workflow: answer_from_sources, refuse_insufficient_sources, run_retrieval, verify_citations, qa_review.
- Add JSON Schema or Pydantic validation to constrained-output scoring. Parsing plus top-level-key checks is not enough.
- For final model choice, build production-shaped golden datasets:
  - daily brief realism cases with 10-20 mixed retrieved items, duplicates, stale items, low-value announcements, finance distractors, and one or two high-value finance-relevant items;
  - finance evidence cases with named firms, dates, metrics, model-risk/compliance implications, and negative examples;
  - retrieval cases with gold source IDs/spans and recall@k/MRR/nDCG/entity recall;
  - citation cases with sentence-level source support;
  - refusal hard negatives: no evidence, partial evidence, conflicting evidence, stale-date claims, exact-price/contract claims, and adversarial "answer anyway" prompts.

Production-shaped constrained JSON run:

- Script: `tools/constrained_output_bench.py`.
- Run: `output_data/model_bench/constrained_json_production_contracts_v1` on the VM.
- Method: `--case-set production --template-mode chat`.
- Cases:
  - `qa_review_result`: mirrors app QA review fields.
  - `citation_verification_result`: mirrors citation verifier fields.
  - `retrieval_gate_result`: generation gate for weak retrieval.
  - `route_decision`: local workflow routing/refusal decision.
- Results:
  1. `unsloth/Phi-4-mini-instruct-GGUF:Phi-4-mini-instruct-Q3_K_M.gguf`: 85.7%, passed 3/4, avg 45.836s.
  2. `lmstudio-community/functiongemma-270m-it-GGUF:functiongemma-270m-it-F16.gguf`: 71.4%, passed 2/4, avg 12.782s.
- Interpretation:
  - The earlier toy structured benchmark overstated FunctionGemma's general structured-verifier quality.
  - Phi-4-mini is the current production-shaped structured-output leader.
  - FunctionGemma remains useful as a fast gate/router/parser candidate, but not yet as the citation-verification or QA-review specialist.
  - This supports a constellation approach: use FunctionGemma for cheap/simple routing only when validation passes; use Phi for harder structured verifier/review contracts.

Strict production-contract rescore and prompt fix:

- Tightened `tools/constrained_output_bench.py` so schema errors and critical semantic misses cannot pass.
- Added clearer production-contract prompts:
  - QA score is explicitly 1-10, not a percentage.
  - Citation verification explicitly says the cited source alone does not support J.P. Morgan analyst assistants.
  - Retrieval gate explicitly says missing required named entities means `query_supported=false` and `coverage_passed=false`.
- VM artifact: `output_data/model_bench/constrained_json_production_contracts_promptfix_top2_v1/constrained_summary.json`.
- Results:
  - `Phi-4-mini`: 97.1%, passed 3/4, avg 24.7s. Remaining failure: citation verifier still chose `verified` instead of `weak`/`invalid`.
  - `FunctionGemma 270M`: 74.3%, passed 1/4, avg 10.9s. Still too weak for QA/citation/route semantics.
- Decision:
  - Use `Phi-4-mini` as the current structured/control-plane candidate.
  - Keep FunctionGemma only for simple gate/router experiments with strict validation.
  - Do not fine-tune structured output yet; next try expanded production traces and validation-error retry before training.

Few-shot verifier result:

- Added `--verifier-few-shot` to `tools/constrained_output_bench.py`.
- The variant appends hard-negative/positive examples only to the citation-verification contract.
- VM artifact: `output_data/model_bench/constrained_json_production_contracts_phi_fewshot_v1/constrained_summary.json`.
- `Phi-4-mini`: 100.0%, passed 4/4, avg 26.5s.
- Correct citation-verifier output:
  - `status`: `invalid`
  - `entity_overlap`: `[]`
  - reason: source supports Gemma local model availability only; it does not mention J.P. Morgan or analyst assistants.
- Structured-output decision: `Phi-4-mini` plus constrained JSON plus few-shot hard negatives is good enough for the current seed production contracts. FunctionGemma remains a fast simple-gate candidate only. Fine-tuning is not justified for this slice until expanded production traces still fail after constrained decoding, validation-error retry, and prompt/example optimization.

Expanded structured/control-plane benchmark:

- Added production contracts for:
  - finance relevance classification;
  - citation correction action;
  - final delivery payload for Brandon's background news summary.
- Added `--validation-retry` to `tools/constrained_output_bench.py`.
  - Retry is one extra constrained generation with validation errors in the prompt.
  - This mirrors the production strategy: schema-constrain, validate, retry once with concrete errors, then reject/escalate.
- VM artifacts:
  - `output_data/model_bench/constrained_json_production_contracts_v2_phi_retry/constrained_summary.json`
  - `output_data/model_bench/constrained_json_production_contracts_v2_functiongemma_retry/constrained_summary.json`
- Results:
  - `Phi-4-mini`: 7/7, 100.0%, avg 30.2s with constrained JSON, verifier few-shot, and validation retry.
  - `FunctionGemma 270M`: 4/7, 86.6%, avg 22.7s. Failures were citation verifier JSON, unsupported-source route semantics, and delivery payload social-copy boolean semantics.
- Interpretation:
  - Use `Phi-4-mini` as the structured/control-plane model.
  - Keep FunctionGemma only for simple low-risk gates where strict validation can reject bad outputs.
  - No structured-output fine-tune is justified yet; add held-out production traces and human-reviewed labels before training.

Human-review packet for held-out eval labels:

- Added `tools/build_human_review_packets.py`.
- Built `output_data/gold_eval/human_review_v1` from:
  - `production_expanded_v2` gold cases;
  - `production_expanded_v2_generation_top3_compact` model outputs;
  - `production_expanded_v2_generation_phi_citation_retry` backup outputs.
- Artifacts:
  - `answer_review.jsonl`: 24 answer-level rows.
  - `citation_span_review.jsonl`: 167 sentence-level rows.
  - `review_summary.json`.
  - `review_packet.md`.
- Packet contents:
  - Models: LFM2-2.6B, Phi-4-mini, and counted-out NVIDIA Nano for negative/failure contrast.
  - Slices: finance-domain signal, pre-fine-tune curation, local-model ops, and source-grounded refusal.
  - Citation rows include both cited sentences and supported-case uncited sentences tagged as `missing_citation_candidate`.
- Use:
  - Label answer acceptability/usefulness/factual consistency.
  - Label each citation sentence as supported, partial, unsupported, or out_of_range.
  - Copy minimal supporting source spans.
  - Keep reviewed rows as held-out eval data until an explicit train/holdout split is made.
- This reduces the current benchmark gap from "no span labels exist" to "review labels need to be filled in."
- Added `tools/build_review_label_splits.py` to keep reviewed holdout rows separate from any future training candidates.
- Current split artifact: `output_data/gold_eval/human_review_v1_splits/split_summary.json`.
- Current split result:
  - Reviewed rows: 0.
  - Candidate training rows pending approval: 0.
  - Unlabeled or invalid rows: 191.
  - Unlabeled by slice: pre-finetune system improvements 63, finance domain signal 72, local model ops 43, source-grounded refusal 13.
- Training implication: the copied/distilled logs and review packet are not yet sufficient for fine-tuning. They are usable for labeling and regression construction, but no rows should be used for SFT/preference tuning until human labels exist and an explicit approval split is generated.

Local chat backend wiring:

- `agents/chat_agent.py` now supports `CHAT_BACKEND=llama_cpp`.
- Hosted Bedrock/Strands remains available with `CHAT_BACKEND=bedrock` or unset.
- Local backend env used on VM smoke:
  - `CHAT_BACKEND=llama_cpp`
  - `CHAT_LLAMA_MODEL=LiquidAI/LFM2-2.6B-GGUF:LFM2-2.6B-Q4_K_M.gguf`
  - `CHAT_LLAMA_CLI=~/opt/llama.cpp/llama-cli`
  - `CHAT_MAX_TOKENS=140-180`
  - `CHAT_LLAMA_THREADS=6`
  - `CHAT_ENABLE_RERANKER=1`
  - `CHAT_CONTEXT_MAX_SOURCES=3`
  - `CHAT_CONTEXT_SOURCE_CHARS=450`
- Local backend behavior:
  - Uses the same retrieval/rerank/context-compression path as hosted chat.
  - Runs llama.cpp through `llama-cli`.
  - Emits the generated text as a token event for the existing SSE/non-streaming endpoints.
  - Retries with stricter numeric citation instructions when a supported answer has no valid citations.
  - Keeps stderr/timing data out of the user-visible answer.
- VM smoke result:
  - Query: "What AI news mentions Balyasny, J.P. Morgan, Mastercard, Visa, or investment research?"
  - Backend: `llama_cpp`.
  - Model: LFM2-2.6B Q4_K_M.
  - Artifact: `output_data/model_bench/local_chat_backend_smoke/local_chat_backend_smoke.json`.
  - Result: pass, elapsed 63.5s on CPU, BGE reranker applied, `citations_count=4`, no llama.cpp timing leakage.
- This wires the main Flask/webchat `ChatAgent` path away from hosted Claude when env is set. Separate AgentCore deployed agents still contain legacy Bedrock/Strands code, but are now disabled by default unless explicitly re-enabled.

Hosted model migration audit:

- Added `tools/audit_hosted_model_paths.py`.
- Artifact: `output_data/model_bench/hosted_model_migration_audit/hosted_model_migration_audit.json`.
- Summary: 12 scanned files still have hosted/stale references, but the statuses are now resolved as local-backed code/docs or guarded legacy paths:
  - `local_backend_available`: 2 files.
  - `legacy_runtime_guarded`: 7 AgentCore runtime files.
  - `hosted_deployment_guarded`: 1 deploy script.
  - `local_docs_updated`: 2 docs files.
  - Current hit counts: 37 Bedrock, 25 Strands, 15 Claude/Anthropic, and 5 Gemini/Google AI references.
- Main local path: `agents/chat_agent.py` has a verified local backend, but still contains hosted fallback code for `CHAT_BACKEND=bedrock`.
- Guarded hosted/default paths:
  - `deploy/agentcore/chat/main.py`, `generator_agent/main.py`, `qe_agent/main.py`, `evolution_agent/main.py`, `debate_agent/main.py`, `ranking/main.py`, and `ranking_orchestrator/main.py` return disabled responses by default unless `ALLOW_LEGACY_AGENTCORE_RUNTIME=1` is set.
  - Strands and Boto3 imports in those AgentCore files are now lazy, so disabled runtimes do not require legacy hosted-model dependencies just to import the module.
  - `deploy/agentcore/deploy.sh` refuses normal hosted deploys unless `ALLOW_HOSTED_AGENTCORE_DEPLOY=1` is set. `--destroy` remains available.
  - `README.md` and `DEPLOY_HETZNER.md` describe local llama.cpp defaults and smoke checks; residual Anthropic/Claude text is historical or explicitly negative, not runtime setup.
- Decision: keep AgentCore disabled for the Brandon workflow unless it is rewritten to call Hetzner-local llama.cpp endpoints using the current model constellation.
- Shared `agents/llm_client.py` has been moved to local-by-default:
  - `LLM_BACKEND=llama_cpp`/`local` uses llama.cpp.
  - Prose default: `LFM2-2.6B`.
  - JSON/control-plane default: `Phi-4-mini`.
  - Bedrock remains available only by explicitly setting `LLM_BACKEND=bedrock`.
  - VM smoke artifact: `output_data/model_bench/llm_client_local_smoke/llm_client_local_smoke.json`.
  - VM smoke result: `call_llm_json()` returned validated JSON `{"route": "refuse_insufficient_sources", "confidence": 0.9}` through local llama.cpp/Phi.

Expanded benchmark next targets:

- Prose/brief/finance:
  - `LFM2-1.2B`
  - `LFM2-2.6B`
  - `gemma-4-E2B`
  - Include `LFM2-8B-A1B` only if checking whether the larger Liquid MoE adds qualitative insight beyond smaller LFMs.
- Refusal/grounding:
  - `LFM2-2.6B`
  - `gemma-4-E2B`
  - `gemma-4-E4B`
  - Add hard-negative variants and preference-style chosen/rejected labels before deciding SFT/DPO.
- Structured boundaries:
  - JSON: `Phi-4-mini-instruct`, `gemma-4-E2B`.
  - XML/tags: `Qwen3.6-35B-A3B`, `gemma-4-E2B`.
  - Also test llama.cpp constrained JSON/GBNF or LLGuidance before concluding fine-tuning is required.
- Citation discipline:
  - `gemma-4-E4B`
  - `Qwen3.6-35B-A3B`
  - `gemma-4-E2B`
  - Include the existing deterministic citation verifier in the end-to-end retrieval benchmark.
- Storage-aware stress candidates not yet benchmarked:
  - `unsloth/gpt-oss-20b-GGUF` or `ggml-org/gpt-oss-20b-GGUF`; about 11.5-12.1 GB, likely needs swap or one-at-a-time test.
  - `unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF`; smallest observed UD file about 5.6 GB, plausible one-at-a-time stress candidate.
  - Do not download DeepSeek V3.2 or Kimi K2.6 on this VM; dry-run sizes are hundreds of GB or more.

## Expanded Benchmark v1

Run completed on the VM at `output_data/model_bench/expanded_v1`. Results copied to host:

- `output_data/model_bench_expanded_v1_results.jsonl`
- `output_data/model_bench_expanded_v1_summary.json`

Expanded benchmark design:

- Two harder cases per role.
- Runs only the per-role candidates from v2 instead of every model on every task.
- Roles: brief, finance, refusal, structured, citation.

Expanded role results:

- Brief:
  - Top: `LFM2-2.6B` and `gemma-4-E2B`, both 80%.
  - `LFM2-1.2B` and `LFM2-8B-A1B` both 70%.
  - Interpretation: base models are usable for brief generation, but citation coverage and second-order insight are still brittle. Prompt/template improvements may be enough before fine-tuning.
- Finance:
  - Top: `gemma-4-E2B`, 90%.
  - Next: `LFM2-1.2B`, `Phi-4-mini-instruct`, and `LFM2-2.6B` around 80%, though `LFM2-2.6B` failed one of two expanded finance tasks.
  - Interpretation: `gemma-4-E2B` is best current base for finance-role synthesis. Fine-tuning is not immediately required, but more finance eval cases and retrieval-agent tests are needed.
- Refusal:
  - All tested models passed missing-source refusal and failed the nuanced conflicting-sources / insufficient-proof refusal.
  - Affected models: `LFM2-2.6B`, `gemma-4-E2B`, `gemma-4-E4B`, `Qwen3.6-35B-A3B`.
  - Interpretation: larger models did not solve nuanced source-grounded refusal. This needs deterministic verifier logic plus targeted hard-negative SFT/preference data if the verifier model must make this judgment.
- Structured:
  - `Phi-4-mini-instruct` and `Qwen3.6-35B-A3B` each passed one of two harder structured tasks; `gemma-4-E2B` passed neither.
  - XML did not generalize: all candidates failed the harder XML agent contract.
  - Interpretation: do not rely on prompt-only JSON or XML. Use constrained decoding / schema validation / retry loops first. Fine-tune only if semantic field quality remains poor after constraints.
- Citation:
  - Top: `Qwen3.6-35B-A3B`, 95%, passed both citation tasks but was very slow.
  - Next: `LFM2-1.2B`, 85%.
  - `gemma-4-E2B` and `gemma-4-E4B` both 80%.
  - Interpretation: base models are usable for citation-format discipline, but final correctness should remain deterministic via source/citation verifier. Qwen3.6 is a quality reference, not automatically production due disk/runtime.

Expanded base-vs-fine-tune assessment:

- Brief generation: base model is likely enough with prompt/template improvements. Best candidates: `LFM2-2.6B`, `gemma-4-E2B`.
- Finance synthesis: base model is likely enough for first deployment if retrieval supplies finance-relevant context. Best candidate: `gemma-4-E2B`; keep `LFM2-1.2B` for cheaper/fast fallback.
- Missing-source refusal: base models are enough for simple absent-fact refusal.
- Nuanced unsupported-source refusal / insufficient-proof refusal: base models are not enough. Use deterministic claim/support checks and create hard-negative SFT or DPO/KTO examples.
- Structured output: do not fine-tune first. Use llama.cpp grammar / JSON schema / GBNF or LLGuidance, validation, and retry. Re-benchmark after constrained decoding.
- Citation discipline: base models are enough for format, not for guarantee. Keep deterministic citation verification. Use fine-tuning only if citation placement remains weak after verifier-guided prompting.

Literature alignment:

- Structured output findings align with JSONSchemaBench and constrained decoding work: prompt-only structured output is unreliable, while constrained decoding improves syntactic/schema validity but does not guarantee semantic correctness.
- Refusal findings align with RAGTruth, Self-RAG, and CRAG-style work: retrieval grounding reduces hallucination but does not remove unsupported or overgeneralized claims; verification/reflection and hard-negative evals are needed.
- Instruction-following slices align with IFEval-style exact-rule checks, but expanded prompts should continue to perturb wording to avoid fixed-prompt overfitting.

## NVIDIA Nemotron 3 Nano Result

After auditing VM cache, a cached NVIDIA text model was found and benchmarked:

- Model: `nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF:NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf`
- Cache size: about 2.7 GB.
- v2 quality benchmark: 83.3%, passed 6/6.
- Expanded structured/refusal/citation benchmark:
  - Structured: 75%, passed both harder structured tasks.
  - Citation: 95%, passed both citation tasks.
  - Refusal: 80%, passed simple missing-source refusal, failed nuanced conflicting-sources refusal.

Impact on current model selection:

- Structured boundaries: Nemotron 3 Nano becomes the best current base candidate, ahead of Phi/Qwen/Gemma, but still needs schema validation or constrained decoding because it had semantic/schema misses.
- Citation discipline: Nemotron matches Qwen3.6 expanded citation score with much less runtime and disk, so it is a better production candidate than Qwen3.6 for citation-format roles.
- Refusal: Nemotron does not solve nuanced insufficient-proof refusal; this remains a deterministic verifier plus hard-negative fine-tuning/preference-data problem.
- Qwen3.6 can now be demoted to quality-reference / optional large-model candidate unless manual output review shows materially better prose quality.

## Storage-Aware Stress Tests

Mistral Small 3.2:

- Candidate: `unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF:Mistral-Small-3.2-24B-Instruct-2506-UD-IQ1_S.gguf`.
- First smoke appeared to pass, but the fixed harness showed it was a false pass caused by a context-size error.
- Retried with `--ctx 2048`; timed out at 900 seconds without a valid answer.
- Removed from VM cache after timeout to recover disk.
- Current conclusion: count out for this 8 GB no-swap VM. It can remain a larger-VM/reference candidate.

GPT-OSS 20B:

- Candidate tested: `ggml-org/gpt-oss-20b-GGUF:gpt-oss-20b-mxfp4.gguf`.
- Made room by removing the slow `Qwen3.6-35B-A3B` cache after Nemotron displaced it for structured/citation roles.
- Smoke failed with return code 1.
- Failure reason: llama.cpp could not allocate a 9.7 GB CPU repack buffer on the 8 GB no-swap VM.
- Removed GPT-OSS cache after failure; VM returned to about 25 GB free.
- Current conclusion: count out GPT-OSS 20B for this VM. Revisit only on a larger RAM VM or with explicit swap/storage provisioning.

Qwen3.6 cache status:

- Removed from VM cache to make room for GPT-OSS testing.
- It remains documented as a quality-reference result, but Nemotron 3 Nano is now the better production candidate for structured/citation roles on this VM.
- Handles Brandon news-summary prompts with citations and insights.
- Runs within acceptable background-agent latency.

Current benchmark harness records:

- output,
- runtime,
- return code,
- output length,
- rough word overlap against Claude reference.

Needed next:

- add a deterministic rubric grader,
- add manual spot-check set,
- add Brandon-news-specific eval prompts,
- run NeMo Curator or equivalent filters on the SFT/eval data,
- compare latest Qwen/Gemma/Liquid/Phi candidates under the same harness.

## Expanded Role Benchmark v1

Added/used VM script:

- `tools/expanded_model_bench.py`

Runs:

- Main expanded run: `output_data/model_bench/expanded_v1`
- NVIDIA Nano follow-up: `output_data/model_bench/expanded_nvidia_nano`

Important caveat:

- Expanded v1 uses the same heuristic scorer as `model_quality_bench.py`. It is useful for ranking obvious failures, but manual review and stricter validators are still needed before deployment.

Expanded role results:

- Brief:
  - `LFM2-2.6B`: 80%, passed 2/2.
  - `gemma-4-E2B`: 80%, passed 2/2.
  - NVIDIA Nano: 75%, passed 2/2.
  - `LFM2-1.2B` and `LFM2-8B-A1B`: 70%, passed 2/2 but weaker citation/second-order behavior.
- Finance:
  - `gemma-4-E2B`: 90%, passed 2/2.
  - `LFM2-1.2B`: 80%, passed 2/2.
  - `Phi-4-mini-instruct`: 80%, passed 2/2.
  - `LFM2-2.6B`: 80%, passed 1/2.
  - NVIDIA Nano: 70%, passed 1/2; missed finance citations and named-company coverage in the domain-model prompt.
- Refusal:
  - Main run: all tested models scored 80% and passed 1/2 because every model struggled with the conflicting-source refusal case.
  - NVIDIA Nano follow-up: 85%, passed 2/2, but still got a "did not refuse unsupported fact" warning on conflicting-source refusal.
  - Interpretation: unsupported-source refusal is trainable and should get hard-negative SFT/preference data plus deterministic guardrails.
- Structured:
  - Main run: `Phi-4-mini-instruct` and `Qwen3.6` both scored 55%, passed 1/2; `gemma-4-E2B` scored 40%, passed 0/2.
  - NVIDIA Nano follow-up: 75%, passed 2/2, but with schema warnings.
  - Interpretation: NVIDIA Nano is the current best unconstrained structured-output candidate, but constrained decoding/JSON schema retry should be tested before fine-tuning.
- Citation:
  - `Qwen3.6-35B-A3B`: 95%, passed 2/2.
  - NVIDIA Nano: 95%, passed 2/2.
  - `LFM2-1.2B`: 85%, passed 2/2.
  - `gemma-4-E2B` and `gemma-4-E4B`: 80%, passed 2/2.
  - Interpretation: Qwen3.6 and NVIDIA Nano are strongest on citation heuristics, but the production path should still use the deterministic citation verifier.

Current best constellation after expanded v1:

- Default summarizer/general brief: compare `LFM2-2.6B` and `gemma-4-E2B` with manual output review; keep NVIDIA Nano as a stronger but slower/less finance-focused fallback.
- Finance-domain summarizer: `gemma-4-E2B` first, `LFM2-1.2B` and `Phi-4-mini-instruct` as alternates; add more finance-specific eval rows before deciding.
- Unsupported/refusal specialist: NVIDIA Nano currently best in expanded run; still requires hard-negative training/guardrails.
- Structured-output/planning specialist: NVIDIA Nano first; test constrained JSON/GBNF/LLGuidance before fine-tuning.
- Citation/verifier assistant: NVIDIA Nano or `Qwen3.6-35B-A3B`; use Qwen3.6 only when slowness is acceptable and memory headroom is available.
- Retrieval embeddings: keep BGE-M3 baseline; NVIDIA Omni/Nemotron embeddings are not currently viable on this VM without dependency/memory work.

Fine-tuning decision after expanded v1:

- Base models look good enough to prototype the workflow with a model constellation plus validators.
- Fine-tuning is still likely useful for:
  - Brandon-specific brief style and second-order insight consistency.
  - finance-domain named-entity coverage and citation density.
  - unsupported/conflicting-source refusal.
  - strict schema outputs if constrained decoding/retry is insufficient.
- Do not fine-tune yet for retrieval. First run the end-to-end webchat/RAG benchmark and confirm whether failures are from retrieval, reranking, context packing, generation, or citation verification.

## RAG / Webchat Inline-Citation Benchmark v1

Added local script:

- `tools/rag_webchat_bench.py`

Purpose:

- Uses the real Python `ChatAgent._retrieve_sources()` path against `output_data/ai_news.db`.
- Uses the real chat context formatting with numbered sources.
- Replaces hosted Strands/Bedrock generation with local llama.cpp GGUF models to test CPU-only webchat behavior.

Runs on VM:

- Retrieval-only probe: `output_data/model_bench/rag_webchat_retrieval_probe`
- Full local generation run: `output_data/model_bench/rag_webchat_v1`

Retrieval-only result:

- Average required-term hit rate: 0.118 across five cases.
- Finance firms query: 0.0 hit rate.
- Local model query: 0.222 hit rate.
- Data flywheel / Phoenix / eval query: 0.0 hit rate.
- Unsupported Anthropic exact-claim query: 0.167 hit rate.
- Citation/webchat generic query: 0.2 hit rate.

Production DB term audit:

- The current `ai_news.db` has almost no searchable coverage for the new finance-role entities and workflow terms:
  - `jpmorgan`: 0 in web/tweets/youtube/tournament_sources.
  - `j.p. morgan`: 1 tweet, 1 tournament source.
  - `citadel`, `mastercard`, `balyasny`, `arrowstreet`, `acadian`, `data flywheel`, `nemo curator`, `llama-factory`, `gguf`, `liquidai`: effectively absent from chat-searched tables.
  - `qwen`, `gemma`, `nvidia`, `arize`, `llama.cpp` have some coverage.

Full local webchat generation run:

- Models:
  - NVIDIA Nano 4B
  - `gemma-4-E2B`
  - `LFM2-2.6B`
- Retrieval stayed weak, so this run primarily measures whether generators refuse or caveat correctly when relevant evidence is missing.

Generation summary:

- `LFM2-2.6B`: 94%, passed 5/5. Manual review: generally uses citations and says sources do not contain the requested finance/details. One unsupported Anthropic answer included an extra `[N]` artifact and was flagged for unsupported pricing/model wording.
- `gemma-4-E2B`: 84%, passed 4/5. Manual review: often refuses correctly when sources lack evidence, but sometimes no inline citations on refusals.
- NVIDIA Nano: 58%, passed 2/5. Manual review: conservative and usually says the sources do not support the claim, but often too terse and omits citations, so it fails inline-citation webchat behavior despite good refusal instincts.

RAG interpretation:

- The current blocker for Brandon's finance-facing webchat is source coverage and retrieval scope, not just the local generator.
- The Python chat agent searches `tweets`, `web_articles`, `youtube_videos`, and chunk tables. It does not search `tournament_sources`, where some model-family terms exist.
- Before judging local models for production webchat quality, ingest or collect finance-role sources and data-flywheel/fine-tuning sources into the chat-searched tables, or add a controlled fixture/eval corpus.
- For inline citations with local models, `LFM2-2.6B` is the most promising current CPU generator because it is fast, cites, and caveats weak retrieval. `gemma-4-E2B` is a strong alternate. NVIDIA Nano should remain the structured/planning/refusal specialist, but needs prompt or fine-tuning work to include citations in terse refusals.

Next RAG gates:

- Add finance-specific sources/RSS/search targets for JPMorgan, Acadian, Balyasny, Arrowstreet, Citadel, Mastercard, Visa, regulated AI, fintech/payments risk, model risk, and financial-services AI workflows.
- Add source coverage checks before webchat generation: if retrieval hit rate or lexical coverage is low, return a retrieval insufficiency message rather than asking the generator to improvise.
- Add end-to-end webchat cases after data ingestion and rerun `tools/rag_webchat_bench.py`.
- Evaluate reranking only after source coverage improves; current failures are too dominated by missing documents.

### RAG/Webchat Retrieval Fix and Sourced Rerun - 2026-05-25

Implemented and synced to VM:

- `tools/rag_webchat_bench.py` now has per-case retrieval coverage minimums and reports `coverage_passed`, `gate_reason`, and `missing_terms`.
- `agents/chat_agent.py` keyword retrieval now prioritizes finance/local-model/data-curation entity terms before generic query words and gives each explicit domain term a chance to contribute a source.
- `ai_news_scraper.py` now seeds finance/regulatory AI handles and source targets for J.P. Morgan, J.P. Morgan Asset Management, Mastercard, Visa, Acadian, and NVIDIA NeMo Curator. The overly broad blocked keyword `president` was narrowed to Trump-specific phrases because it filtered ordinary finance/newsroom titles.
- `tools/seed_curated_rag_sources.py` can explicitly seed curated URLs into `web_articles` and `article_paragraphs` for benchmark slices missed by normal feed pages.
- Cloudflare source constants were updated with the same finance/data-curation source targets; TypeScript now gets past the source-list type issue after making source boost sets `Set<string>`. A pre-existing `src/shared/embeddings.ts` Workers AI response typing issue remains.

Curated seed run on VM:

- Command: `.venv/bin/python tools/seed_curated_rag_sources.py`
- OpenAI Balyasny and Reuters/Investing mirror URLs returned HTTP 403 from the VM.
- Reachable seeded sources included J.P. Morgan AI Research, J.P. Morgan Asset Management AI, Mastercard AI topic page, Visa Newsroom, Acadian Our Edge, NVIDIA NeMo Curator docs/blog, Arize Phoenix RAG eval docs, and LLaMA-Factory data-prep docs.
- Result: 12 attempted sources, 8 AI-relevant before the Mastercard re-seed; 316 chunks from the first run. Mastercard re-seeded as AI-relevant after narrowing the blocked keyword.

Retrieval-only benchmark after fixes:

- Output: `output_data/model_bench/rag_webchat_after_priority_entity_fix`
- Average required-term hit rate improved from 0.118 to 0.431.
- Coverage passed: 5/5 cases.
- Finance firms: 0.455 hit rate; hits `jpmorgan`, `j.p. morgan`, `mastercard`, `visa`, `payments`. Still missing Citadel, Balyasny, Arrowstreet, Acadian, hedge fund, asset manager in the top retrieved set.
- Local models: 0.667 hit rate; hits `llama.cpp`, `qwen`, `gemma`, `liquid`, `nvidia`, `local`.
- Data flywheel/eval: 0.667 hit rate; hits `phoenix`, `arize`, `nemo`, `curator`, `fine-tuning`, `eval`.

Full local RAG/webchat generation after retrieval/source fixes:

- Output: `output_data/model_bench/rag_webchat_after_retrieval_fix`
- Retrieval coverage passed 5/5, so this run is a better webchat-model signal than the earlier weak-retrieval run.

Model summary:

- `LiquidAI/LFM2-2.6B-GGUF:LFM2-2.6B-Q4_K_M.gguf`: 96%, passed 5/5, average 51.9s. Best overall webchat/RAG generator in this run. Manual caveat: emitted an invalid placeholder citation `[N]` in the unsupported Anthropic refusal, so citation linting was tightened after this run.
- `nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF:NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf`: 90%, passed 5/5, average 69.6s. Best citation-specific case; still terse on unsupported refusal and omitted citations there.
- `unsloth/gemma-4-E2B-it-GGUF:gemma-4-E2B-it-UD-IQ2_M.gguf`: 84%, passed 4/5, average 97.8s. Strong finance and curation answers, but failed citation-specific case due no valid inline citations and weak answer-term coverage.

Slice leaders from sourced webchat run:

- Finance: 1. LFM2-2.6B, 2. Gemma 4 E2B, 3. NVIDIA Nemotron Nano.
- Local models: 1. LFM2-2.6B, 2. NVIDIA Nemotron Nano, 3. Gemma 4 E2B.
- Curation/data-flywheel: 1. LFM2-2.6B, 2. NVIDIA Nemotron Nano, 3. Gemma 4 E2B.
- Refusal: 1. LFM2-2.6B, 2. Gemma 4 E2B, 3. NVIDIA Nemotron Nano.
- Citation/webchat: 1. NVIDIA Nemotron Nano, 2. LFM2-2.6B, 3. Gemma 4 E2B.

Updated interpretation:

- Base models are sufficient for a first local CPU webchat deployment if retrieval supplies relevant context.
- For RAG/webchat, promote LFM2-2.6B to the default generator candidate and keep NVIDIA Nemotron Nano as the citation/structured/refusal specialist.
- Fine-tuning is still useful for citation hygiene, refusal citation style, and finance-entity specificity, not required just to make the system answer sourced questions.
- Remaining retrieval gap: source coverage for Citadel, Balyasny, Arrowstreet, hedge fund, and asset-manager terms is still incomplete because several direct article sources were blocked from the VM.

### Finance Coverage and Tight Citation Follow-Up - 2026-05-25

Additional implementation:

- `tools/seed_curated_rag_sources.py` now falls back through `https://r.jina.ai/<url>` for pages that block direct VM fetches.
- Added reachable finance sources:
  - `openai_balyasny_case`: OpenAI customer story for Balyasny Asset Management.
  - `balyasny_openai_feature`: Balyasny page pointing to the OpenAI customer story.
  - `citadel_ai_efc`: eFinancialCareers article on Citadel CTO Umesh Subramanian and AI/hedge-fund alpha.
  - `arrowstreet_ai_search`: Arrowstreet homepage as an initial entity anchor.
- `tools/rag_webchat_bench.py` now supports `--cases` for focused reruns, uses `--max-sources=10` by default to match production chat retrieval, and treats placeholder citations such as `[N]` as invalid.
- The finance RAG case query now explicitly names Balyasny, Arrowstreet, and Acadian because they were already required terms.
- `agents/chat_agent.py` now validates priority keyword hits with word boundaries, preventing short brands like `Visa` from matching unrelated substrings like `advantage`.

New source seeding on VM:

- Command: `.venv/bin/python tools/seed_curated_rag_sources.py --only openai_balyasny_case balyasny_openai_feature citadel_ai_efc arrowstreet_ai_search`
- Result: 4/4 sources AI-relevant, 40 chunks.
- OpenAI/Balyasny content is now available through the reader fallback.
- Citadel coverage now exists through eFinancialCareers. Direct Citadel and Reuters/Investing pages still block or require challenge handling from the VM.

Finance retrieval after boundary fix:

- Output: `output_data/model_bench/rag_webchat_finance_boundary_fix_k10`
- Finance hit rate improved to 0.727 with `max_sources=10`.
- Hit terms: `jpmorgan`, `j.p. morgan`, `citadel`, `mastercard`, `visa`, `balyasny`, `arrowstreet`, `hedge fund`.
- Still missing in top-10: `acadian`, `asset manager`, `payments`.

Focused generation rerun with `max_sources=10` and tight citation scorer:

- Output: `output_data/model_bench/rag_webchat_focus_k10_tight_citations`
- Cases: `finance_firms`, `unsupported_exact_claim`, `citation_webchat`.
- Retrieval coverage passed 3/3.

Focused run model summary:

- `LiquidAI/LFM2-2.6B-GGUF:LFM2-2.6B-Q4_K_M.gguf`: 93.3%, passed 3/3, average 48.3s. Leads focused webchat overall and citation case; finance 10/10, citation 10/10. Refusal initially scored 8/10 because the scorer missed "do not specify"; after phrase fix, its short rerun scored 7/10 due no valid inline citation and invalid `[N]` placeholder. Good default generator, but needs citation hygiene guardrail/fine-tune.
- `unsloth/gemma-4-E2B-it-GGUF:gemma-4-E2B-it-UD-IQ2_M.gguf`: 76.7%, passed 2/3, average 124s. Finance 10/10 and citation 8/10; unsupported refusal in the focused run failed due `[N]` placeholder/no valid inline citation, but a short refusal rerun scored 8/10 with no valid citation as the remaining issue. Strong finance writer, too slow and inconsistent for default webchat.
- `nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF:NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf`: 66.7%, passed 2/3, average 80.7s. Finance 8/10, refusal 7/10, citation 5/10 after placeholder citation linting. Earlier citation strength was overstated because `[N]` placeholders were not penalized.

Short refusal rerun after adding "do not specify" refusal wording:

- Output: `output_data/model_bench/rag_webchat_refusal_phrase_fix`
- Refusal slice order on that one case:
  1. Gemma 4 E2B: 8/10, no valid inline citations.
  2. LFM2-2.6B: 7/10, no valid inline citations and invalid `[N]`.
  3. NVIDIA Nemotron Nano: 7/10, terse and no citations.

Updated RAG/webchat interpretation:

- Default sourced webchat generator remains `LFM2-2.6B` because it is fastest among the top performers and wins the focused three-case run.
- Finance writer backup remains `gemma-4-E2B`, but it is much slower on longer sourced prompts.
- NVIDIA Nano should no longer be treated as the citation specialist until citation prompting or fine-tuning fixes `[N]` placeholders and missing inline citations.
- Unsupported-source refusal is not solved by any base model: all three can refuse semantically, but all still have citation-style failures. This is now the strongest fine-tuning/guardrail candidate slice.

### Benchmark Scoring Corrections - 2026-05-25

Additional benchmark fixes:

- `tools/rag_webchat_bench.py` now parses bracketed citation lists such as `[1, 2]` as valid citations while still rejecting alphabetic placeholders like `[N]`.
- Retrieval coverage checks now use word-boundary matching and aliases:
  - `asset manager` includes `asset management`.
  - `payments` includes `payment`.
  - `model` includes `models`.
  - `eval` includes `evaluation`/`evaluations`.
  - `retrieval` includes `rag`.
- Added `--rescore-from-dir` so scoring fixes can be applied to expensive existing generation outputs without rerunning llama.cpp.

Rescore result:

- Command: `.venv/bin/python tools/rag_webchat_bench.py --rescore-from-dir output_data/model_bench/rag_webchat_focus_k10_tight_citations --out-dir output_data/model_bench/rag_webchat_focus_k10_rescored_citations`
- `LFM2-2.6B` rescored to 100%, passed 3/3, because its unsupported-source answer used the valid wording "do not specify" and a valid numeric citation.
- `Gemma 4 E2B` stayed 76.7%, passed 2/3; still has `[N]` placeholder on unsupported refusal in the focused run.
- NVIDIA Nano stayed 66.7%, passed 2/3; still has no valid citations on unsupported refusal and `[N]` placeholders in the citation case.

Alias retrieval probe:

- Output: `output_data/model_bench/rag_webchat_alias_retrieval_probe_v2`
- Finance coverage is now 0.818 with top-10 sources.
- Hit terms: `jpmorgan`, `j.p. morgan`, `citadel`, `mastercard`, `visa`, `balyasny`, `arrowstreet`, `hedge fund`, `asset manager`.
- Still missing: `acadian`, `payments`.
- Citation-webchat retrieval passes at the minimum 0.2 through `eval`/`evaluation`; this case remains generator-focused rather than retrieval-rich.

Corrected RAG/webchat role conclusion:

- Default RAG/webchat generator: `LFM2-2.6B`.
- Finance synthesis: `LFM2-2.6B` first, `Gemma 4 E2B` second, NVIDIA Nano third.
- Inline citation generation: `LFM2-2.6B` first after corrected citation parsing; `Gemma 4 E2B` second; NVIDIA Nano third and currently not reliable enough as a citation specialist.
- Unsupported-source refusal: `LFM2-2.6B` is best on the focused output after corrected scoring, but this slice still needs guardrails/fine-tuning because reruns can produce `[N]` placeholders or uncited refusals.

### Consolidated Model Constellation Report - 2026-05-25

Added script:

- `tools/build_model_constellation_report.py`

Generated VM artifacts:

- `output_data/model_bench/constellation/model_constellation_report.json`
- `output_data/model_bench/constellation/model_constellation_report.md`

Copied host artifacts:

- `output_data/model_bench_constellation_report.json`
- `output_data/model_bench_constellation_report.md`

Evidence inputs:

- `output_data/model_bench/expanded_v1/expanded_summary.json`
- `output_data/model_bench/expanded_nvidia_nano/expanded_summary.json`
- `output_data/model_bench/rag_webchat_focus_k10_rescored_citations/rag_webchat_summary.json`
- `output_data/model_bench/rag_webchat_alias_retrieval_probe_v2/rag_webchat_summary.json`
- `output_data/model_bench/retrieval_components_v1/retrieval_component_summary.json`

Current deployment constellation:

- Default generator / sourced webchat / general brief: `LiquidAI/LFM2-2.6B-GGUF:LFM2-2.6B-Q4_K_M.gguf`
- Finance-writing backup: `unsloth/gemma-4-E2B-it-GGUF:gemma-4-E2B-it-UD-IQ2_M.gguf`
- Structured/planning candidate: `nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF:NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf`
- Retriever: `BAAI/bge-m3`
- Fallback retriever: `sentence-transformers/all-MiniLM-L6-v2`

Top-3 by task slice:

- General Brandon brief:
  1. `LFM2-2.6B`
  2. `Gemma 4 E2B`
  3. `NVIDIA Nemotron 3 Nano`
- Finance-domain signal/synthesis:
  1. `LFM2-2.6B`
  2. `Gemma 4 E2B`
  3. `NVIDIA Nemotron 3 Nano`
- Source-grounded refusal:
  1. `LFM2-2.6B`
  2. `NVIDIA Nemotron 3 Nano`
  3. `Gemma 4 E2B`
- RAG/webchat inline citations:
  1. `LFM2-2.6B`
  2. `Gemma 4 E2B`
  3. `NVIDIA Nemotron 3 Nano`
- Agentic planning / structured output:
  1. `NVIDIA Nemotron 3 Nano`
  2. `Phi-4-mini-instruct`
  3. `Qwen3.6-35B-A3B`
- Retrieval embeddings:
  1. `BAAI/bge-m3`
  2. `sentence-transformers/all-MiniLM-L6-v2`
  3. `google/embeddinggemma-300m` only as a blocked/gated candidate, not currently usable.

Current base-vs-fine-tune decision:

- Base models are sufficient for a first local CPU deployment if paired with retrieval coverage gates, citation verification, schema validation, and retry.
- Fine-tuning is not required before first deployment.
- Fine-tuning priority remains:
  1. Unsupported-source refusal with citation-grounded refusal style.
  2. Citation hygiene and `[N]` placeholder suppression.
  3. Brandon/finance style and second-order insight consistency.
  4. Semantic planning only after constrained decoding is tested.

Do not restore/prune guidance:

- Keep or retest: `LFM2-2.6B`, `Gemma 4 E2B`, `NVIDIA Nemotron 3 Nano`, `Phi-4-mini-instruct`, and `Qwen3.6-35B-A3B` only if disk allows a slow quality reference.
- Do not restore unless new evidence appears: `LFM2-8B-A1B`, Qwen3.5 small variants, Gemma 4 26B on this 8GB VM, NVIDIA llama-nemotron-embed-1b-v2 on the current no-swap VM.

Remaining gaps from the report:

- Acadian and payments are still missing from the finance top-10 retrieval probe.
- Citation-webchat retrieval passes only at the minimum gate; add richer citation/retrieval eval sources.
- Structured output still needs constrained decoding / GBNF / LLGuidance validation before production use.
- Production wiring from hosted Claude/Bedrock to a local llama.cpp service remains a separate implementation step if migration is in scope.

### Constrained Structured Planning Benchmark - 2026-05-25

Added script:

- `tools/constrained_structured_bench.py`

Purpose:

- Test the agentic-planning slice with actual llama.cpp constrained decoding instead of prompt-only JSON/XML.
- Use `llama-completion` in raw `-no-cnv` mode, not `llama-cli`, because `llama-cli` conversation mode prepends chat/thinking tokens that conflict with JSON-schema grammar sampling.
- Use bounded JSON Schema plus semantic scoring for required workflow responsibilities and checks.

Important runtime finding:

- `llama-cli --json-schema-file` failed at sampler initialization for chat-template models because the generation prefix included `<think>`/assistant template tokens before the JSON grammar.
- `llama-completion -no-cnv --json-schema-file` works.
- `--log-disable` must not be used with `llama-completion` in this harness because it suppresses the captured completion stream.
- Constrained decoding is materially slower than unconstrained prompting, but acceptable for background agentic planning.

VM artifacts:

- `output_data/model_bench/constrained_structured_v1/constrained_structured_summary.json`
- `output_data/model_bench/constrained_structured_v1/constrained_structured_results.jsonl`
- `output_data/model_bench/constrained_structured_v1/planning_schema.json`

Host artifacts:

- `output_data/model_bench/constrained_structured_v1/constrained_structured_summary.json`
- `output_data/model_bench/constrained_structured_v1/constrained_structured_results.jsonl`
- `output_data/model_bench/constrained_structured_v1/planning_schema.json`

Results:

1. `LiquidAI/LFM2-2.6B-GGUF:LFM2-2.6B-Q4_K_M.gguf`: 100%, passed, 91.7s.
2. `unsloth/Phi-4-mini-instruct-GGUF:Phi-4-mini-instruct-Q3_K_M.gguf`: 100%, passed, 126.9s.
3. `unsloth/gemma-4-E2B-it-GGUF:gemma-4-E2B-it-UD-IQ2_M.gguf`: 100%, passed, 142.3s.
4. `nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF:NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf`: 70%, failed semantic coverage, 293.0s. Produced schema-shaped JSON but missed the required planning labels/check names under raw constrained mode.
5. `unsloth/Qwen3.6-35B-A3B-GGUF:Qwen3.6-35B-A3B-UD-IQ1_M.gguf`: 60%, failed semantic coverage and six-step workflow count, 304.2s.

Updated structured/planning conclusion:

- For first deployment, use bounded JSON-schema constrained decoding for planning contracts.
- `LFM2-2.6B` is now the best measured structured/planning candidate and can serve as both default generator and constrained planner if keeping the constellation simple.
- `Phi-4-mini-instruct` is the best separate structured-planner backup if we want role separation.
- `Gemma 4 E2B` is a viable third planning candidate.
- NVIDIA Nano should be demoted for this specific constrained planning path despite prior prompt-only structured strength.
- Fine-tuning is not needed for planning before first deployment; add more constrained planning cases first. Fine-tune only if semantic role/tool choice fails after schema constraints and retries.

Regenerated consolidated report:

- VM: `output_data/model_bench/constellation/model_constellation_report.json`
- VM: `output_data/model_bench/constellation/model_constellation_report.md`
- Host: `output_data/model_bench_constellation_report.json`
- Host: `output_data/model_bench_constellation_report.md`

Current final constellation after constrained planning:

- Default generator / sourced webchat / general brief: `LiquidAI/LFM2-2.6B-GGUF:LFM2-2.6B-Q4_K_M.gguf`
- Finance-writing backup: `unsloth/gemma-4-E2B-it-GGUF:gemma-4-E2B-it-UD-IQ2_M.gguf`
- Structured/planning candidate: `LiquidAI/LFM2-2.6B-GGUF:LFM2-2.6B-Q4_K_M.gguf`
- Structured/planning backup: `unsloth/Phi-4-mini-instruct-GGUF:Phi-4-mini-instruct-Q3_K_M.gguf`
- Retriever: `BAAI/bge-m3`
- Fallback retriever: `sentence-transformers/all-MiniLM-L6-v2`

Remaining gaps after this update:

- Acadian and payments are still missing from the finance top-10 retrieval probe.
- Citation-webchat retrieval passes only at the minimum gate; add richer citation/retrieval eval sources.
- Structured output now passes one bounded constrained-schema planning contract for LFM2-2.6B, Phi-4-mini, and Gemma 4 E2B, but needs more planning cases and a production wrapper around `llama-completion` raw mode.
- Production wiring from hosted Claude/Bedrock to a local llama.cpp service remains a separate implementation step if deployment migration is in scope.

### Fine-Tuning / Data Status - 2026-05-25

VM production database copy:

- The host `output_data/ai_news.db` is an old local DB: 29 MB, modified 2026-03-06.
- The VM production DB is 70 MB, modified 2026-05-25, and has now been copied to the host as `output_data/ai_news.vm.db`.
- Use `output_data/ai_news.vm.db` for production-shaped gold dataset construction unless explicitly testing the old local fixture DB.
- Current VM DB row counts after copy: tweets=1617, web_articles=867, youtube_videos=1902, article_paragraphs=1621, youtube_segments=0.

Unsloth CPU-only fine-tuning conclusion:

- Do not plan to fine-tune with Unsloth on the Hetzner CPU-only VM.
- Current Unsloth docs distinguish CPU-only support for chat/data-recipes/export from training support; training is GPU-oriented, with Studio training currently NVIDIA-first and other backends evolving.
- Final deployment can still run CPU-only GGUF via llama.cpp. The recommended path is: train LoRA/QLoRA on a GPU with Unsloth or LLaMA-Factory, merge/export, convert/quantize to GGUF, then deploy the quantized GGUF on the Hetzner CPU VM.
- MLX fine-tuning is viable for Apple-Silicon experiments, but it is not the cleanest production path when the serving target is Linux CPU llama.cpp. Prefer GPU fine-tuning plus GGUF export for reproducibility.

State-of-AI / prior RAG research reuse:

- No repo named `state-of-ai` / `stateofai` was found under `/Users/erki/sundai` in the current search.
- Useful local RAG research exists in nearby repos:
  - `6s093-social-media-agent/workshop-5-frontend/rag.py`: simple BM25 + semantic hybrid retrieval with normalized scores.
  - `hackathon-tv5/apps/agentdb/simulation/docs/reports/latent-space/MASTER-SYNTHESIS.md`: dynamic-k, graph/HNSW, self-healing, and reranking-style retrieval ideas.
- For this repo, the reusable ideas are weight sweeps for BM25/dense hybrid search, dynamic top-k based on query/source coverage, recall/MRR/citation-support telemetry, and periodic index-health checks. Do not import speculative neural-vector claims directly into production without a small controlled benchmark.

### Production Gold Eval Seed - 2026-05-25

Added:

- `tools/build_production_gold_eval.py`
- `tools/production_gold_bench.py`

Artifacts:

- `output_data/gold_eval/production_seed_v1.jsonl`
- `output_data/gold_eval/production_seed_v1_openai_messages.jsonl`
- `output_data/gold_eval/production_seed_v1_summary.json`
- `output_data/gold_eval/production_seed_v1_bench/production_gold_summary.json`
- `output_data/gold_eval/production_seed_v1_bench_parent_recall/production_gold_summary.json`
- `output_data/gold_eval/production_seed_v1_retrieval_k5/production_gold_summary.json`
- `output_data/gold_eval/production_seed_v1_retrieval_k15/production_gold_summary.json`
- `output_data/gold_eval/production_seed_v1_generation_curation_top3_smoke/production_gold_summary.json`
- `output_data/gold_eval/production_seed_v1_generation_rag_top3_singleturn_v2/production_gold_summary.json`
- `output_data/model_bench/production_decision/production_model_decision.json`
- `output_data/model_bench/production_decision/production_model_decision.md`

Dataset summary:

- 8 seed cases from the copied VM DB, backed by 34 unique production sources.
- Case types: 3 retrieval, 3 provided-source RAG generation, 2 citation-pair checks.
- OpenAI-message export exists only as reviewed seed material; it is not yet sufficient or approved for final SFT.

Cheap-gate results:

- Retrieval: 3/3 pass after fixing the scorer to use parent-document recall for production webchat.
  - The previous curation/eval "failure" was a granularity mismatch: gold cases contained paragraph IDs, while production retrieval returns web article IDs.
  - At `max_sources=10`: finance parent recall 0.5, curation parent recall 0.667, local-model parent recall 0.3.
  - At `max_sources=15`: finance parent recall 0.6, curation parent recall 0.667, local-model parent recall 0.5.
  - At `max_sources=5`: only 1/3 retrieval cases pass; this is too narrow for finance and local-model coverage.
  - Recommendation: retrieve 10-15 candidates, then compress/rerank to top 3 for local generation.
- Citation pair support: 2/2 pass after tightening the positive Phoenix tracing/eval fixture.

VM provided-source generation smoke:

- Initial timeout result was a harness bug: `llama-cli` stayed in interactive mode after answering.
- Fixed `tools/production_gold_bench.py` to add `--single-turn`, source limiting, per-source char limiting, prompt diagnostics, source-compressed expected-term scoring, and explicit thinking-leak detection.
- Micro LFM rerun with 1 source, 80 chars, `ctx=512`, `max_tokens=32`: pass in 12.6s.
- Corrected top-3 RAG generation run with 3 sources, 250 chars/source, `ctx=1024`, `max_tokens=128`:
  - `LFM2-2.6B`: 3/3 pass, avg 21.3s.
  - `Phi-4-mini`: 3/3 pass, avg 36.8s.
  - `Gemma 4 E2B`: 2/3 raw pass, avg 61.5s, but 0/3 under strict user-facing gate because it leaks `[Start thinking]` and fails unsupported-source refusal.
- Prompting Gemma with "Do not include hidden reasoning..." did not fix the thinking leak on the unsupported-source case.
- Current production-seed conclusion: base models are good enough for this seed RAG generation slice using LFM2-2.6B or Phi-4-mini plus source compression. Gemma is demoted for webchat/refusal unless a model-specific reasoning-off template is found.

Reranked service-shaped RAG generation:

- Updated `tools/rag_webchat_bench.py` to match the service path: retrieve 15 candidates, optionally rerank with `BAAI/bge-reranker-v2-m3`, compress to top 3 context sources, and run local GGUF generation.
- Fixed a benchmark prompt ambiguity: the old prompt said cite with `[N]`, which caused some models to literally output `[N]`. The prompt now asks for numeric citations such as `[1]` or `[2]`.
- Fixed llama.cpp output capture to tolerate non-UTF-8 bytes with replacement instead of aborting the run.
- Adjusted unsupported-refusal scoring so a correct refusal is not penalized for lacking inline citations.
- VM artifact copied locally: `output_data/model_bench/rag_webchat_reranked_top3_numeric_citations_rescored/rag_webchat_summary.json`.
- VM results after reranking + numeric-citation prompt:
  - `LFM2-2.6B`: 100.0%, 5/5 pass, avg 29.0s.
  - `Phi-4-mini`: 92.0%, 5/5 pass, avg 39.4s.
  - `NVIDIA Nano`: 82.0%, 4/5 pass, avg 36.3s; failed local-model RAG due length/citation/coverage.
- Per-slice leaders:
  - Finance: `LFM2-2.6B` 10/10, then NVIDIA Nano and Phi at 9/10.
  - Local models: `LFM2-2.6B` 10/10, Phi 8/10, NVIDIA Nano 5/10 fail.
  - Curation/data flywheel: all three reached 10/10, with LFM fastest.
  - Refusal: LFM and Phi 10/10, NVIDIA Nano 9/10.
  - Citation: LFM 10/10, Phi 9/10, NVIDIA Nano 8/10.
- Current RAG decision: base models are good enough for this seed service-shaped RAG slice when paired with BGE-M3 retrieval, BGE reranking, top-3 source compression, numeric citation prompting, and deterministic citation/refusal scoring. Do not fine-tune this slice before expanding the gold set with sentence/span citation labels and harder production traces.

Production model decision artifact:

- Added `tools/build_production_model_decision.py`.
- Current recommendation:
  - default local generator: `LiquidAI/LFM2-2.6B-GGUF:LFM2-2.6B-Q4_K_M.gguf`
  - webchat backup: `unsloth/Phi-4-mini-instruct-GGUF:Phi-4-mini-instruct-Q3_K_M.gguf`
  - structured/control-plane model: `unsloth/Phi-4-mini-instruct-GGUF:Phi-4-mini-instruct-Q3_K_M.gguf`
  - simple gate/router candidate: `lmstudio-community/functiongemma-270m-it-GGUF:functiongemma-270m-it-F16.gguf`
  - avoid for webchat/refusal until fixed: `unsloth/gemma-4-E2B-it-GGUF:gemma-4-E2B-it-UD-IQ2_M.gguf`
  - retriever: `BAAI/bge-m3`
  - reranker: `BAAI/bge-reranker-v2-m3`
- Regenerated decision artifact after reranked RAG and Phi few-shot structured runs:
  - `output_data/model_bench/production_decision/production_model_decision.json`
  - `output_data/model_bench/production_decision/production_model_decision.md`
- Current slice decisions:
  - RAG/webchat: `LFM2-2.6B` primary, 100.0%, 5/5; `Phi-4-mini` backup, 92.0%, 5/5.
  - Structured/control-plane: `Phi-4-mini` primary. It passed both the expanded seed contracts and the held-out production-derived contracts.
- Held-out structured/control-plane evidence:
  - Added `tools/build_heldout_production_traces.py`.
  - Added `--cases-jsonl` to `tools/constrained_output_bench.py`.
  - Artifacts:
    - `output_data/gold_eval/heldout_production_traces_v1/heldout_traces.jsonl`
    - `output_data/gold_eval/heldout_production_traces_v1/heldout_summary.json`
    - `output_data/model_bench/constrained_json_heldout_production_traces_v1_phi_functiongemma/constrained_summary.json`
    - `output_data/model_bench/constrained_json_heldout_production_traces_v1_lfm/constrained_summary.json`
  - Trace mix: 16 held-out structured contracts from real expanded-v2 gold cases: finance relevance, delivery payload, unsupported-source route, retrieval gate, and citation verifier.
  - Top-three held-out result:
    - `Phi-4-mini`: 16/16, 100.0%, avg 61.7s.
    - `FunctionGemma 270M`: 7/16, 82.1%, avg 32.0s.
    - `LFM2-2.6B`: 0/16, 10.6%, avg 4.5s, failed through llama.cpp sampler initialization errors on the JSON-schema path.
  - Decision: use Phi for structured/control-plane; FunctionGemma is no longer a general simple-gate candidate beyond trivial validated routes; LFM2 is counted out for constrained JSON control-plane under current llama.cpp.
- Fine-tuning needed before first local deployment: `False` for current seed evidence.
- Added `tools/model_regression_gate.py` as the current deployment gate:
  - Local and VM result: `deployment_gate_passed=true`.
  - `fine_tune_ready=false` because the human-review split still has zero reviewed rows and zero approved training candidates.
  - Hard checks cover LFM2-2.6B RAG primary, Phi citation-retry backup, Phi held-out structured control-plane, counted-out NVIDIA Nano RAG, counted-out FunctionGemma general structured use, counted-out LFM2 JSON-schema path, local ChatAgent smoke, local shared `llm_client` smoke, and hosted runtime guard status.
  - The gate now emits a `model_roster` that records keep/count-out status by task, so future model tests can be compared against the same deployment roles instead of a single blended score.
  - Added `tools/local_llm_client_smoke.py` so the VM can generate `output_data/model_bench/llm_client_local_smoke/llm_client_local_smoke.json` with a real local Phi JSON call instead of relying on mocked tests.
- Remaining before marking goal complete: human-label the review packet, keep the model regression gate passing, add more held-out traces as production behavior changes, and keep the disabled AgentCore/hosted paths guarded unless they are rewritten to call Hetzner-local services.
- Added `tools/benchmark_coverage_audit.py` to audit benchmark maturity by workflow slice.
  - Current result: deployment/model gate passes and benchmark coverage passes after adding more `data_curation_eval` cases, but `human_label_ready=false` because the review split still has zero reviewed rows and zero approved training candidates.
  - `structured_control_plane` coverage is counted through all held-out structured contracts, not only rows literally tagged with that slice.
  - The audit is non-blocking for first local deployment but blocks any claim that fine-tuning readiness is complete.
- Added `tools/model_slice_scoreboard.py` to aggregate current model pass rates and average scores per workflow slice and role/case type from the available result artifacts. Use the role-grouped view for decisions; the raw slice-level view can mix RAG and structured-control-plane checks.
  - The scoreboard now emits a per-role `comparative_score_pct`, rank, and recommendation (`keep_primary`, `keep_candidate`, `investigate_or_repair`, `count_out_for_role`).
  - Comparative score is quality-first: `70 * pass_rate + 30 * avg_score_pct/100`. Runtime remains visible but is only a tie-breaker because this background workflow can tolerate slow models if quality converges.
  - Current VM role leaders:
    - `data_curation_eval` RAG: `LFM2-2.6B` and `Phi-4-mini` both pass the currently scored curation generation case; `NVIDIA Nano` is counted out for this role.
    - `finance_domain_signal` RAG: `LFM2-2.6B` leads; `Phi-4-mini` remains an investigate/repair backup; `NVIDIA Nano` is counted out.
    - Structured finance relevance, delivery payload, citation verification, and retrieval gate: `Phi-4-mini` leads.
    - Unsupported-source route: `FunctionGemma 270M` and `Phi-4-mini` both pass; keep FunctionGemma only for this narrow validated route, not general structured control-plane.
    - Source-grounded refusal generation: `LFM2-2.6B` leads by current pass/failure status; keep deterministic insufficient-source gates because refusal average-score fields are less meaningful than pass/fail here.
  - Authoritative VM artifacts:
    - `output_data/model_bench/slice_scoreboard/model_slice_scoreboard.{json,md}`
    - `output_data/model_bench/production_decision/production_model_decision.{json,md}`

### Chat Service Source Compression - 2026-05-25

Implemented the production-seed retrieval recommendation in `agents/chat_agent.py`:

- Default retrieval candidate count is now `CHAT_RETRIEVAL_MAX_SOURCES` with default `15`.
- Optional reranking now runs between retrieval and source compression when `CHAT_ENABLE_RERANKER=1`.
  - Default model: `BAAI/bge-reranker-v2-m3`.
  - Override with `CHAT_RERANKER_MODEL`.
  - Limit candidates with `CHAT_RERANKER_MAX_SOURCES`, default `15`.
  - The reranker is lazy-loaded and disabled by default to avoid surprise downloads or memory use.
- Prompt context is compressed through `_select_context_sources()` with `CHAT_CONTEXT_MAX_SOURCES` default `3`.
- Source text in prompt context is clipped by `_build_context()` using `CHAT_CONTEXT_SOURCE_CHARS` default `250`.
- Citation extraction now maps against the compressed context sources so `[N]` citations match prompt numbering.
- When reranking is applied, context compression respects reranker order.
- The `sources` stream event now reports the compressed context sources and includes `retrieved_source_count`, `reranked`, and `reranker_model` for observability.
- If the reranker fails to load or score, the service emits a warning and falls back to base retrieval order.

Validation:

- `python3 -m py_compile agents/chat_agent.py test_chat_agent.py`
- `uv run python test_chat_agent.py`
- Added tests for context source selection, source text clipping, reranker ordering, and reranker fallback.

Important implementation note:

- The service path now has candidate widening, optional reranking, and source compression.
- Synced `agents/chat_agent.py`, `tools/production_gold_bench.py`, and `test_chat_agent.py` to the VM.
- VM validation:
  - `python3 -m py_compile agents/chat_agent.py tools/production_gold_bench.py test_chat_agent.py`
  - `.venv/bin/python test_chat_agent.py`
- VM production-seed retrieval-context comparison:
  - Baseline `max_sources=15`, top-3 heuristic context: candidate retrieval 3/3, compressed context 2/3, avg context recall 0.256, avg context MRR 0.5.
  - `BAAI/bge-reranker-v2-m3`, top-3 context: candidate retrieval 3/3, compressed context 3/3, avg context recall 0.322, avg context MRR 1.0.
  - Reranker fixed the compressed evidence pack for finance and improved local-model context from 1 to 2 parent-source hits.
- VM storage after adding the reranker: `/dev/sda1` has about 15 GB free; `~/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3` uses about 2.2 GB.
- Next benchmark step: rerun the top provided-source/local generation models through the service-shaped RAG path with `CHAT_ENABLE_RERANKER=1` and compare answer/citation/refusal scores against the source-compression baseline.

### Brandon Bulletin Style and Source Coverage - 2026-05-26

Updated product target:

- Brandon does not want narrative LinkedIn-style summaries. He wants terse internal news bulletins.
- Prioritize things he may not already have seen on x.com:
  - YouTube/video drops and demos;
  - model releases, GGUF/local-model availability, and SLM provider updates;
  - genuinely novel papers, benchmarks, eval harnesses, and datasets;
  - GitHub/project releases, RAG tooling, data curation, governance, and enterprise AI;
  - conference deadlines, calls for papers, Boston/NYC AI events, and major frontier-lab events.
- De-emphasize generic X discourse, influencer hooks, hashtags, engagement questions, motivational framing, and broad "AI is transforming X" filler.

Prompt/ranking updates:

- `agents/chat_agent.py` now frames the local RAG assistant as Brandon's source-grounded news scout and defaults to compact bulletins.
- `agents/variant_generator.py` now generates bulletin variants by focus area rather than viral LinkedIn hook style.
- `agents/qe_agent.py` now scores missed-news value, novelty/source quality, source grounding, finance/workflow relevance, bulletin format, and actionability.
- `agents/debate_agent.py` now ranks candidates by Brandon usefulness rather than virality.
- `agents/evolution_agent.py` now rewrites toward compact cited bulletins and removes narrative/viral mechanics.
- `tools/local_chat_backend_smoke.py` now checks for bulletin style, novelty terms, citations, and narrative-filler hits.

Scraper coverage updates:

- `ai_news_scraper.py` now includes more source categories for:
  - frontier lab releases: OpenAI, Anthropic, Google DeepMind, Meta AI;
  - SLM/model providers: Mistral, Hugging Face, Liquid AI, Qwen;
  - benchmark/eval sources: MLCommons, lm-evaluation-harness releases, Stanford HELM;
  - enterprise AI governance: Singapore IMDA AI Verify, Singapore agentic AI governance framework, NIST AI RMF;
  - conferences/events/CFPs: ConferenceDeadlines, MIT CSAIL, MIT calendar, NYU CDS.
- `youtube_channel_scraper.py` is the authoritative YouTube scraper; it already runs from `run_scrapers.py`. It now has expanded channel and keyword coverage for frontier labs, SLM providers, multimodal demos, eval/governance sources, and Boston/NYC events.
- Singapore governance signal:
  - IMDA / AI Verify published the Model AI Governance Framework for Generative AI in 2024.
  - IMDA launched the Model AI Governance Framework for Agentic AI on January 22, 2026 and published an updated version on May 20, 2026.
  - For Brandon, this should be summarized as an enterprise-agent governance checklist: agent autonomy boundaries, sensitive data/tool access, human accountability, testing/evals, transparency, audit logs, and runtime controls.

State-of-AI research note:

- The requested local path `~/consulting/stateofai` was not present on this host. The public `stateofai.pages.dev` URL did not resolve through search/open during this pass, so deeper stateofai-specific ingestion remains pending until the repo or correct public route is available on this machine.

### Scraper Schedule and Embedding Maintenance - 2026-05-26

- The Hetzner VM crontab runs `/home/appuser/run_scrapers.sh` at 08:00 and 20:00 UTC daily.
- That shell wrapper runs `uv run python run_scrapers.py --no-alert` from `/home/appuser/twitter_influencer`, then posts to the local tournament endpoint.
- `run_scrapers.py` default behavior was changed to Brandon-news ingestion:
  - web/news sources via `ai_news_scraper.py --web`;
  - YouTube RSS sources via `youtube_channel_scraper.py`;
  - content chunk backfill via `backfill_content_chunks.py --no-transcripts`;
  - missing web/YouTube embeddings via `regenerate_embeddings.py --articles --youtube`.
- Twitter/X scraping is now opt-in via `run_scrapers.py --twitter`; it is no longer part of the default cron path.
- VM dry-run before this change found missing embeddings for 776 web articles and 1,516 YouTube videos, so the first scheduled run after deployment may be slower while it catches up.
- `trafilatura` was missing in the VM venv during web-source scraping, so RSS/HTML metadata landed but full article bodies were not fetched for many web articles. The script headers declare `trafilatura>=1.6.0`; install/sync the script dependency environment if deeper article citations are required.
- Bad YouTube channel IDs previously added for Microsoft Research, NVIDIA, MLCommons, and MIT CSAIL imported unrelated feeds. Those polluted video rows were removed on the VM, the bad channel IDs were deactivated, and the scraper now validates expected feed titles for corrected high-risk seed IDs.
- Corey Quinn / Last Week in AWS was added as an `enterprise_ai_cloud` RSS source for snarky but useful AWS/cloud AI, cost, Bedrock, agent, and governance signal.

### Retrieval Query Embedding Runtime - 2026-05-26

- Retrieval still needs an embedding model at query time because sqlite-vec can only search stored vectors after the incoming user query is converted into the same vector space.
- The stored tweet/article/YouTube vectors are not recomputed during chat retrieval; `ChatAgent._retrieve_sources()` calls `encode_texts_hybrid([query])` once for the query, then searches `*_embeddings_dense` / `*_embeddings_sparse` tables.
- Production Flask chat already calls `warmup_embedding_model(precompute_queries=COMMON_CHAT_QUERIES)` at `linkedin_feed.py` startup, so the normal service should pay the BGE-M3 load once per process rather than once per request.
- Benchmark/smoke scripts start fresh Python processes, so their logs can show repeated BGE-M3 loads even though a long-running service would keep the singleton model in memory.
- `agents/hybrid_retriever.py` now caches single-query embeddings after first encode, not only startup-precomputed queries. Repeated identical chat queries in the same process should skip re-encoding.
- `agents/chat_agent.py` now returns a clear retrieval warning if the query embedder is unavailable, because stored vectors cannot be searched without a query vector.

### Embedding/Reranker Matrix Updates - 2026-05-28

- HF access check on the VM now passes for the new embedding/reranker candidates:
  - Granite: `ibm-granite/granite-embedding-small-english-r2`, `granite-embedding-english-r2`, `granite-embedding-97m-multilingual-r2`, `granite-embedding-311m-multilingual-r2`, and `granite-embedding-reranker-english-r2`.
  - Jina: `jinaai/jina-embeddings-v4` and `jinaai/jina-embeddings-v4-text-code-GGUF`.
  - Qwen: `Qwen/Qwen3-Embedding-0.6B`, `Qwen/Qwen3-Embedding-0.6B-GGUF`, `Qwen/Qwen3-Reranker-0.6B`.
  - Perplexity: `perplexity-ai/pplx-embed-v1-0.6b`, `pplx-embed-context-v1-0.6b`, and 4B reference variants.
  - ZeroEntropy/Mixedbread rerankers are also accessible.
- Retrieval matrix `output_data/model_bench/retrieval_pipeline_matrix_text_embedders_v1`:
  - `sentence-transformers/all-MiniLM-L6-v2`: context recall 0.487, top-10 recall 0.565, 18.9s.
  - `BAAI/bge-m3`: context recall 0.450, top-10 recall 0.653, 321.4s.
  - production `BGE-M3` hybrid path: context recall 0.414, top-10 recall 0.506, 28.2s.
  - `google/embeddinggemma-300m`: now accessible after gate approval, but previous row was gated; retest pending.
  - `nvidia/llama-nemotron-embed-1b-v2`: counted out for this 8 GB CPU VM after projecting roughly 2.5 hours for a 1,800-document embedding matrix while consuming most CPU/RAM.
- MiniLM reranker matrix `retrieval_pipeline_matrix_minilm_bge_reranker_v1`:
  - MiniLM without reranker: context recall 0.460, top-10 recall 0.612, 67.7s.
  - MiniLM + `BAAI/bge-reranker-v2-m3`: context recall 0.467, top-10 recall 0.626, 229.1s.
  - Interpretation: BGE reranking gives only a small lift at `max_sources=15`; keep it for focused service-path validation, but do not assume it is worth the CPU cost for every background retrieval job.
- Granite perf matrix `retrieval_pipeline_matrix_granite_perf_v1` with `--text-chars 1200 --model-max-length 512`:
  - `ibm-granite/granite-embedding-small-english-r2`: context recall 0.489, top-10 recall 0.637, 138.7s.
  - `ibm-granite/granite-embedding-97m-multilingual-r2`: context recall 0.443, top-10 recall 0.655, 91.9s.
  - MiniLM baseline in same run: context recall 0.446, top-10 recall 0.567, 52.9s.
  - Earlier Granite 97M default text-length run scored context recall 0.489, top-10 recall 0.673, 134.7s.
  - `granite-embedding-311m-multilingual-r2` was stopped as too slow on this VM before completion.
  - Interpretation: Granite small English is the best current context-recall candidate among new accessible embedders, but MiniLM remains the latency baseline. Granite 97M improves top-10 recall but loses context recall under the shorter text/max-length setting. Granite 311M is not a practical CPU embedding candidate for this VM without a dedicated optimized runtime.
- Benchmark harness updates:
  - `tools/retrieval_pipeline_matrix_bench.py` now supports `--text-chars`, `--model-max-length`, `--truncate-dim`, per-row resumability, and Jina query/passage prompt handling.
  - `tools/check_hf_model_access.py` records candidate access/gating status without downloading weights.

### Gemini/Hosted Chat Retrieval Audit - 2026-05-28

- Background audit found no active Gemini runtime path. Gemini references are stale comments/dependencies or query examples; main Flask chat uses `ChatAgent`.
- Main Flask `ChatAgent` hosted/local branches share retrieval, optional reranking, context compression, source numbering, and basic citation extraction.
- Divergences:
  - `ChatAgent` still defaults to hosted Bedrock/Strands when `CHAT_BACKEND` is unset; deployment should set `CHAT_BACKEND=llama_cpp` or fail closed for Hetzner.
  - Local llama.cpp branch has missing-citation retry; hosted Bedrock branch does not.
  - Citation extraction is range checking only; neither hosted nor local chat performs the full hybrid citation support validation used in `linkedin_autopilot.py`.
  - Legacy AgentCore chat remains disabled by default; if re-enabled, it is a retrieval regression because it fetches `/api/feed` instead of using BGE/chunk/rerank retrieval.
