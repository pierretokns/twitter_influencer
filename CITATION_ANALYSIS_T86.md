# Tournament 86 Citation Analysis Report

## Executive Summary

Tournament 86 generated a LinkedIn post about AI trends in 2026. Analysis reveals **6 major citation problems** where sources are matched to claims they don't directly support. The root cause is BGE-M3 semantic similarity matching with a threshold too low (0.25) and no entity-level validation.

## Post Content & Citations Analysis

### Claim 1: "88% of healthcare professionals still don't trust AI for medical insights"
**Citation**: [1] Fireship - "The unhinged world of tech in 2026..."
**Status**: ❌ WRONG
- Post opens with a healthcare-specific statistic
- Citation is to generic tech video overview
- No healthcare data in Fireship content
- **Why matched**: Generic overlap on "tech" and "2026"

### Claim 2: "Their new ChatGPT feature explains inflammation better than my med school textbook"
**Citation**: [1] Fireship + [5] OpenAI - "Understanding Inflammation | with ChatGPT"
**Status**: ⚠️ PARTIAL
- [5] is correct - OpenAI ChatGPT video about inflammation
- [1] (Fireship) should not be here - it's generic tech content
- **Why double-cited**: Algorithm found semantic similarity for "explains better than textbook"

### Claim 3: "Personal intelligence update from Gemini...Drawing insights from YOUR data"
**Citation**: [2] Matt Wolfe "Best VR Glasses" + [3] Gemini
**Status**: ❌ WRONG + ⚠️ SPLIT
- [3] (Gemini tweet) is correct
- [2] (VR Glasses review) is completely unrelated
- **Why matched**: "Personal" and "insights" keywords matched both sources incorrectly

### Claim 4: "Customized responses based on YOUR context"
**Citation**: [3] Gemini Personal Intelligence
**Status**: ✓ CORRECT

### Claim 5: "Qwen just demonstrated AI agents completing real tasks autonomously"
**Citation**: [4] Alibaba Cloud - "The Future of AI Is Action: See Qwen App Complete a Real Task in Seconds"
**Status**: ✓ CORRECT

### Claim 6: "Document AI courses are teaching extraction workflows that replace entire teams"
**Citation**: [2] DeepLearning.AI + [3] Gemini
**Status**: ⚠️ PARTIAL
- DeepLearning.AI Document AI course is correct
- [2] should be UNIQUE citation, not shared with VR Glasses
- [3] (Gemini) should NOT be here - document extraction has nothing to do with Gemini
- **Why matched**: "extraction" and "workflows" found in multiple sources via TF-IDF fallback

### Claim 7: "The winners in 2026 won't be the companies with the biggest models"
**Citation**: [1] Fireship
**Status**: ❌ WRONG
- This is editorial commentary, not supported by any source
- Fireship is generic tech overview, not relevant

### Claim 8-10: Disconnected Citations
- **[6]** BNY Sales uses OpenAI - unrelated to main post theme
- **[8]** TiVo history - completely different hardware topic
- **[10]** Tamagotchi plants + Federal Reserve video - both clearly not about AI intelligence trends

## Citation Accuracy Score

- **Correct** (✓): 2 citations
- **Partial/Mixed** (⚠️): 4 citations
- **Wrong** (❌): 5+ citations
- **Accuracy**: ~20% of citation placements are reliable

## Root Cause Analysis

### Algorithm Issue: `find_sentence_source_mapping()` at hybrid_retriever.py:307-401

**Three Cascading Problems**:

1. **No Entity Overlap Validation**
   - Line 311: `require_entity_overlap: bool = False  # Disabled`
   - This was explicitly disabled in favor of "semantic similarity instead"
   - Result: "Personal Intelligence" matches ANY mention of "personal" or "intelligence"
   - Example: Matches Gemini tweet to VR Glasses video (both have "personal" features)

2. **Threshold Too Low (0.25 for BGE-M3, 0.15 for TF-IDF)**
   - BGE-M3 semantic similarity operates on paraphrase matching
   - 0.25 threshold catches tangential semantic relationships
   - Example: "explains inflammation" → matches "best VR glasses" (both explain/teach something)
   - Should be 0.4+ to require stronger semantic alignment

3. **No One-to-Many Deduplication**
   - Same source can be cited multiple times (e.g., [2] appears 4 times)
   - Algorithm assigns new citation numbers for each sentence match
   - Should consolidate multiple matches to same source into single citation

### Specific Code Problem: variant_generator.py:419

```python
# This line:
mapping = find_sentence_source_mapping(all_sentences, source_texts, threshold=0.25)

# Does semantic matching WITHOUT:
# - Validating entities are present in source
# - Checking if the source actually has information about the specific claim
# - Deduplicating multiple matches to same source
```

## System Issues (Not Just Threshold)

**Issue 1: Sentence-level matching without context**
- "AI stopped trying to be everything" matches ANY sentence about "AI" and "everything"
- No phrase/claim-level validation

**Issue 2: Generic source titles**
- "The unhinged world of tech in 2026" is too broad to support specific claims
- BGE-M3 finds it semantically related to ANY tech topic

**Issue 3: One-to-many with no consolidation**
- [5] has TWO sources: OpenAI (correct) + Alibaba CosyVoice (wrong)
- Both tagged with citation [5] but completely different topics

## Recommendations (Priority Order)

### Quick Fixes (Change threshold)
1. Increase BGE-M3 threshold from 0.25 → 0.4
2. Increase TF-IDF threshold from 0.15 → 0.25
3. Test on recent tournaments

### Medium-term (Enable validation)
1. Re-enable entity overlap checking in hybrid_retriever.py:311
2. Extract key entities from both claim and source
3. Require entity overlap for citation to be valid
4. Example: Only cite OpenAI source if claim mentions "OpenAI" or "ChatGPT"

### Long-term (Structural changes)
1. Add manual review layer before posting
2. Spot-check random 5% of citations
3. Compare against source content directly
4. Implement uncertain/unverified citation flagging
5. Add citation confidence scores to posts

## Test Cases to Validate Fixes

**Test 1**: Claim "Document AI courses teach extraction"
- Should cite DeepLearning.AI video ONLY
- Should NOT cite Gemini (no document extraction capability)
- Score: 1 citation vs 2 currently

**Test 2**: Claim "ChatGPT explains inflammation"
- Should cite OpenAI video ONLY
- Should NOT cite Fireship (generic tech)
- Score: 1 correct vs 2 wrong currently

**Test 3**: Editorial "winners won't have biggest models"
- Should have 0 citations (opinion, not factual claim)
- Currently has [1] Fireship (wrong)
- Score: 0 citations vs 1 wrong

## Files to Monitor

- `agents/hybrid_retriever.py` - Citation extraction logic (entity validation disabled)
- `agents/variant_generator.py` - Citation insertion (line 419 threshold)
- `db_migrations.py` - Track tournament_sources schema changes
- `backfill_citations.py` - Tool to fix past tournaments after algorithm changes
