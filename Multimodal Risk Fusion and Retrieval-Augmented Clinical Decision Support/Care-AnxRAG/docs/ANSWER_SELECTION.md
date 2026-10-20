# Extractive answer selection

CARE keeps Ollama embeddings and extractive answers. Retrieved passages must
pass the existing retrieval, confidence, safety, and conflict gates before the
answer stage. The answer stage selects up to three distinct, complete source
sentences across the supplied context, preserving the original citation IDs.
It never paraphrases a result or infers a treatment effect from a study objective.

## Selection and insufficient evidence

Treatment/effectiveness questions require an eligible finding, recommendation,
or evidence statement. Obvious objectives, methods, planned effects, and delivery
background cannot answer an effectiveness question. Explicit results and
negative/uncertain findings take priority over generic background within a passage.
Descriptive questions about which interventions are discussed can select a
discussion statement. General information queries can still select definitions.
Exact duplicates are omitted.

Explicit treatment, subtype, outcome, and comorbidity requirements must be
supported by the selected sentence itself. Recognized population scope from the
passage, title, or ingested `clinical_facets` must remain in the selected sentence.
An adolescent finding can be quoted with its adolescent qualifier for a general
question; it cannot answer an explicitly adult question. If the scope occurs only
elsewhere in the document, the sentence is omitted rather than generalized.
Negative results, uncertain findings, and within-sentence limitations are retained
verbatim. Source sentences are not truncated, including at common abbreviations
such as `vs.` and `e.g.`. Selection and grounding share the same sentence-boundary
handling. Existing exact-source grounding still checks every returned medical
statement against its citation.

If no sentence is eligible, `/v1/ask` returns HTTP 200 with:

```json
{
  "abstained": true,
  "abstention_reason": "insufficient_answer_evidence",
  "citations": []
}
```

The answer explains that eligible statements/findings preserving clinical context
and population scope were unavailable. This is distinct from a formatting error,
a citation-validation failure, or an unresolved retrieval conflict. The API schema
is unchanged. Confidence remains the retrieval score; it is not answer accuracy.

## Validation and limits

Run `python -m pytest tests/test_answer_selection.py -q` and `./scripts/validate.sh`.
The selection fixtures are synthetic software tests, not clinical benchmark labels.
They cover objective-versus-finding selection, objectives/methods-only abstention,
late-context citation mapping, duplicate sentences, negative/uncertain results,
population scope, requested outcomes, API abstention, and existing conflict gates.

The selector uses conservative English phrase rules and the existing deterministic
clinical concept vocabulary. It can omit useful findings expressed in unfamiliar
language or separated from their context. A recognized finding phrase does not
prove clinical correctness. Separate limitations sentences are not synthesized
into an answer. These behaviors need human review on development data before
freezing the corpus, selector configuration, and held-out benchmark for final
evaluation. No accuracy, clinical safety, or latency improvement is established
by the software tests alone.
