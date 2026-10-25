// Validation guard for claims of clinical validation. Negated statements
// are research limitations, not assertions of clinical efficacy.
const validationPhrase = /\bclinically validated\b/gi;
const negatedContext = /\b(?:not(?:\s+(?:yet|a|an|been|fully|independently)){0,3}|never(?:\s+been)?|without\s+being)\s*$/i;

export function unsupportedClinicalValidationClaims(content) {
  const matches = [];
  for (const match of content.matchAll(validationPhrase)) {
    const preceding = content.slice(Math.max(0, match.index - 90), match.index);
    // Match within the same sentence/clause; a negation in a previous
    // sentence must not excuse a new affirmative claim.
    const clause = preceding.split(/[.!?;\n]/).at(-1);
    if (!negatedContext.test(clause)) {
      matches.push(match[0]);
    }
  }
  return matches;
}
