import test from "node:test";
import assert from "node:assert/strict";
import { unsupportedClinicalValidationClaims } from "../scripts/contentGuards.mjs";

test("clinical validation guard accepts explicit limitations", () => {
  for (const statement of [
    "This is not clinically validated.",
    "The system is not a clinically validated diagnostic tool.",
    "It has not been clinically validated.",
    "A model has never been clinically validated.",
    "It is not yet clinically validated.",
    "Without being clinically validated, this cannot be used in care.",
  ]) {
    assert.deepEqual(unsupportedClinicalValidationClaims(statement), [], statement);
  }
});

test("clinical validation guard rejects unsupported affirmative claims", () => {
  for (const statement of [
    "This is clinically validated.",
    "It is a clinically validated decision support tool.",
    "This has been clinically validated.",
    "It is not diagnostic. This system is clinically validated.",
  ]) {
    assert.equal(unsupportedClinicalValidationClaims(statement).length, 1, statement);
  }
});
