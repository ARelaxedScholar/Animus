# Fact-Checker Pipeline Implementation Skill

This skill documents the pattern for implementing an API-based fact-checking pipeline in the Animus codebase.

## Objective
To add a high-quality, GPU-less, API-based fact-checking stage to the production pipeline that performs surgical removal of refuted claims.

## Implementation Pattern

### 1. Configuration (settings.rs)
- Define a `FactCheckerConfig` struct.
- Integrate into `Settings` and provide defaults for API-based services.
- Update `from_env()` to pull environment variables (e.g., GROQ_API_KEY).

### 2. Node Implementation (src/nodes/fact_checker.rs)
- Implement `FactCheckerLogic` as an `AsyncNodeLogic`.
- **Claim Extraction**: Use an LLM (e.g., Groq/Llama-3) to identify atomic claims as JSON.
- **Evidence Gathering**: Programmatically search via DuckDuckGo (or similar).
- **Verification (NLI)**: Use an efficient LLM (e.g., Gemini Flash) to perform entailment checks.
- **Surgical Filtering**: Use local logic to drop refuted sentences.

### 3. Pipeline Integration (flows/video_production.rs)
- Inject the `fact_checker_node` between `ScriptWriter` and `TTS`.

### 4. Observability
- Log verification stats:
  - Total claims extracted.
  - Verification statuses (Supported/Refuted/NotVerifiable).
  - Number of surgical cuts performed.
  - Total execution time.

## Safety Mechanisms
- **Fail Open**: If APIs are unavailable, log the error and continue, marking the video as unverified in the metadata.
- **Tiered Verification**:
  - Direct quotes, historical figures: Strictly verified.
  - General claims: Randomly sampled (configurable).
