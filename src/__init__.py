"""Package init cho src/ — Medical Information Extraction pipeline.

Modules:
- llm_client: OpenAI-compatible wrapper cho Ollama + JSON parser.
- prompts: SYSTEM_PROMPT (NER rules + few-shot loader).
- rxnorm_rag: RxNorm retrieval (vector + BM25 + exact match hybrid).
- icd_rag: ICD-10 retrieval (vector + BM25 hybrid trên BYT data VN).
- postprocess: Validate, dedupe, fix positions, populate candidates.
- inference: Main driver — orchestrate pipeline offline.
"""
