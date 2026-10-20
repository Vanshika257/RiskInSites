# Construction Risk Analyzer Copilot

## Overview

Construction Risk Analyzer Copilot is an AI-assisted project for exploring construction risk records with natural-language questions. It aims to help users identify important risks, review potential schedule and financial impacts, and understand mitigation information from the available risk dataset.

Example questions the project is designed to explore:

- Which projects have risks that may cause delays?
- Which risks have the highest financial impact?
- What high-severity safety risks are reported?
- Which critical risks in a region are related to a particular issue, such as foundation instability?

## Architecture and Workflow

The project combines structured data filtering with semantic retrieval and language-model response generation. The processing path depends on the detected query type.

1. **User query and intent routing** — classify a question as structured, semantic, hybrid, or general, and identify structured conditions and/or a semantic question.
2. **Structured filtering (Pandas)** — apply exact conditions such as severity, region, risk type, status, or financial impact to the CSV data.
3. **Semantic search (ChromaDB)** — retrieve semantically similar risk records using embeddings and the user's semantic question.
4. **Intersection / merge** — for hybrid queries, match candidate risk IDs from structured and semantic results and retrieve the corresponding risk records.
5. **AI risk analysis and response** — pass relevant records as evidence to the language model to generate a clear natural-language answer grounded in the supplied data.

## Technology Stack

- Python
- Pandas for structured data filtering and aggregation
- ChromaDB for vector storage and semantic retrieval
- Ollama for local language-model and embedding-model integration
- Llama 3.2 for response generation, depending on local configuration
- `nomic-embed-text` for embeddings, depending on local configuration
- Retrieval-Augmented Generation (RAG)
- Hybrid query processing

## Dataset

The project uses construction risk records with fields such as:

- `risk_id`
- `project_id`
- `project_name`
- `date_reported`
- `risk_type`
- `description`
- `severity_level`
- `likelihood`
- `impact_days`
- `financial_impact_usd`
- `status`
- `mitigation_plan`
- `reported_by`
- `region`
- `tags`


## Project Status and Current Focus

**Status: Under Refinement / In Progress**

The project is still being developed and tested. Current areas of refinement include:

- Improving intent classification and route-plan generation.
- Making structured filters and aggregations more reliable.
- Tuning semantic retrieval and the relevance of candidate risks.
- Improving matching between structured and semantic results for hybrid questions.
- Keeping generated answers tied to retrieved risk evidence.
- Testing queries across structured, semantic, hybrid, and general categories.

These are ongoing refinement areas for the project.
