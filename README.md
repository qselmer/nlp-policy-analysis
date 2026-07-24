# Climate-adaptive management of small pelagic fisheries

## Systematic evidence synthesis for highly variable upwelling systems

This repository implements a reproducible and auditable evidence-synthesis workflow to examine how fisheries management has **implemented, evaluated, recommended, or proposed** responses to environmental variability and climate change, with emphasis on:

- small pelagic fishes;
- Eastern Boundary Upwelling Systems;
- highly variable marine environments;
- climate-informed and ecosystem-based fisheries management;
- transferability to the north-central Peruvian anchoveta stock (*Engraulis ringens*).

The project supports the development of a scientific manuscript provisionally framed as:

> **Managing small pelagic fisheries in highly variable upwelling systems under climate change: a systematic evidence synthesis and decision-support framework**

The repository is intended to support a critical synthesis rather than a descriptive literature inventory. It distinguishes management ideas from management performance and evaluates whether evidence is sufficiently independent, transferable, operationally specific, and robust to inform the Peruvian anchoveta case.

## Core research question

**Which fisheries-management measures have been implemented, evaluated, recommended, or proposed for small pelagic fisheries exposed to high environmental variability and climate change, what evidence supports them, and under what conditions could they inform climate-adaptive management of the north-central Peruvian anchoveta stock?**

## Evidence scope

The workflow accommodates multiple evidence streams:

- peer-reviewed articles, reviews, books, chapters, and theses;
- stock assessments, scientific advice, and technical reports;
- projects, programmes, pilot applications, and implementation evaluations;
- fisheries policies, adaptation strategies, and management plans;
- legal instruments, regulations, operational protocols, and conservation measures.

Sources may describe different levels of maturity. These are coded separately:

```text
proposed → recommended → planned → piloted → implemented → evaluated
```

Implementation is not treated as evidence of effectiveness unless outcomes were explicitly evaluated.

## Methodological guardrails

The following distinctions are maintained throughout the workflow:

- proposal or recommendation ≠ implementation;
- implementation ≠ effectiveness;
- activity ≠ output ≠ outcome ≠ impact;
- projection or scenario ≠ observed trend;
- academic recommendation ≠ empirical result;
- plan or strategy ≠ binding regulation;
- number of findings ≠ number of independent studies;
- source quality ≠ transferability to anchoveta;
- expert acceptance ≠ analytical robustness;
- analytical robustness ≠ management efficacy.

Each finding retains its source, document locator, supporting text, evidence type, validation status, quality appraisal, and transferability assessment.

## Reproducible workflow

The applied workflow consists of 13 linked phases:

```text
01  Initialise the general and regulatory evidence corpora
02  Design and register multilingual searches
03  Import, harmonise, and deduplicate records
04  Screen titles and abstracts
05  Retrieve and audit full texts
06  Extract and segment documents
07  Conduct full-text eligibility screening
08  Extract structured findings
09  Appraise source quality
10  Assess transferability to Peruvian anchoveta
11  Build conditional decision-support candidates
12  Audit sensitivity, source dependence, and robustness
13  Generate scientific tables, figures, and reports
```

See [`NOTEBOOK_WORKFLOW.md`](NOTEBOOK_WORKFLOW.md) for the execution order, inputs, outputs, human-review stages, stop/go criteria, and update routes.

## Repository structure

```text
config/                   Controlled vocabularies and phase-specific rules
data/templates/           Versioned blank registers and review templates
notebooks/                Reproducible workflow notebooks, phases 01–13
src/evidence_review/      General evidence-review and synthesis modules
src/regulatory_review/    Legal, institutional, and operational analysis
tests/                    Validation and regression tests
NOTEBOOK_WORKFLOW.md       Detailed notebook execution guide
REGULATORY_REVIEW.md       Regulatory-review design and coding principles
```

The directories `data/raw/`, `data/interim/`, and `data/processed/` are intentionally excluded from version control except for documentation files. Real documents, model responses, human-review sheets, and generated results remain local.

## Main analytical products

The pipeline produces:

- a source registry and screening audit trail;
- structured climate–ecological–fishery–management findings;
- source-specific quality appraisals;
- finding-level transferability assessments for anchoveta;
- climate–response–management evidence chains;
- a management-option portfolio and operational-readiness matrix;
- research-priority and evidence-gap matrices;
- expert-reviewed conditional recommendations;
- sensitivity scenarios and leave-one-source-out analyses;
- publication-ready tables, figures, and narrative reports.

## Use in the manuscript

The repository is designed to support a review article with three connected contributions:

1. **Systematic evidence synthesis** — what management responses have been proposed, recommended, implemented, or evaluated for small pelagics under environmental variability and climate change.
2. **Evidence maturity and robustness** — how source quality, implementation stage, source independence, and transferability constrain interpretation.
3. **Anchoveta application** — a conditional framework for testing climate-adaptive management options in the north-central Peruvian anchoveta fishery.

The Peruvian case is used as an application and transferability test, not as a basis for assuming that measures effective elsewhere will be effective locally.

## Running the workflow

Create and activate a Python environment, install the project dependencies, and run the tests before executing the notebooks:

```powershell
$env:PYTHONPATH = "src"
pytest -q
```

Notebooks must be executed sequentially unless `NOTEBOOK_WORKFLOW.md` specifies an alternative restart point for new or corrected evidence.

After executing a notebook, remove generated notebook outputs before committing:

```powershell
git restore notebooks/<executed_notebook>.ipynb
git status --short
```

## Original Climate Change AI tutorial

This repository originated as a fork of the Climate Change AI tutorial **NLP Models for Climate Policy Analysis** by Daniel Spokoyny, Max Callaghan, and Tobias Schimanski.

The original educational notebooks are preserved unchanged:

- `part1_evidence_synthesis.ipynb` — supervised NLP for evidence identification and classification;
- `part2_paris_prompts.ipynb` — prompt-based classification of Paris Agreement NDC text.

They are optional learning resources and are not required inputs to the applied fisheries workflow.

Original tutorial citation:

> Spokoyny, D., Callaghan, M., & Schimanski, T. (2026). *NLP Models for Climate Policy Analysis* [Tutorial]. Climate Change AI Summer School. https://doi.org/10.5281/zenodo.21446699

## Project status

The applied workflow through phase 13 is implemented. The evidence base remains updateable: new databases, citation searches, technical reports, legal instruments, and implementation evaluations can be added through the restart routes documented in `NOTEBOOK_WORKFLOW.md`.

## License

The repository retains the MIT License of the original tutorial. New code and documentation added in this fork are distributed under the same license unless otherwise stated.
