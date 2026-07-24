# NLP Models for Climate Policy Analysis

Explore how Natural Language Processing (NLP) can be used to assist in identifying and mapping climate-relevant literature using a supervised learning approach and leverage a state of the art Large Language Model (LLM) to classify climate policy documents.

Author(s):
* Daniel Spokoyny, Carnegie Mellon University, dspokoyn@cs.cmu.edu
* Max Callaghan, Mercator Research Institute on Global Commons and Climate - Berlin, callaghan@mcc-berlin.net
* Tobias Schimanski, University of Zurich, tobias.schimanski@df.uzh.ch

Originally presented at Climate Change AI Summer School 2022, revised annually for 2023, 2024, and 2026.

## Experimental adaptation: climate change and marine fisheries

This fork contains an independent extension of the tutorial for a systematic and auditable global evidence map on climate change, environmental variability, adaptation, and marine fisheries, with emphasis on small pelagics and transferability to north-central Peruvian anchoveta.

The extension does **not** replace or modify the original tutorial notebooks. It adds a separate workflow capable of reviewing multiple source types:

- scientific articles, reviews, books, and chapters;
- technical and institutional reports;
- projects, programmes, and implementation evaluations;
- policies, strategies, adaptation plans, and fishery management plans;
- legal instruments, regulations, protocols, and conservation measures.

Start with:

- `NOTEBOOK_WORKFLOW.md` for the execution order, inputs, human-review points, outputs, stop/go criteria, and source-update routes;
- `REGULATORY_REVIEW.md` for the methodological design;
- `notebooks/01_build_evidence_corpus.ipynb` for the general evidence corpus;
- `notebooks/01_build_regulatory_corpus.ipynb` for the specialised legal and operational subcorpus;
- `config/taxonomy.yml` for controlled vocabularies;
- `src/evidence_review/` for mixed-evidence schemas and prompts;
- `src/regulatory_review/` for legal-mechanism extraction.

The workflow separates source authority, methodological quality, operational specificity, and transferability. These dimensions must not be collapsed into a single claim of evidence strength.

## Access this tutorial

We recommend executing this notebook in a Colab environment to gain access to GPUs and to manage all necessary dependencies.

Part 1: <a target="_blank" href="https://colab.research.google.com/github/climatechange-ai-tutorials/nlp-policy-analysis/blob/main/part1_evidence_synthesis.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Part 2: <a target="_blank" href="https://colab.research.google.com/github/climatechange-ai-tutorials/nlp-policy-analysis/blob/main/part2_paris_prompts.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Estimated time to execute end-to-end: 2 hours

## Contribute to this tutorial

Please refer to these [GitHub instructions](https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project#about-forking) to open a pull request via the "fork and pull request" workflow.

Pull requests will be reviewed by members of the Climate Change AI Tutorials team for relevance, accuracy, and conciseness.

## Climate Change AI Tutorials

Check out the [tutorials page](https://www.climatechange.ai/tutorials?) on our website for a full list of tutorials demonstrating how AI can be used to tackle problems related to climate change.

## License

Usage of this tutorial is subject to the MIT License.

## Cite

### Plain Text

Spokoyny, D., Callaghan, M, & Schimanski, T. (2026). NLP Models for Climate Policy Analysis [Tutorial]. In Climate Change AI Summer School. Climate Change AI. https://doi.org/10.5281/zenodo.21446699

### BibTeX

```
@misc{spokoyny2026nlp,
  title={NLP Models for Climate Policy Analysis},
  author={Spokoyny, Daniel and Callaghan, Max and Schimanski, Tobias},
  year={2026},
  organization={Climate Change AI},
  type={Tutorial},
  doi={https://doi.org/10.5281/zenodo.21446699},
  booktitle={Climate Change AI Summer School},
  howpublished={\url{https://github.com/climatechange-ai-tutorials/nlp-policy-analysis}}
}
```
