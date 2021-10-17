# Removing gender bias from pre-trained language models [WIP]

The aim of this project is to remove gender bias from pre-trained language
models, as described by [Kaneko et al](https://aclanthology.org/2021.eacl-main.107.pdf).
The idea, in brief, is to enforce _attribute_ and _target_ embeddings to be orthogonal via fine-tuning.
Think of _attributes_ as gender-related words (e.g man, woman), and _targets_ as stereotypes (e.g doctor, nurse).

**NB: This is a work in progress. The code will change and it will go beyond the scope of the original idea.**


## How to debias
```bash
pip install -r requirements.txt
python run.py +debiaser.model_name='distilbert-base-uncased'
```
* The first run will download, process and cache datasets.
* By default, debiasing will run on a single gpu. For more options, see [configs](configs/). 
    * This project uses [hydra](https://hydra.cc/docs/intro#quick-start-guide) for config managements and [pytorch lightning](https://www.pytorchlightning.ai/) for training loops. 
* [WIP] You can choose `model_name` ∈ {`bert-base-uncased`, `distilbert-base-uncased`, ~~`roberta-base`~~, ~~`albert-base-v2`~~, ~~`google/electra-small-discriminator`~~}.
    * These are pre-trained [🤗 transformers](https://huggingface.co/).


## Credits
* The original debiaing idea:
```bibtex
@inproceedings{kaneko-bollegala-2021-context,
    title={Debiasing Pre-trained Contextualised Embeddings},
    author={Masahiro Kaneko and Danushka Bollegala},
    booktitle = {Proc. of the 16th European Chapter of the Association for Computational Linguistics (EACL)},
    year={2021}
}
```
* Hydra+lightning template by [ashleve](https://github.com/ashleve/lightning-hydra-template).
