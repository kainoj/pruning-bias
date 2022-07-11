#### _Gender Biases and Where to Find Them:_
# Exploring Gender Bias Using Movement Pruning ✂

_Accepted to NAACL2022, Workshop on Gender Bias in Natural Language Processing_

👉 [arxiv paper](https://arxiv.org/abs/2207.02463) 👈

## What does it do
We freeze weights of a pre-trained BERT and we fine-prune it on a [gender debiasing loss](https://aclanthology.org/2021.eacl-main.107.pdf). Optimized are only the pruning scores -- they act a gate to the BERT's weights. We utilzie [block movement pruning](https://arxiv.org/abs/2005.07683).



## Reproducibility
Setup
```bash
conda env create -f envs/pruning-bias.yaml
conda activate debias
pip uninstall nn_pruning
pip install git+https://github.com/[anonymized]/nn_pruning.git@automodel
```

Block pruning
```bash
python run.py --multirun \
    experiment=debias_block_pruning_frozen \
    model.embedding_layer=last,all \
    model.debias_mode=sentence,token \
    prune_block_size=32,64
```

Pruning enitre heads
```bash
python run.py --multirun \
    experiment=debias_head_pruning_frozen_values_only \
    model.embedding_layer=last,all \
    model.debias_mode=sentence,token
```

Debiasing-only:
```bash
python run.py --multirun \
    model.embedding_layer=first,last,all,intermediate \
    model.debias_mode=sentence,token
```

* The first run will download, process, and cache datasets.
* By default, debiasing will run on a single GPU. For more options, see [configs](configs/). 
    * This project uses [hydra](https://hydra.cc/docs/intro#quick-start-guide) for config managements and [pytorch lightning](https://www.pytorchlightning.ai/) for training loops. 
    * All experiments are defined in [configs/experiment/](configs/experiment/)
* We use [run_glue.py](https://github.com/huggingface/transformers/blob/master/examples/pytorch/text-classification/run_glue.py) to evaluate GLUE. To evaluate pruned models, we manually load the pruning scores state dicts.


## Credits
* Block pruning:
```bibtex
@article{Lagunas2021BlockPF,
  title={Block Pruning For Faster Transformers},
  author={Franccois Lagunas and Ella Charlaix and Victor Sanh and Alexander M. Rush},
  journal={ArXiv},
  year={2021},
  volume={abs/2109.04838}
}
```
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
