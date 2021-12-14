# `distilbert-base-uncased`
```bash
CUDA_VISIBLE_DEVICES=0 python run.py --multirun \
    model.model_name='google/electra-small-discriminator' \
    model.embedding_layer=all,first,last \
    model.debias_mode=token,sentence
```

## All models
- bert-base-uncased
- roberta-base
- albert-base-v2
- distilbert-base-uncased
- google/electra-small-discriminator





# GLUE
```bash
export TASK_NAME=mrpc

python run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir ./tmp/$TASK_NAME/
```


```
google/electra-small-discriminator/all-layer/sentence-debias/version_0


python run.py  \
    model.model_name='google/electra-small-discriminator' \
    model.embedding_layer=all \
    model.debias_mode=sentence



CUDA_VISIBLE_DEVICES=1 python run.py  \
    model.model_name='bert-base-uncased' \
    model.embedding_layer=last \
    model.debias_mode=token



# Fine prune


CUDA_VISIBLE_DEVICES=3 python run.py --multirun \
    experiment=debias_with_pruning_frozen \
    model.embedding_layer=last \
    model.debias_mode=sentence,token


CUDA_VISIBLE_DEVICES=2 python run.py --multirun \
    experiment=debias_with_pruning_frozen \
    model.embedding_layer=last,all \
    model.debias_mode=sentence,token


# VISUALS
```
for name, param in bert.named_parameters():
    name_parts = name.split('.')
    if name_parts[-1] == 'mask_scores' and name_parts[3] == 'attention':
        print(name)
```


Layer 0:
```
encoder.layer.0.attention.self.query.weight
encoder.layer.0.attention.self.query.bias
encoder.layer.0.attention.self.query.mask_module.context_modules.0.mask_scores
encoder.layer.0.attention.self.key.weight
encoder.layer.0.attention.self.key.bias
encoder.layer.0.attention.self.key.mask_module.context_modules.0.mask_scores
encoder.layer.0.attention.self.value.weight
encoder.layer.0.attention.self.value.bias
encoder.layer.0.attention.self.value.mask_module.context_modules.0.mask_scores
encoder.layer.0.attention.output.dense.weight
encoder.layer.0.attention.output.dense.bias
encoder.layer.0.attention.output.dense.mask_module.context_modules.0.mask_scores
encoder.layer.0.attention.output.LayerNorm.weight
encoder.layer.0.attention.output.LayerNorm.bias
encoder.layer.0.intermediate.dense.weight
encoder.layer.0.intermediate.dense.bias
encoder.layer.0.intermediate.dense.mask_module.context_modules.0.mask_scores
encoder.layer.0.output.dense.weight
encoder.layer.0.output.dense.bias
encoder.layer.0.output.LayerNorm.weight
encoder.layer.0.output.LayerNorm.bias
```

Layer 0 mask scores:
```
encoder.layer.0.attention.self.query.mask_module.context_modules.0.mask_scores
encoder.layer.0.attention.self.key.mask_module.context_modules.0.mask_scores
encoder.layer.0.attention.self.value.mask_module.context_modules.0.mask_scores
encoder.layer.0.attention.output.dense.mask_module.context_modules.0.mask_scores
encoder.layer.0.intermediate.dense.mask_module.context_modules.0.mask_scores
```

# How to get density manually
```
threshold = model.sparse_args.final_threshold

layer = bert.encoder.layer[0]
Q = layer.attention.self.query.weight
Q_mask_scores = layer.attention.self.query.mask_module.context_modules[0].mask_scores

sigmoid = torch.nn.Sigmoid()
num_zero = (sigmoid(Q_mask_scores) < threshold).sum()
num_total = Q_mask_scores.numel()

density = 1 - num_zero / num_total
f'Density {density:.2f}', num_zero
```