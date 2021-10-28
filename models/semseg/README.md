# Semantic Segmentation

## Data
Upload the comma10k and mapillary data to a MongoDB. See data/comma10k.py and data/mappilary.py for the upload script and label_spec.py for the label spec.

## Training
```bash
# In docker:
python models/semseg/train.py

# To get tensorboard:
tensorboard --logdir /path/to/saved/models
```

## Inference
```bash
# In docker:
python models/semseg/inference.py --model_path ./trained_models/semseg_*/tf_model_*
```
