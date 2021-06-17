# Link Property Prediction - ogbl-biokg
This code includes implementation of many KG models with OGB evaluator. It is based on this [repository](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding).

The repository contains working implementation for the following models:
- TransE (same as reference repository)
- DistMult (same as reference repository)
- ComplEx (same as reference repository)
- RotatE (same as reference repository)
- ConvE (use branch `1-to-N` as it contains implementation with 1-N scoring)
- NormConvKB (ConvKB with Lp normalization instead of Batch Normalization)
- TransM (RESCAL)
- HolE
- RelConv
- QuatE
- ConvFM (Our custom model)
- ConEx
- ConvQuatE (Our custom model)
- OctonionE

### Results
The top scoring results (only for models implemented in addition to the reference) have been summarized below:

| Model Architecture        | Details | Valid MRR | Test MRR
|:-------------------------:|:-------:|:-------------------------:|:-----------------------:|
| HolE  | Effective batch size = 1024, starting lr = 0.001 | 0.81384 | 0.81374
| QuatE  | Effective batch size = 1024, starting lr = 0.0005 | 0.81169 | 0.81104
| OctonionE  | Effective batch size = 1024, starting lr = 0.0005 | 0.81218 | 0.81058
| NormConvKB  | Effective batch size = 1024, starting lr = 0.0005 | 0.76682 | 0.76697

### For training the HolE, OctonionE, and QuatE models, run:
```bash
bash examples.sh
```

## References
- [PyKEEN Library](https://github.com/pykeen/pykeen)
- [Reference implementation for ogbl-kg](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding)
