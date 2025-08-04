# Aptamer Sequence Generator

Hey! This is currently a work in progress but instructions for setting up the repository are in the instructions.md(instructions.md) file.

Some other details for how I am running this:

- OS: Ubuntu 20.04.6 LTS
- NVCC: release 12.1, V12.1.105
- [UTexas Aptamer Data](https://sites.utexas.edu/aptamerdatabase/)
- [AptaTrans Data](https://github.com/pnumlb/AptaTrans)

Approximate project layout

```
.
├── data/
│   └── Human, Flybrain, Melanoma, and Aptamer Data
├── lightning_modules/
│   └── PyTorch Lightning classes setup to use models 
├── model/
│   └── Models and Architectures
├── selene/
│   └── Cloned Selene Repository
├── utils/
│   └── Utility functions for logging, dataset classes, visualizing, etc
├── workdir/
│   └── Previous Runs and Checkpoints...
├── train_classifier.py
├── train_dna.py
├── train_promo.py
├── trial.ipynb
└── run.py
```

More details for this will be out soon. Please feel free to reach out to me for any questions!

~ Maharshii Patel

# Credit

### [Original Paper](http://arxiv.org/abs/2402.05841)

