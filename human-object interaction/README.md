# human-object interaction
This repo contains the source code for our TPAMI work: **Egocentric Action Recognition by Automatic Relation Modeling**.

## Prerequisites
1. Pytorch 1.3. (If you use anaconda, you can create an environment with file [/environment/pytorch_1.3.yml](/environment/pytorch_1.3.yml).)

## Data preparation
1. Download the [EPIC-KITCHENS](https://epic-kitchens.github.io/2020-55.html) and [EGTEA Gaze+](http://cbs.ic.gatech.edu/fpv/) datasets, and extract frames and optical flows for videos.

## Train and test
Modify the arguments in the scripts [/experiments/*.sh] accordingly, and train or test with the command
```
cd experiments
### train backbone and hourglass network
bash train_bb_hg.sh
### search relational-LSTM structure
bash train_search.sh
### train all the model parameters
bash train_all.sh
### evaluation
bash test.sh
```

## Contact
If you have any problem please email me at lihaoxin05@gmail.com. I may not look at the issues.