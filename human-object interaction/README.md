# human-object interaction
This repo contains the source code for our work under review.

## Prerequisites
1. Pytorch 1.3. (If you use anaconda, you can create an environment with file [/environment/pytorch_1.3.yml](/environment/pytorch_1.3.yml).)

## Data preparation
1. Download the [EPIC-KITCHENS](https://epic-kitchens.github.io/2020-55.html) and [EGTEA Gaze+](http://cbs.ic.gatech.edu/fpv/) datasets, and extract frames and optical flows for videos.

## Train and test
Modify the arguments in the script [/experiments/run.sh](/experiments/run.sh) accordingly, and train or test with the command
```
cd experiments
bash run.sh
```

## Contact
If you have any problem please email me at lihaoxin05@gmail.com or lihx39@mail2.sysu.edu.cn. I may not look at the issues.