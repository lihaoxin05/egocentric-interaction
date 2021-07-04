# human-human interaction
This repo contains the source code for our CVPR'19 work: **Deep Dual Relation Modeling for Egocentric Interaction Recognition**.

## Prerequisites
1. Tensorflow 1.4. (If you use anaconda, you can create an environment with file [/environment/tensorflow_1_4.yaml](/environment/tensorflow_1_4.yaml).)

## Data preparation
1. Download the [PEV](https://www.dropbox.com/s/ihy5qdoliktfozx/yks_cvpr2016_release.zip?dl=0), [NUSFPID](https://sites.google.com/site/sanathn/Datasets) and [JPL](http://michaelryoo.com/jpl-interaction.html) datasets, and extract frames for videos.
2. Extract reference masks for each frame using [JPPNet](https://github.com/Engineering-Course/LIP_JPPNet).
3. We format the directory structure as follows:
```
    -PEV
        -frames
            -00320_s  (video id)
                -frame000028.jpg  (frame filename)
        -parsing
            -00320_s  (video id)
                -frame000028.jpg  (mask filename)
```

## Train and test
Modify the arguments in the script accordingly, and train or test with the command
```
bash run.sh
bash run_test.sh
```

## Contact
If you have any problem please email me at lihaoxin05@gmail.com or lihx39@mail2.sysu.edu.cn. I may not look at the issues.