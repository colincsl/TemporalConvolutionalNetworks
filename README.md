# Temporal Convolutional Networks

This code implements the video- and sensor-based action segmentation models from [Temporal Convolutional Networks for Action Segmentation and Detection](https://arxiv.org/abs/1611.05267) by
[Colin Lea](http://colinlea.com/), [Michael Flynn](https://zo7.github.io/), Rene Vidal, Austin Reiter, Greg Hager 
arXiv 2016 (in-review). 

It was originally developed for use with the [50 Salads](http://cvip.computing.dundee.ac.uk/datasets/foodpreparation/50salads/), [GTEA](http://ai.stanford.edu/~alireza/GTEA/), [MERL Shopping](http://www.merl.com/demos/merl-shopping-dataset), and [JIGSAWS](http://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/) datasets. Recently we have also achieved high action segmentation performance on medical data, in robotics applications, and using accelerometer data from the [UCI Smartphone](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones) dataset.

An abbreviated version of this work was described at the [ECCV 2016  Workshop on BNMW](http://bravenewmotion.github.io/).

Requirements: TensorFlow, Keras (1.0.8+)

Requirements (optional): 
* [Numba](http://numba.pydata.org/): This makes the metrics much faster to compute but can be removed is necessary.
* [LCTM](https://github.com/colincsl/LCTM): Our older Conditional Random Field-based models.

Tested on Python 3.5. May work on Python 2.7 but is untested.


### Contents (code folder)

* `TCN_main.py.` -- Main script for evaluation. I suggest interactively working with this in an iPython shell.
* `compare_predictions.py` -- Script to output stats on each set of predictions.
* `datasets.py` -- Adapters for processing specific datasets with a common interface.
* `metrics.py` -- Functions for computing other performance metrics. These usually take the form `score(P, Y, bg_class)` where `P` are the predictions, `Y` are the ground-truth labels, and `bg_class` is the background class.
* `tf_models.py` -- Models built with TensorFlow / Keras.
* `utils.py` -- Utilities for manipulating data.

### Data

The features used for many of the datasets we use are linked below. The video features are the output of a Spatial CNN trained using image and motion information as mentioned in the paper. To get features from the MERL dataset talk to Bharat Signh at UMD.

Each set of features should be placed in the ``features`` folder (e.g., `[TCN_directory]/features/GTEA/SpatialCNN/`). 

* [50 Salads (mid-level action granularity)](https://drive.google.com/open?id=0B2EDVAtaGbOtUTJpdWxOc0pEaEk)
* [50 Salads (eval/higher-level action granularity)](https://drive.google.com/open?id=0B2EDVAtaGbOtUUFISWNxMjFBQkk)
* [GTEA](https://drive.google.com/open?id=0B2EDVAtaGbOtZWpLZmo0dURHdU0)
* [JIGSAWS](https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/): Email colincsl@gmail.com for permission. Can only be used for academic purposes. 
* [MERL Shopping](http://www.merl.com/demos/merl-shopping-dataset): Email Bharat Signh at UMD for features.

Each .mat file contains three or four types of data: 'Y' refers to the ground truth action labels for each sequence, 'X' is the per-frame probability as output from a Spatial CNN applied to each frame of video, 'A' is the 128-dim intermediate fully connected layer from the Spatial CNN applied at each frame, and if available 'S' is the sensor data (accelerometer signals in 50 Salads, robot kinematics in JIGSAWS). 

There are a set of corresponding splits for each dataset in `[TCN_directory]/splits/[dataset].` These should be easy to use with the dataset loader included here.

