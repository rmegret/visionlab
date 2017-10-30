# Preparation

## Format

The tutorial consists of lecture segments, followed by hands-on
exercises.  We strongly encourage you to bring a laptop with all the
required packages installed in order to participate fully.

## Software required

- Python

  If you are new to Python, please install the
  [Anaconda distribution](https://www.continuum.io/downloads) for
  **Python version 3** (available on OSX, Linux and Windows).
  Everyone else, feel free to use your favorite distribution, but
  please ensure the requirements below are met:

  - `numpy` >= 1.12
  - `scipy` >= 0.19
  - `matplotlib` >= 2.0
  - `skimage` >= 0.13
  - `sklearn` >= 0.18
  
  Please see "Test your setup" below.
  
  Although not required for the main labs, we may use also OpenCV later on, 
  which you can install from the channel `menpo`:
  
      conda install -c menpo opencv3

- Jupyter

  The lecture material includes Jupyter notebooks.  Please follow the
  [Jupyter installation instructions](http://jupyter.readthedocs.io/en/latest/install.html),
  and ensure you have version 4 or later:

  ```bash
  $ jupyter --version
  4.1.0
  ```

## Download lecture material

For the class, we will use a modified version of the [scikit-image tutorials](https://github.com/scikit-image/skimage-tutorials)

To download the lecture materials, clone the following repo: [https://github.com/rmegret/visionlab](https://github.com/rmegret/visionlab)


    git clone --depth 1 git@github.com:rmegret/visionlab.git

or

    git clone --depth 1 https://github.com/rmegret/visionlab.git


## Test your setup

Please switch into the repository you downloaded in the previous step, and run 

    python check_setup.py

to validate your installation.

On my computer, I see (but your version numbers may differ):

```
[✓] scikit-image  0.13.0
[✓] scipy         0.19.1
[✓] matplotlib    2.1.0
[✓] notebook      5.1.0
[✓] scikit-learn  0.19.1
```

**If you do not have a working setup, please contact the instructor.**


