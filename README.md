## NPST3

Implementation of Neural Policy Style Transfer with Twin-Delayed DDPG (NPST3) framework for Shared Control of Robotic Manipulators.

The algorithm can be executed with the `policy_exec_nostyle_incremental.py` script. Each numbered trained model in the repo corresponds to a different Style.

	Trained Styles
	1: Happy
	2: Calm
	3: Sad
	4: Angry

## Installation

You can install this repo running `python setup.py install` within the root of the repo. In case of compatibility issues you can use the `requirements-stable.txt` within `setup.py` for the last tested versions for each package.

## Autoencoder

The Autoencoder is trained running the script `conv_AE.py` 

The trained model can be executed with `conv_AE_predict.py`.

## TD3 Network

The TD3 network is trained with `td3_st_train_no_style_incremental.py`

## Dataset
The dataset is available [here](https://zenodo.org/record/5718543#.YZu-57so9uQ)

In order to use it, download it and save it in the root directory of the repo within a new "dataset" folder.





