## NPST3

Implementation of Neural Policy Style Transfer with Twin-Delayed DDPG (NPST3) framework for Shared Control of Robotic Manipulators.

The algorithm can be executed with the `policy_exec_nostyle_incremental.py` script. Each numbered trained model in the repo corresponds to a different Style.

	Trained Styles
	1: Happy
	2: Calm
	3: Sad
	4: Angry



## Autoencoder

The Autoencoder is trained running the script `conv_AE.py` 

The trained model can be executed with `conv_AE_predict.py`. To change the selected model change the path at **TODO**

## TD3 Network

The TD3 network is trained with `td3_st_train_no_style_incremental.py`

## Dataset
The dataset is available [here](https://zenodo.org/record/5718543#.YZu-57so9uQ)





