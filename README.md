## Meta-Autoencoder

This repository contains code for "[Meta-Learning to Communicate: Fast End-to-End Training for Fading Channels](https://arxiv.org/abs/1910.09945)" - 
Sangwoo Park, Osvaldo Simeone, and Joonhyuk Kang.

### Dependencies

This program is written in python 3.7 and uses PyTorch 1.2.0 and scipy.
Tensorboard for pytorch is used for visualization (e.g., https://pytorch.org/docs/stable/tensorboard.html).
- pip install tensorboard and pip install scipy might be useful.

### Basic Usage

- Train and test a model:
    
    To train the autoencoder with default settings, execute
    ```
    python main.py
    ```
    For the default settings and other argument options, see top of `main.py`
    
    Once training is done, test will be started automatically based on the trained model.
    


### Toy Example
    
-  In the 'run/toy' folder, _meta-learning_, _joint training, and fixed initialization_ schemes can be tested based on the pretrained two autoencoder architectures (vanilla autoencoder, autoencoder with RTN)
    
   In order to train from scratch, remove '--path_for_meta_trained_net ' part.
    
   In the 'saved_data/toy/nets' folder, trained models used to generate Fig. 2 can be found (proper paths are given in the shell script).
   
   In order to regenerate Fig. 3, 'toy_visualization.ipynb' may be useful.

### A More Realistic Scenario
    
-  In the 'run/realistic' folder, _meta-learning_, _joint training, and fixed initialization_ schemes can be tested based on the pretrained two autoencoder architectures (vanilla autoencoder, autoencoder with RTN)
    
   In order to train from scratch, remove '--path_for_meta_trained_net ' part.
    
   In order to train and/or test with new channels, remove '--path_for_meta_training_channels' and/or '--path_for_test_channels' part.
   
   In the 'saved_data/realistic/nets' folder, trained models used to generate Fig. 4 can be found (proper paths are given in the shell script).
   
   