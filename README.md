# COLING2018
Python code for training and testing the model in the COLING 2018 paper: "Convolutional Neural Network for Universal Sentence Embeddings". This simple CNN model achieves strong performance on semantic similarity tasks in transfer learning setting, and it can also act as effective initialization for downstream tasks.

This code is modified from https://github.com/jwieting/acl2017, and we add our CNN model to compare with their models.

The code is written in python and requires numpy, scipy, theano, and the lasagne libraries.
And all the datasets have been tokenized with Stanford CoreNLP.

For transfer learning, you should focus on the file train.py in the main directory.
The best model we have trained is CSE_Wiki.pickle with the hyperparameters "-margin 0.6 -batchsize 256 -LW 1e-06 -LC 0.0 -dropout 0.1"

For verifying our experimental results, you should run "python train.py -model conv -loadmodel ./CSE_Wiki.pickle -epochs 0", and you will get the results of Table 2 in our paper.

For supervised learning, you should run "python train_simi.py -loadmodel ./CSE_Wiki.pickle -LW 1e-4 -LC 1e-5 -L2 1e-7 -batchsize 32", then the training process will appear on your screen, and you find the best pearson's r in validation set, finally you can get the results we show in Table 3 of the paper.

The trained model CSE_Wiki.pickle is a big file, you can get them here: https://www.dropbox.com/sh/ro6zgyac1qpasf9/AACYEs5B2T2ppuLADCXzv7MVa?dl=0
