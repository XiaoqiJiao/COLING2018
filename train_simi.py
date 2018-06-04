import utils
import lasagne
import random
import numpy as np
import sys
import argparse
from example import example
from conv_simi_model import conv_simi_model

def getSimiDataset(filename):
    data = open(filename,'r')
    lines = data.readlines()
    examples = []
    for i in lines:
        i=i.strip()
        if(len(i) > 0):
            i=i.split('\t')
            if len(i) == 3:
                e = (example(i[0]), example(i[1]), float(i[2]))
                examples.append(e)
            else:
                print i
    return examples

def str2bool(v):
    if v is None:
        return False
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise ValueError('A type that was supposed to be boolean is not boolean.')

def str2learner(v):
    if v is None:
        return None
    if v.lower() == "adagrad":
        return lasagne.updates.adagrad
    if v.lower() == "adam":
        return lasagne.updates.adam
    raise ValueError('A type that was supposed to be a learner is not a learner.')

random.seed(1)
np.random.seed(8023)

#for universal setting -LW 1e-4 -LC 1e-5 -L2 1e-7 -batchsize 32
parser = argparse.ArgumentParser()
parser.add_argument("-LW", help="Regularization on embedding parameters", type=float, default=1e-4)
parser.add_argument("-LC", help="Regularization on composition parameters", type=float, default=1e-5)
parser.add_argument("-L2", help="Regularization on embedding parameters", type=float, default=1e-7)
parser.add_argument("-batchsize", help="Size of batch", type=int, default=32)
parser.add_argument("-memsize", help="Size of batch", type=int, default=300)
parser.add_argument("-dim", help="Dimension of model", type=int, default=300)
parser.add_argument("-wordfile", help="Word embedding file", default="../data/paragram_sl999_small.txt")
parser.add_argument("-epochs", help="Number of epochs in training", type=int, default=10)
parser.add_argument("-eta", help="Learning rate", type=float, default=0.001)
parser.add_argument("-learner", help="Either AdaGrad or Adam", default="adam")
parser.add_argument("-loadmodel", help="Name of pickle file containing model", default="./CSE_Wiki.pickle")

params = parser.parse_args()
params.learner = str2learner(params.learner)

train_data = getSimiDataset("../data/sicktrain")
dev_data = getSimiDataset("../data/sickdev")
test_data = getSimiDataset("../data/sicktest")

if params.wordfile:
    (words, We) = utils.get_wordmap(params.wordfile)

model = conv_simi_model(We, params)

print " ".join(sys.argv)
print "Num train examples:", len(train_data)
print "Num dev examples:", len(dev_data)
print "Num test examples:", len(test_data)

data = (train_data, dev_data, test_data)
model.train(data, words, params)
