import numpy as np
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from utils import lookupIDX

def get_seqs(p1, p2, words):
    p1 = p1.split()
    p2 = p2.split()
    X1 = []
    X2 = []
    for i in p1:
        X1.append(lookupIDX(words,i))
    for i in p2:
        X2.append(lookupIDX(words,i))
    return X1, X2

def get_correlation(model, words, filename):
    f = open(filename,'r')
    lines = f.readlines()
    preds = []
    golds = []
    seq1 = []
    seq2 = []
    ct = 0
    for i in lines:
        i = i.split("\t")
        p1 = i[0]; p2 = i[1]; score = float(i[2])
        X1, X2 = get_seqs(p1, p2, words)
        seq1.append(X1)
        seq2.append(X2)
        ct += 1
        if ct % 100 == 0:
            x1,m1 = model.prepare_data(seq1)
            x2,m2 = model.prepare_data(seq2)
            scores = model.scoring_function(x1,x2,m1,m2)
            scores = np.squeeze(scores)
            preds.extend(scores.tolist())
            seq1 = []
            seq2 = []
        golds.append(score)
    if len(seq1) > 0:
        x1,m1 = model.prepare_data(seq1)
        x2,m2 = model.prepare_data(seq2)
        scores = model.scoring_function(x1,x2,m1,m2)
        scores = np.squeeze(scores)
        preds.extend(scores.tolist())
    
    #when need score for every sentence pair, just uncomment following codes 
    '''
    s = filename + "\n"
    for pred, gold in zip(preds, golds):
        scaled_pred = (pred+1.0)*2.5
        s += str(scaled_pred) + "\t" + str(gold) + "\t" + str(abs(gold-scaled_pred)) + "\n"
    
    with open("compare.txt", "w") as f:
        f.write(s)
    '''
    return pearsonr(preds,golds)[0], spearmanr(preds,golds)[0]

def evaluate_all(model, words):
    prefix = "../data/"
    parr = []; sarr = []

    farr = ["sicktest",
            
            "STS/STS2012-test/MSRpar.stanford", 
            "STS/STS2012-test/MSRvid.stanford",
            "STS/STS2012-test/SMTeuroparl.stanford",
            "STS/STS2012-test/surprise.OnWN.stanford",
            "STS/STS2012-test/surprise.SMTnews.stanford",
            
            "STS/STS2013-test/FNWN.stanford",
            "STS/STS2013-test/headlines.stanford",
            "STS/STS2013-test/OnWN.stanford",
            
            "STS/STS2014-test/deft-forum.stanford",
            "STS/STS2014-test/deft-news.stanford",
            "STS/STS2014-test/headlines.stanford",
            "STS/STS2014-test/images.stanford",
            "STS/STS2014-test/OnWN.stanford",
            "STS/STS2014-test/tweet-news.stanford",
            
            "STS/STS2015-test/answers-forums.stanford",
            "STS/STS2015-test/answers-students.stanford",
            "STS/STS2015-test/belief.stanford",
            "STS/STS2015-test/headlines.stanford",
            "STS/STS2015-test/images.stanford",
            
            "STS/STS2016-test/answer-answer.stanford",
            "STS/STS2016-test/headlines.stanford",
            "STS/STS2016-test/plagiarism.stanford",
            "STS/STS2016-test/postediting.stanford",
            "STS/STS2016-test/question-question.stanford"]

    for i in farr:
        p,s = get_correlation(model, words, prefix + i)
        parr.append(p); sarr.append(s)

    s = ""
    for i,j,k in zip(parr,sarr,farr):
        s += str(i)+" "+str(j)+" "+k+" | " +"\n"
    
    parr = np.asarray(parr)
    
    SICK = parr[0]
    STS_2012 = np.mean(parr[1:6])
    STS_2013 = np.mean(parr[6:9])
    STS_2014 = np.mean(parr[9:15])
    STS_2015 = np.mean(parr[15:20])
    STS_2016 = np.mean(parr[-5:])
    
    s += "STS_2012 : " + str(STS_2012) + "\n"
    s += "STS_2013 : " + str(STS_2013) + "\n"
    s += "STS_2014 : " + str(STS_2014) + "\n"
    s += "STS_2015 : " + str(STS_2015) + "\n"
    s += "STS_12to15 : " + str(np.mean(parr[1:20])) + "\n"
    s += "STS_2016 : " + str(STS_2016) + "\n"
    s += "SICK : " + str(SICK) + "\n"
    
    '''
    with open("results.txt", "a") as f:
        f.write(s)
    '''
    print s