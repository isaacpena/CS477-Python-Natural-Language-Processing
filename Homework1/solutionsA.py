import math
import nltk
import time
import re
from collections import Counter

# Constants to be used by you when you fill the functions
START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
MINUS_INFINITY_SENTENCE_LOG_PROB = -1000

# TODO: IMPLEMENT THIS FUNCTION
# Calculates unigram, bigram, and trigram probabilities given a training corpus
# training_corpus: is a list of the sentences. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function outputs three python dictionaries, where the keys are tuples expressing the ngram and the value is the log probability of that ngram
def calc_probabilities(training_corpus):
    unigrams = Counter()
    bigrams = Counter()
    trigrams = Counter()
    unigrams2 = Counter()
    i = 0   
 
    for sentence in training_corpus:
        tokens = sentence.split()   
        # Get tokens for each sentence       
 
        tokens.append(STOP_SYMBOL)
        for token in tokens:
            unigrams[token] += 1
        
        tokens.insert(0, START_SYMBOL)
        for token in tokens:
            unigrams2[token] += 1
        for token in list(nltk.bigrams(tokens)):
            bigrams[token] += 1

        tokens.insert(0, START_SYMBOL)
        for token in list(nltk.trigrams(tokens)):
            trigrams[token] += 1

        # Add tokens (or bigrams/trigrams of tokens) to counters as necessary, with adequate # of START_SYMBOLs
    
    unigram_p = {}
    bigram_p = {}
    trigram_p = {}
    
 
    n = float(sum(unigrams.values()))
    for token, freq in unigrams.items():
        logprob = math.log(freq / n, 2.0)
        unigram_p.update({(token,): logprob})
        # Probability of a unigram is (# instances of a token / # total instances of all tokens)   

    for token, freq in bigrams.items():
        logprob = math.log(float(freq) / unigrams2.get(token[0]), 2.0)
        bigram_p.update({(token): logprob})
        # Probability of a bigram is (# instances of a bigram / # instances of bigrams with same first token)
        # Unigrams2 needed here cause otherwise we can't get the number of instances of START_SYMBOLs 
    
    for token, freq in trigrams.items():
        logprob = 0.0
        if token[0] == '*' and token[1] == '*':
            logprob = math.log(float(freq) / unigrams2.get(token[0]), 2.0)
        else:
            logprob = math.log(float(freq) / bigrams.get((token[0], token[1])), 2.0)
        trigram_p.update({(token): logprob})
        # Probability of a trigram is (# instances of a trigram / # instances of trigrams with same first two tokens)
        # We have to make recourse again to Unigrams2 here when we have ('*', '*', ?) as a trigram because there's no ('*', '*') bigram
        # This is fine, though: the frequency of the ('*') unigram - one per sentence - would be the same as the frequency of ('*', '*') bigrams if the bigrams were evaluated with two start symbols.
        # I could add them and use a second "bigrams2" distribution, but this also works (and is probably computationally cheaper).
 
    return unigram_p, bigram_p, trigram_p

# Prints the output for q1
# Each input is a python dictionary where keys are a tuple expressing the ngram, and the value is the log probability of that ngram
def q1_output(unigrams, bigrams, trigrams, filename):
    # output probabilities
    outfile = open(filename, 'w')

    unigrams_keys = unigrams.keys()
    unigrams_keys.sort()
    for unigram in unigrams_keys:
        outfile.write('UNIGRAM ' + unigram[0] + ' ' + str(unigrams[unigram]) + '\n')

    bigrams_keys = bigrams.keys()
    bigrams_keys.sort()
    for bigram in bigrams_keys:
        outfile.write('BIGRAM ' + bigram[0] + ' ' + bigram[1]  + ' ' + str(bigrams[bigram]) + '\n')

    trigrams_keys = trigrams.keys()
    trigrams_keys.sort()    
    for trigram in trigrams_keys:
        outfile.write('TRIGRAM ' + trigram[0] + ' ' + trigram[1] + ' ' + trigram[2] + ' ' + str(trigrams[trigram]) + '\n')

    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence
# ngram_p: python dictionary of probabilities of uni-, bi- and trigrams.
# n: size of the ngram you want to use to compute probabilities
# corpus: list of sentences to score. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function must return a python list of scores, where the first element is the score of the first sentence, etc. 
def score(ngram_p, n, corpus):
    scores = []

    for sentence in corpus:
        tokens = sentence.split()
        tokens.append(STOP_SYMBOL)
        if n > 1:
            tokens.insert(0, START_SYMBOL)
        if n > 2:
            tokens.insert(0, START_SYMBOL)
        # Correct number of START_SYMBOLS added based on whether uni/bi/trigrams are being evaluated
        score = 0.0
        missing = 0
        grams = []
        if n == 1: 
            grams = list((token,) for token in tokens)
        if n == 2:
            grams = list(nltk.bigrams(tokens))
        if n == 3:
            grams = list(nltk.trigrams(tokens))
        # Compilation of n-grams for a given sentence
        
        for token in grams:
            if ngram_p.get(token, None) == None:
                missing = 1
                scores.append(MINUS_INFINITY_SENTENCE_LOG_PROB)
                break
                # If an ngram is missing - if we haven't seen it before - we want to set the whole sentence to -1000 log probability.
                # I set the flag "missing" to 1, and break out from that token.
            else:
                score += ngram_p[token] 
                # If an n-gram isn't missing we just add its score for the sentence.
        if missing == 0:
            scores.append(score)
            # If we never encounter a missing n-gram, we append the score to the list of scores.

    return scores

# Outputs a score to a file
# scores: list of scores
# filename: is the output file name
def score_output(scores, filename):
    outfile = open(filename, 'w')
    for score in scores:
        outfile.write(str(score) + '\n')
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence with a linearly interpolated model
# Each ngram argument is a python dictionary where the keys are tuples that express an ngram and the value is the log probability of that ngram
# Like score(), this function returns a python list of scores
def linearscore(unigrams, bigrams, trigrams, corpus):
    scores = []
    s = 0
    for sentence in corpus:
        uniweight = float(1) / 3
        biweight = float(1) / 3
        triweight = float(1) / 3
            
        score = 0.0
        missing = 0        
        
        tokens = sentence.split()
        tokens.append(STOP_SYMBOL)

        tokens.insert(0, START_SYMBOL)
        tokens.insert(0, START_SYMBOL)

        # In calculating linear scores, we want all the START_SYMBOLs in place so the initial trigram for each sentence is available.

        for i in range(2, len(tokens)):
            # We start at 2 - thus, the initial trigram is ('*', '*', ?), the initial bigram is ('*', ?), and the initial unigram is (?). 
            # This is because of our two initial start symbols. 
            if unigrams.get((tokens[i],), None) == None or bigrams.get((tokens[i-1], tokens[i]), None) == None or trigrams.get((tokens[i-2], tokens[i-1], tokens[i]), None) == None:
                score = MINUS_INFINITY_SENTENCE_LOG_PROB
                break
                # Same principle as in the score() function - missing n-grams nil the whole sentence. 
            else:
                uniscore = math.pow(2, unigrams[(tokens[i],)]) * uniweight
                biscore = math.pow(2, bigrams[(tokens[i-1], tokens[i])]) * biweight
                triscore = math.pow(2, trigrams[(tokens[i-2], tokens[i-1], tokens[i])]) * triweight
                # scores need to be weighted & then added together in non-log form. Log addition is analog to normal multiplication; however, normal addition has no log counterpart.  
                score += math.log(uniscore + biscore + triscore, 2)        
        scores.append(score)
    return scores

DATA_PATH = '/home/classes/cs477/data/' # absolute path to use the shared data
OUTPUT_PATH = 'output/'

# DO NOT MODIFY THE MAIN FUNCTION
def main():
    # start timer
    time.clock()

    # get data
    infile = open(DATA_PATH + 'Brown_train.txt', 'r')
    corpus = infile.readlines()
    infile.close()

    # calculate ngram probabilities (question 1)
    unigrams, bigrams, trigrams = calc_probabilities(corpus)

    # question 1 output
    q1_output(unigrams, bigrams, trigrams, OUTPUT_PATH + 'A1.txt')

    # score sentences (question 2)
    uniscores = score(unigrams, 1, corpus)
    biscores = score(bigrams, 2, corpus)
    triscores = score(trigrams, 3, corpus)

    # question 2 output
    score_output(uniscores, OUTPUT_PATH + 'A2.uni.txt')
    score_output(biscores, OUTPUT_PATH + 'A2.bi.txt')
    score_output(triscores, OUTPUT_PATH + 'A2.tri.txt')

    # linear interpolation (question 3)
    linearscores = linearscore(unigrams, bigrams, trigrams, corpus)

    # question 3 output
    score_output(linearscores, OUTPUT_PATH + 'A3.txt')

    # open Sample1 and Sample2 (question 5)
    infile = open(DATA_PATH + 'Sample1.txt', 'r')
    sample1 = infile.readlines()
    infile.close()
    infile = open(DATA_PATH + 'Sample2.txt', 'r')
    sample2 = infile.readlines()
    infile.close() 

    # score the samples
    sample1scores = linearscore(unigrams, bigrams, trigrams, sample1)
    sample2scores = linearscore(unigrams, bigrams, trigrams, sample2)

    # question 5 output
    score_output(sample1scores, OUTPUT_PATH + 'Sample1_scored.txt')
    score_output(sample2scores, OUTPUT_PATH + 'Sample2_scored.txt')

    # print total time to run Part A
    print "Part A time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()
