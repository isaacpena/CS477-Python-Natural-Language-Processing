import sys
import nltk
import math
import time
from collections import Counter

START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
RARE_SYMBOL = '_RARE_'
RARE_WORD_MAX_FREQ = 5
LOG_PROB_OF_ZERO = -1000


# TODO: IMPLEMENT THIS FUNCTION
# Receives a list of tagged sentences and processes each sentence to generate a list of words and a list of tags.
# Each sentence is a string of space separated "WORD/TAG" tokens, with a newline character in the end.
# Remember to include start and stop symbols in yout returned lists, as defined by the constants START_SYMBOL and STOP_SYMBOL.
# brown_words (the list of words) should be a list where every element is a list of the tags of a particular sentence.
# brown_tags (the list of tags) should be a list where every element is a list of the tags of a particular sentence.
def split_wordtags(brown_train):
    brown_words = []
    brown_tags = []
    for sentence in brown_train:
        tokens = sentence.split()
        swords = [] # words in this sentence
        stags = [] # tags in this sentence
        for token in tokens:
            wordtag = token.rsplit('/', 1) 
            # rsplit starts from the back - none of the tags have / in them, so the first / seen from the end of the string is the WORD/TAG separator
            swords.append(wordtag[0])
            stags.append(wordtag[1])
    
        swords.insert(0, START_SYMBOL)
        stags.insert(0, START_SYMBOL)
        swords.insert(0, START_SYMBOL)
        stags.insert(0, START_SYMBOL)
        swords.append(STOP_SYMBOL)
        stags.append(STOP_SYMBOL)
        # Prep the words and tags for the sentence with two START_SYMBOLs and one STOP_SYMBOL.
    
        brown_words.append(swords)
        brown_tags.append(stags)
    
    return brown_words, brown_tags


# TODO: IMPLEMENT THIS FUNCTION
# This function takes tags from the training data and calculates tag trigram probabilities.
# It returns a python dictionary where the keys are tuples that represent the tag trigram, and the values are the log probability of that trigram
def calc_trigrams(brown_tags):
    q_values = {}
    
    tritags = []
    bitags = []
    for stags in brown_tags:
        # where stags is a list of tags in a given sentence:
        tritags += list(nltk.trigrams(stags))
        bitags += list(nltk.bigrams(stags))
        # Add bigrams & trigrams of the current sentence tags to these lists


    trifreq = nltk.FreqDist(tritags)
    bifreq = nltk.FreqDist(bitags)
    for token, freq in trifreq.items():
        # Trigram calculation is still based on prefix - hence calculating the bigrams as well.
        prefixcount = bifreq.get((token[0], token[1]))
        logprob = math.log(float(freq) / prefixcount, 2)
        q_values.update({token:logprob})
     
    return q_values

# This function takes output from calc_trigrams() and outputs it in the proper format
def q2_output(q_values, filename):
    outfile = open(filename, "w")
    trigrams = q_values.keys()
    trigrams.sort()  
    for trigram in trigrams:
        output = " ".join(['TRIGRAM', trigram[0], trigram[1], trigram[2], str(q_values[trigram])])
        outfile.write(output + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and returns a set of all of the words that occur more than 5 times (use RARE_WORD_MAX_FREQ)
# brown_words is a python list where every element is a python list of the words of a particular sentence.
# Note: words that appear exactly 5 times should be considered rare!
def calc_known(brown_words):
    known_words = set([])
    wordfreqs = Counter()
    for swords in brown_words:
        for word in swords:
            wordfreqs[word] += 1
            # Takes frequency of each word in each sentence
    
    for word, freq in wordfreqs.items():
        if freq > RARE_WORD_MAX_FREQ:
            # Only those whose frequency is greater than 5 are added to the known_words set
            known_words.add(word)
    
    return known_words

# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and a set of words that should not be replaced for '_RARE_'
# Returns the equivalent to brown_words but replacing the unknown words by '_RARE_' (use RARE_SYMBOL constant)
def replace_rare(brown_words, known_words):
    brown_words_rare = []
    for sentence in brown_words:
        raresent = []
        for word in sentence:
            if word in known_words:
                raresent.append(word)
            else:
                raresent.append(RARE_SYMBOL)
            # Fairly self-explanatory
        brown_words_rare.append(raresent)  
    return brown_words_rare


# This function takes the ouput from replace_rare and outputs it to a file
def q3_output(rare, filename):
    outfile = open(filename, 'w')
    for sentence in rare:
        outfile.write(' '.join(sentence[2:-1]) + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates emission probabilities and creates a set of all possible tags
# The first return value is a python dictionary where each key is a tuple in which the first element is a word
# and the second is a tag, and the value is the log probability of the emission of the word given the tag
# The second return value is a set of all possible tags for this data set
def calc_emission(brown_words_rare, brown_tags):
    e_values = {}
    taglist = set([])
    
    emisscount = Counter()
    tagcount = Counter()
    for i in range(0, len(brown_words_rare)):
         for j in range(0, len(brown_words_rare[i])):
            emisscount[(brown_words_rare[i][j], brown_tags[i][j])] += 1
            tagcount[brown_tags[i][j]] += 1  
            taglist.add(brown_tags[i][j])
            # Add one to frequency counter for each word, tag pair to get emission frequencies
            # Add one to the count of that tag as well
            # And lastly, attempt to add the tag to the set of tags - if it's already there it won't matter

    for (word, tag), freq in emisscount.items():
        denomprob = tagcount[tag]
        numerprob = float(emisscount[(word, tag)])
        # Emission probability is the number of times a tag manifests as a particular word divided by the frequency of that tag
        logprob = math.log(numerprob / denomprob , 2)
        e_values.update({(word, tag):logprob})
        
    return e_values, taglist

# This function takes the output from calc_emissions() and outputs it
def q4_output(e_values, filename):
    outfile = open(filename, "w")
    emissions = e_values.keys()
    emissions.sort()  
    for item in emissions:
        output = " ".join([item[0], item[1], str(e_values[item])])
        outfile.write(output + '\n')
    outfile.close()


def get_tags(k, tags):
    if k <= 0:
        return ['*']
    else:
        return tags

# TODO: IMPLEMENT THIS FUNCTION
# This function takes data to tag (brown_dev_words), a set of all possible tags (taglist), a set of all known words (known_words),
# trigram probabilities (q_values) and emission probabilities (e_values) and outputs a list where every element is a tagged sentence 
# (in the WORD/TAG format, separated by spaces and with a newline in the end, just like our input tagged data)
# brown_dev_words is a python list where every element is a python list of the words of a particular sentence.
# taglist is a set of all possible tags
    
    for tokens in brown_dev_words:
        n = len(tags)
        t = len(tokens)
            
        vitmatrix = [[[LOG_PROB_OF_ZERO for i in range(n)] for j in range(n)] for k in range(t)]
        bp = [[[-1 for i in range(n)] for j in range(n)] for k in range(t)]
    
        # t observations, the * at t = -1(I guess) is implicit
        # n states/tags + one * only used at the beginning  

        # Fill in first column - when t=0 (i.e. first token), the previous symbols must be *, *. 
        # As such, the "v" here is each state, while "u" and "w" = * (so the vitmatrix[k-1][u][v] term doesn't come into play)
        word = tokens[0]
        if word not in known_words:
            word = RARE_SYMBOL
        for v in range(n):
            transval = q_values.get((START_SYMBOL, START_SYMBOL, tags[v]), None)
            emissval = e_values.get((word, tags[v]), None)
            if transval != None:
                vitmatrix[0][0][v] = transval
        
        # Fill in second column, when t = 1 (second token). W is still going to be * here;
        # therefore we don't need to loop through it. Calculation of values here is
        # transition probability *->state u in range(1, n) (excepting *)-> state v in range(1, n) also excepting *
        # + emission probability P(tokens[1] | tags[v])
        # + previous value for vitmatrix[0][0][v] (the only possible one last observation)
        word = tokens[1]
        if word not in known_words:
            word = RARE_SYMBOL
        for v in range(1, n):
            for u in range(1, n):
                transval = q_values.get((START_SYMBOL, tags[u], tags[v]), None)
                emissval = e_values.get((word, tags[v]), None)
                if transval != None and emissval != None:
                    vitmatrix[1][u][v] = vitmatrix[0][0][v] + transval + emissval
                    bp[1][u][v] = u
        
       
        # Fill in remaining t-2 columns (final one is indexed t-1).
        # This is not exactly simple to loop through, but it /is/ consistent:
        # the calculation of vitmatrix[k][u][v] = vitmatrix[k-1][w][u] + q_values[tags[w], tags[u], tags[v]] + e_values[tokens[k], tags[v]] every time
        # * is left out in analysis of each token   
        # if vitmatrix[k-1][w][u] = -1000 or q_values/e_values call returns None, DO NOT CHANGE
        k = 2
        while k < t:
            # for each token from 2 to (including) t-1
            word = tokens[k]
            if word not in known_words:
                word = RARE_SYMBOL
            # replacement of rare words with RARE_SYMBOL
            
            flag = 0
            for v in range(1, n):       
                emissval = e_values.get((word, tags[v]), None)
                if emissval == None:
                   continue 
                # Unseen emissions should not do anything

                for u in range(1, n):
                    # for each state that could have been the previous state, find the maximum 
                    maxim = -1000000
                    argmaxim = -1
                    
                    for w in range(1, n):
                        # finding maximum here       
                        transval = q_values.get((tags[w], tags[u], tags[v]), None)
                        formval = vitmatrix[k-1][w][u]

                        # disallow unseen transitions and unreachable previous states
                        if transval != None:
                            totalval = formval + transval + emissval
                        else:
                            totalval = formval + emissval + LOG_PROB_OF_ZERO
                            
                        if totalval >= maxim:
                            maxim = totalval
                            argmaxim = w
                            flag = 1
                    
                    vitmatrix[k][u][v] = maxim 
                    bp[k][u][v] = argmaxim           
            k += 1
       
        # Of all the states that the final observation (usually a period, or some other punctuation) can be in;
        # find the one which has the maximum value of (transition from this state to STOP + this state's value) 
        maxim = -1000000
        argmaxim = (0, 0)
        for i in range(1, n):
            for j in range(1, n):
                transval = q_values.get((tags[j], tags[i], STOP_SYMBOL), None)
                if transval != None:
                    val = transval + vitmatrix[t-1][j][i]
                else:
                    val = LOG_PROB_OF_ZERO + vitmatrix[t-1][j][1]
                
                if val >= maxim:
                    maxim = val
                    argmaxim = (j, i)

        #(j, i) is the maximum u,v pair for observations t-2 and t-1 

        fintags = [-1 for i in range(t)]
        acttags = ['NOUN' for i in range(t)]
        
        fintags[t-1] = argmaxim[1]
        fintags[t-2] = argmaxim[0]

        acttags[t-1] = tags[argmaxim[1]]
        acttags[t-2] = tags[argmaxim[0]]
        
        # Follow backpointers to get the maximum tag sequence probability for the full sentence
        # This part of the algorithm is taken from the Michael Collins notes at Columbia 
        k = t - 3
        while k >= 0:
            point = bp[k+2][fintags[k+1]][fintags[k+2]]
            fintags[k] = point
            acttags[k] = tags[point]
            k = k - 1 
        
        # Assemble the final string & then append it to the list of tagged sentences
        finstr = ""
        for m in range(t):
            finstr = finstr + tokens[m] + "/" + acttags[m] + " "
        strippedstring = finstr.strip() + "\n"
        tagged.append(strippedstring)
          
    return tagged

# This function takes the output of viterbi() and outputs it to file
def q5_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# This function uses nltk to create the taggers described in question 6
# brown_words and brown_tags is the data to be used in training
# brown_dev_words is the data that should be tagged
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a 
# terminal newline, not a list of tokens. 
def nltk_tagger(brown_words, brown_tags, brown_dev_words):
    # Hint: use the following line to format data to what NLTK expects for training
    training = [ zip(brown_words[i],brown_tags[i]) for i in xrange(len(brown_words)) ]

    # IMPLEMENT THE REST OF THE FUNCTION HERE
    tagged = []
    
    deftag = nltk.DefaultTagger('NOUN')
    bitag = nltk.BigramTagger(training, backoff=deftag)
    tritag = nltk.TrigramTagger(training, backoff=bitag)

    for tokens in brown_dev_words:
        wordtags = list(tritag.tag(tokens))
        finstr = ""
        for tup in wordtags:
            finstr = finstr + tup[0] + "/" + tup[1] + " "
        stripstring = finstr.strip() + "\n"
        tagged.append(stripstring)            

    return tagged

# This function takes the output of nltk_tagger() and outputs it to file
def q6_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

DATA_PATH = '/home/classes/cs477/data/'
OUTPUT_PATH = 'output/'

def main():
    # start timer
    time.clock()

    # open Brown training data
    infile = open(DATA_PATH + "Brown_tagged_train.txt", "r")
    brown_train = infile.readlines()
    infile.close()

    # split words and tags, and add start and stop symbols (question 1)
    brown_words, brown_tags = split_wordtags(brown_train)

    # calculate tag trigram probabilities (question 2)
    q_values = calc_trigrams(brown_tags)

    # question 2 output
    q2_output(q_values, OUTPUT_PATH + 'B2.txt')

    # calculate list of words with count > 5 (question 3)
    known_words = calc_known(brown_words)

    # get a version of brown_words with rare words replace with '_RARE_' (question 3)
    brown_words_rare = replace_rare(brown_words, known_words)

    # question 3 output
    q3_output(brown_words_rare, OUTPUT_PATH + "B3.txt")

    # calculate emission probabilities (question 4)
    e_values, taglist = calc_emission(brown_words_rare, brown_tags)

    # question 4 output
    q4_output(e_values, OUTPUT_PATH + "B4.txt")

    # delete unneceessary data
    del brown_train
    del brown_words_rare

    # open Brown development data (question 5)
    infile = open(DATA_PATH + "Brown_dev.txt", "r")
    brown_dev = infile.readlines()
    infile.close()

    # format Brown development data here
    brown_dev_words = []
    for sentence in brown_dev:
        brown_dev_words.append(sentence.split(" ")[:-1])

    # do viterbi on brown_dev_words (question 5)
    viterbi_tagged = viterbi(brown_dev_words, taglist, known_words, q_values, e_values)

    # question 5 output
    q5_output(viterbi_tagged, OUTPUT_PATH + 'B5.txt')

    # do nltk tagging here
    nltk_tagged = nltk_tagger(brown_words, brown_tags, brown_dev_words)

    # question 6 output
    q6_output(nltk_tagged, OUTPUT_PATH + 'B6.txt')

    # print total time to run Part B
    print "Part B time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()
