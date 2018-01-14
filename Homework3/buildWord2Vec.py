import gensim
import numpy as np
import json

def main():
    
    filepointer = open("hist_split.json")
    data = json.load(filepointer)
    prelim_sentences = data[u'train'] 

    sentences = []
    for datum in prelim_sentences:
        sublist = []
        for word in datum[0]:
            if word[0] == None:
                sublist.append("<UNK>")
            else:
                sublist.append(word[0])
        sentences.append(sublist)

    model = gensim.models.Word2Vec(size = 100, window = 5, min_count = 1)
    model.build_vocab(sentences)
    alpha, min_alpha, passes = (0.025, .001, 20)
    alpha_delta = (alpha - min_alpha) / passes

    for epoch in range(passes):
        model.alpha, model.min_alpha = alpha, alpha
        model.train(sentences)
    
        print('completed pass %i at alpha %f' % (epoch + 1, alpha))
        alpha -= alpha_delta

        np.random.shuffle(sentences)

    model.save('test')


if __name__ == "__main__":
    main()
