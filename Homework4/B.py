import nltk
import A
import collections as coll

class BerkeleyAligner():

    def __init__(self, align_sents, num_iter):
        self.t, self.q = self.train(align_sents, num_iter)

    # TODO: Computes the alignments for align_sent, using this model's parameters. Return
    #       an AlignedSent object, with the sentence pair and the alignments computed.
    def align(self, align_sent):
        best_alignment = []
        m = len(align_sent.mots)
        l = len(align_sent.words)
       

        best_algmt = []
 
        for j, targ_word in enumerate(align_sent.words):
            best_prob = 0
            best_algn_point = None
            for i, source_word in enumerate(align_sent.mots):
                align_prob = (self.t[(targ_word, source_word)] * self.q[(i, j, m, l)])
                if align_prob >= best_prob:
                    best_prob = align_prob
                    best_algn_point = i
            best_algmt.append((j, best_algn_point))

        new_asent = nltk.AlignedSent(align_sent.words, align_sent.mots, nltk.Alignment(best_algmt))

        return new_asent
    
    # TODO: Implement the EM algorithm. num_iters is the number of iterations. Returns the 
    # translation and distortion parameters as a tuple.
    def train(self, aligned_sents, num_iters):
        ne = coll.Counter()
        nf = coll.Counter()
        t = {}
        q = {} 
        for sentence in aligned_sents:
            for e in sentence.words:
                ne[e] += len(sentence.mots)
            for f in sentence.mots:
                nf[f] += len(sentence.words)
        
        for sentence in aligned_sents:
            for e in sentence.words:
                for f in sentence.mots:
                    t[(f, e)] = 1.0/(ne[e])
                    t[(e, f)] = 1.0/(nf[f])
            
            m = len(sentence.mots)
            l = len(sentence.words)
            
            for i in range(m):
                for j in range(l):
                    q[(j, i, l, m)] = 1.0 / (l + 1)
                    q[(i, j, m, l)] = 1.0 / (m + 1)
                    

        for S in range(num_iters):
            tprobs = coll.Counter()
            tprefs = coll.Counter()
            qprobs = coll.Counter()
            qprefs = coll.Counter()

            for sentence in aligned_sents:
                m = len(sentence.mots)
                l = len(sentence.words)
                
                for i in range(m):
                    for j in range(l):
                        d1sum = 0
                        d2sum = 0
                        for k in range(l):
                            d1sum += (q[(k, i, l, m)] * t[(sentence.mots[i], sentence.words[k])])
                            d2sum += (q[(i, k, m, l)] * t[(sentence.words[k], sentence.mots[i])])
                        
                        delta1 = (q[(j, i, l, m)] * t[(sentence.mots[i], sentence.words[j])]) / d1sum
                        delta2 = (q[(i, j, m, l)] * t[(sentence.words[j], sentence.mots[i])]) / d2sum
    
                        delta = (delta1 + delta2) / 2
                        
                        tprobs[(sentence.words[j], sentence.mots[i])] += delta
                        tprobs[(sentence.mots[i], sentence.words[j])] += delta
                        tprefs[(sentence.words[j])] += delta
                        tprefs[(sentence.mots[i])] += delta

                        qprobs[(j, i, l, m)] += delta
                        qprobs[(i, j, m, l)] += delta
                        qprefs[(i, l, m)] += delta
                        qprefs[(j, m, l)] += delta

            for (e, f) in list(tprobs):
                t[(f, e)] = tprobs[(e, f)] / tprefs[(e)]

            for (i, j, m, l) in list(qprobs):
                q[(i, j, m, l)] = qprobs[(i, j, m, l)] / qprefs[(j, m, l)]
 
        return (t,q)

def main(aligned_sents):
    ba = BerkeleyAligner(aligned_sents, 10)
    A.save_model_output(aligned_sents, ba, "ba.txt")
    avg_aer = A.compute_avg_aer(aligned_sents, ba, 50)

    print ('Berkeley Aligner')
    print ('---------------------------')
    print('Average AER: {0:.3f}\n'.format(avg_aer))
