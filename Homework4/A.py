import nltk

# TODO: Initialize IBM Model 1 and return the model.
def create_ibm1(aligned_sents):
    ibm = nltk.align.IBMModel1(aligned_sents, 10)
    return ibm

# TODO: Initialize IBM Model 2 and return the model.
def create_ibm2(aligned_sents):
    ibm = nltk.align.IBMModel2(aligned_sents, 10)
    return ibm

# TODO: Compute the average AER for the first n sentences
#       in aligned_sents using model. Return the average AER.
def compute_avg_aer(aligned_sents, model, n):
    aer_sum = 0
    for i in range(n):
        als = model.align(aligned_sents[i])
        aer_sum += als.alignment_error_rate(aligned_sents[i])
    return aer_sum / n

# TODO: Computes the alignments for the first 20 sentences in
#       aligned_sents and saves the sentences and their alignments
#       to file_name. Use the format specified in the assignment.
def save_model_output(aligned_sents, model, file_name):
    fp = open(file_name, 'w')
    for i in range(20):
        sent = model.align(aligned_sents[i])
        fp.write(str(sent.words))
        fp.write("\n")
        fp.write(str(sent.mots))
        fp.write("\n")
        fp.write(str(sent.alignment))
        fp.write("\n\n")
    fp.close()
    return


def main(aligned_sents):
    ibm1 = create_ibm1(aligned_sents)
    save_model_output(aligned_sents, ibm1, "ibm1.txt")
    avg_aer = compute_avg_aer(aligned_sents, ibm1, 50)

    print ('IBM Model 1')
    print ('---------------------------')
    print('Average AER: {0:.3f}\n'.format(avg_aer))

    ibm2 = create_ibm2(aligned_sents)
    save_model_output(aligned_sents, ibm2, "ibm2.txt")
    avg_aer = compute_avg_aer(aligned_sents, ibm2, 50)
    
    print ('IBM Model 2')
    print ('---------------------------')
    print('Average AER: {0:.3f}\n'.format(avg_aer))
