import sklearn.manifold
from dependencyRNN import DependencyRNN
import matplotlib.pyplot as plt

def main():
    drnn = DependencyRNN.load("random_init.npz")
    x = drnn.answers
    v = len(x.keys())
    d = len(x[x.keys()[0]])

    x_mat = [x[word] for word in x.keys()]

    tsne = sklearn.manifold.TSNE(n_components=2, perplexity=30.0)
    x_reduced = tsne.fit_transform(x_mat)
    
    x_vals = x_reduced[:, 0]
    y_vals = x_reduced[:, 1]
    label_vals = x.keys()
    
    fig, ax = plt.subplots()
     
    for i, j, l in zip(x_vals, y_vals, label_vals):
        plt.text(i, j, l, fontsize = 4)
   
    plt.ylim(-27, 27) 
    plt.xlim(-25, 30)
    
    plt.savefig('tsne_visualization.png', format='png', dpi=1000)

    

if __name__ == "__main__":
    main()
