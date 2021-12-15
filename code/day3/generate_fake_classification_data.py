import numpy as np 
import matplotlib.pyplot as plt 
import numpy.random as rng 
import sklearn.datasets

NUM_POINTS = 30

def main():


    X, y = sklearn.datasets.make_moons(n_samples=1000, noise=.4)

    n_classes = np.max(y) + 1

    f, ax = plt.subplots(1, 1, figsize=(10, 10))
    for c in range(n_classes):
        ix = y == c 
        xs = X[ix,:]
        ax.scatter(xs[:,0], xs[:,1])
    
    plt.show()

    np.savez('classification_data', X=X, y=y)


if __name__ == "__main__":
    main()
