import numpy as np 
import matplotlib.pyplot as plt 
import numpy.random as rng 
import itertools
NUM_POINTS = 30

func = lambda x: np.cos(x) + np.sin(2 * x) + np.sin(1.5 * x) + np.cos(2.5 * x)

def main():

    rng.seed(3424)
    X, y = generate_data(func, NUM_POINTS)
    
    plot_priors(X, y)
    
def generate_data(func, N):
    Xtrain = rng.uniform(0, 10, (N, 1))
    ytrain = func(Xtrain) + rng.normal(0, 1, (N, 1))
    return Xtrain, ytrain

def plot_priors(Xtrain, ytrain):

    f, ax = plt.subplots(1, 1, figsize=(20, 10))

    ax.plot(Xtrain, ytrain, 'o', color='gold', markersize=30, 
        markeredgewidth=4, markeredgecolor='black')

    xs = np.linspace(0, 10, 1000)
    
    gamma0 = [-1, 0, 1]
    gamma1 = [-2, 0, 2]
    beta0 = [0]
    beta1 = [-3, -1, 0,]

    thetas = list(itertools.product(beta0, beta1, gamma0, gamma1))


    colors = plt.cm.cool(np.linspace(0,1,len(thetas)))


    l = 0
    for l in range(len(colors)):
        theta = thetas[l]
        ys = theta[2] + theta[3] * np.exp(theta[0] + theta[1] * xs)
        ax.plot(xs, ys, linewidth=5, color=colors[l])

    ax.set_xlabel('$x$', fontsize=26)
    ax.set_ylabel('$y$', fontsize=26)
    ax.xaxis.set_tick_params(labelsize=26)
    ax.yaxis.set_tick_params(labelsize=26)
    ax.set_ylim([-3, 3])
    ax.set_xlim([0, 10])
    
    plt.savefig('figures/nonlinear_priors.png', trasparent=False, pad_inches=0., bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
