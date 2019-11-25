import numpy as np 
import matplotlib.pyplot as plt 
import numpy.random as rng 

NUM_POINTS = 30

func = lambda x: np.cos(x) + np.sin(2 * x) + np.sin(1.5 * x) + np.cos(2.5 * x)

def main():

    rng.seed(3424)
    X, y = generate_data(func, NUM_POINTS)
    rng.seed(3354)

    plot_line_priors(X, y)
    
def generate_data(func, N):
    Xtrain = rng.uniform(0, 10, (N, 1))
    ytrain = func(Xtrain) + rng.normal(0, 1, (N, 1))
    return Xtrain, ytrain

def plot_line_priors(Xtrain, ytrain):

    f, ax = plt.subplots(1, 1, figsize=(20, 10))

    ax.plot(Xtrain, ytrain, 'o', color='gold', markersize=30, 
        markeredgewidth=4, markeredgecolor='black')

    xs = np.linspace(0, 10, 1000)
    beta_0s = [-2, -1, 0, 1, 2]
    beta_1s = [-0.5, 0, .5]

    colors = plt.cm.cool(np.linspace(0,1,len(beta_0s) * len(beta_1s)))

    l = 0
    for b0 in beta_0s:
        for b1 in beta_1s:
            ys = b0 + b1 * xs 
            ax.plot(xs, ys, linewidth=5, color=colors[l])
            l+=1
    ax.set_xlabel('$x$', fontsize=26)
    ax.set_ylabel('$y$', fontsize=26)
    ax.xaxis.set_tick_params(labelsize=26)
    ax.yaxis.set_tick_params(labelsize=26)
    ax.set_ylim([-3, 3])
    ax.set_xlim([0, 10])
    
    plt.savefig('figures/line_priors.png', trasparent=False, pad_inches=0., bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
