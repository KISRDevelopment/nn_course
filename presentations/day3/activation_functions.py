import numpy as np 
import matplotlib.pyplot as plt 
import numpy.random as rng 

def main():
    funcs = [linear, tanh, relu]

    xs = np.linspace(-2, 2, 1000)

    f, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    for i, func in enumerate(funcs):
        ys = func(xs)
        axes[i].plot(xs, ys, linewidth=5, color='magenta')
        axes[i].set_title(func.__name__, fontsize=26)
        axes[i].plot(xs, np.zeros_like(xs), linewidth=2, linestyle='--', color='red')
        axes[i].xaxis.set_tick_params(labelsize=26)
        axes[i].yaxis.set_tick_params(labelsize=26)
        if i == 1:
            axes[i].set_ylabel('Activation', fontsize=26)

    plt.savefig('figures/activation_functions.png', trasparent=False, pad_inches=0., bbox_inches='tight')
    
    #plt.show()


def tanh(x):
    return np.tanh(x)

def linear(x):
    return x

def relu(x):
    return np.maximum(0, x)

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
