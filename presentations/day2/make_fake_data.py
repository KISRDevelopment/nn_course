import numpy as np 
import matplotlib.pyplot as plt 
import numpy.random as rng 

NUM_POINTS = 30

func = lambda x: np.cos(x) + np.sin(2 * x) + np.sin(1.5 * x) + np.cos(2.5 * x)

def main():

    rng.seed(3424)
    X, y = generate_data(func, NUM_POINTS)
    plot_points(X, y, 'figures/fake_data.png')
    np.savez('tmp/fake_data', X=X, y=y)

    rng.seed(879)
    X = np.linspace(1, 9, 5)
    y = func(X) + rng.normal(0, 1, (5,))
    predictor = lambda xs: 0.5 * np.ones_like(xs)
    plot_points(X, y, 'figures/loss_demo.png', predictor)

    predictor = lambda xs: -2 + 0.5*xs
    plot_points(X, y, 'figures/loss_demo_line.png', predictor)


def generate_data(func, N):
    Xtrain = rng.uniform(0, 10, (N, 1))
    ytrain = func(Xtrain) + rng.normal(0, 1, (N, 1))
    return Xtrain, ytrain

def plot_points(Xtrain, ytrain, output_path, predictor=None):

    f, ax = plt.subplots(1, 1, figsize=(20, 10))


    ax.plot(Xtrain, ytrain, 'o', color='gold', markersize=30, markeredgewidth=4, markeredgecolor='black')

    if predictor is not None:
        xs = np.linspace(0, 10, 1000)
        ys = predictor(xs)
        ax.plot(xs, ys, '--', color='purple', linewidth=5, label='prediction')
        for k in range(Xtrain.shape[0]):
            ax.plot([Xtrain[k], Xtrain[k]], [ytrain[k], predictor(Xtrain[k])], color='red', linewidth=5)
        
    ax.set_xlabel('$x$', fontsize=26)
    ax.set_ylabel('$y$', fontsize=26)
    ax.xaxis.set_tick_params(labelsize=26)
    ax.yaxis.set_tick_params(labelsize=26)
    ax.set_ylim([-3, 3])
    ax.set_xlim([0, 10])

    if predictor is not None:
        ax.legend(fontsize=26)
    
    plt.savefig(output_path, trasparent=False, pad_inches=0., bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
