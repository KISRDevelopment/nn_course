import numpy as np 
import matplotlib.pyplot as plt 

def main():

    data = np.load('fake_data.npz')
    X, y = data['X'], data['y']

    beta = search(X, y)
    plot(X, y, beta)

def search(X, y):
    N = 100

    # initialize range of beta values to examine
    beta0_vals = np.linspace(-3, 3, N)
    beta1_vals = beta0_vals

    # for each value of  in the range
    model_losses = np.zeros((N, N))
    for i, beta0 in enumerate(beta0_vals):
        for j, beta1 in enumerate(beta1_vals):

            # compute model predictions
            yhat = beta0 + beta1 * X

            # compute loss, given those predictions
            model_losses[i, j] = np.mean(np.abs(y - yhat))
    
    # pick the beta that minimizes the loss
    beta_star_flat_index = np.argmin(model_losses, axis=None)
    beta_star_index = np.unravel_index(beta_star_flat_index, model_losses.shape)
    beta_0 = beta0_vals[beta_star_index[0]]
    beta_1 = beta1_vals[beta_star_index[1]]

    print("Best beta0: %0.3f, beta1: %0.3f, MAE: %0.3f" % (beta_0, beta_1, 
        model_losses[beta_star_index[0], beta_star_index[1]]))
    
    return beta_0, beta_1

def plot(X, y, beta):

    f, ax = plt.subplots(1, 1, figsize=(20, 10))

    ax.plot(X, y, marker='o', linestyle='', color='gold', 
        markersize=30, markeredgewidth=4, markeredgecolor='black')

    xs = np.linspace(0, 10, 1000)
    ys = beta[0] + beta[1] * xs
    ax.plot(xs, ys, linestyle='dashed', color='purple', 
        linewidth=5)
    ax.set_xlabel('$x$', fontsize=26)
    ax.set_ylabel('$y$', fontsize=26)

    plt.savefig('tmp/best_linear.png', trasparent=False, pad_inches=0., bbox_inches='tight')

    plt.show()

if __name__ == "__main__":
    main()
