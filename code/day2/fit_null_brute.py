import numpy as np 
import matplotlib.pyplot as plt 

def main():

    data = np.load('fake_data.npz')
    X, y = data['X'], data['y']

    c_star, c_star_index = search(X, y)
    plot(X, y, c_star)
    plot_gradient(y, np.linspace(-3, 3, 100), c_star_index)

def search(X, y):

    # initialize range of c values to examine
    c_vals = np.linspace(-3, 3, 100)

    # for each value of c in the range
    model_losses = []
    for c in c_vals:

        # compute model predictions
        yhat = c * np.ones_like(y)

        # compute loss, given those predictions
        model_loss = np.mean(np.abs(y - yhat))
        model_losses.append(model_loss)
    
    # pick the c that minimizes the loss
    c_star_index = np.argmin(model_losses)
    c_star = c_vals[c_star_index]

    print("Best c: %0.3f, MAE: %0.3f" % (c_star, model_losses[c_star_index]))
    print("Median: %0.3f, Mean: %0.3f" % (np.median(y), np.mean(y)))

    plot_loss(c_vals, model_losses, c_star_index)

    return c_star, c_star_index

def plot(X, y, c_star):

    f, ax = plt.subplots(1, 1, figsize=(20, 10))

    ax.plot(X, y, marker='o', linestyle='', color='gold', 
        markersize=30, markeredgewidth=4, markeredgecolor='black')

    xs = np.linspace(0, 10, 1000)
    ys = c_star * np.ones_like(xs)
    ax.plot(xs, ys, linestyle='dashed', color='purple', 
        linewidth=5)
    ax.set_xlabel('$x$', fontsize=26)
    ax.set_ylabel('$y$', fontsize=26)
    ax.xaxis.set_tick_params(labelsize=26)
    ax.yaxis.set_tick_params(labelsize=26)

    plt.savefig('tmp/best_null.png', trasparent=False, pad_inches=0., bbox_inches='tight')

    plt.show()

def plot_loss(c_vals, model_losses, c_star_index):

    f, ax = plt.subplots(1, 1, figsize=(10, 10))

    ax.plot(c_vals, model_losses, linewidth=5)

    ax.plot(c_vals[c_star_index], model_losses[c_star_index], 
        marker='*',
        color='red',
        markersize=30
    )

    
    ax.set_xlabel('$c$', fontsize=26)
    ax.set_ylabel('MAE', fontsize=26)
    ax.xaxis.set_tick_params(labelsize=26)
    ax.yaxis.set_tick_params(labelsize=26)

    plt.savefig('tmp/null_loss.png', trasparent=False, pad_inches=0.05, 
        bbox_inches='tight')

    plt.show()

def plot_gradient(ytrue, c_vals, c_star_index):

    grads = []
    for c in c_vals:
        grads.append(grad_loss(ytrue, c))
    
    f, ax = plt.subplots(1, 1, figsize=(10, 10))

    ax.plot(c_vals, grads, linewidth=5)

    ax.plot(c_vals[c_star_index], grads[c_star_index], 
        marker='*',
        color='red',
        markersize=30
    )

    ax.set_xlabel('$c$', fontsize=26)
    ax.set_ylabel('Gradeint', fontsize=26)
    ax.xaxis.set_tick_params(labelsize=26)
    ax.yaxis.set_tick_params(labelsize=26)


    plt.savefig('tmp/null_gradient.png', trasparent=False, pad_inches=0.05, 
        bbox_inches='tight')

    plt.show()

def grad_loss(ytrue, ypred):

    # compute c - yi 
    diff = ypred - ytrue

    # compute the derivative d | c - yi | / dc 
    grad_i = 1 * (diff > 0) + (-1) * (diff < 0)

    # compute the overall gradient
    return np.mean(grad_i)

if __name__ == "__main__":
    main()
