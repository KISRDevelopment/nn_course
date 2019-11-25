import numpy as np 
import matplotlib.pyplot as plt 
import numpy.random as rng 

def main():

    data = np.load('fake_data.npz')
    X, y = data['X'], data['y']

    c_vals, model_losses = compute_loss(X, y)

    f, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=True)


    steps, step_losses = search(X, y, 0.1)
    plot_steps(axes[0], c_vals, model_losses, steps, step_losses, '$\eta = 0.1$')

    steps, step_losses = search(X, y, 1.0)
    plot_steps(axes[1], c_vals, model_losses, steps, step_losses, '$\eta = 1.0$')

    steps, step_losses = search(X, y, 5.0)
    plot_steps(axes[2], c_vals, model_losses, steps, step_losses, '$\eta = 3.0$')


    plt.savefig('tmp/null_gd_steps.png', trasparent=False, pad_inches=0.05, 
        bbox_inches='tight')


    plt.show()
def search(X, y, eta):

    # randomly initialize c 
    c = 2.5

    # repeat N times
    steps = []
    losses = []

    for i in range(15):
        
        # compute model predictions
        ypred = c 

        # compute gradient 
        grad = grad_loss(y, c)

        # compute loss (for reporting only)
        mae = np.mean(np.abs(c - y))

        steps.append(c)
        losses.append(mae)

        #print("c = %0.3f, grad = %0.3f, loss = %0.3f" % (c, grad, mae))

        # update 
        c = c - eta * grad


    mae = np.mean(np.abs(c - y))

    steps.append(c)
    losses.append(mae)
    print("c = %0.3f, loss = %0.3f" % (c, mae))
    
    return steps, losses

def grad_loss(ytrue, ypred):

    # compute c - yi 
    diff = ypred - ytrue

    # compute the derivative d | c - yi | / dc 
    grad_i = 1 * (diff > 0) + (-1) * (diff < 0)

    # compute the overall gradient
    return np.mean(grad_i)

def compute_loss(X, y):

    c_vals = np.linspace(-3, 3, 100)
    # for each value of c in the range
    model_losses = []
    for c in c_vals:

        # compute model predictions
        yhat = c * np.ones_like(y)

        # compute loss, given those predictions
        model_loss = np.mean(np.abs(y - yhat))
        model_losses.append(model_loss)
    
    return c_vals, model_losses

def plot_steps(ax, c_vals, model_losses, steps, step_losses, title):

    ax.plot(c_vals, model_losses, linewidth=5, label='loss')

    
    ax.plot(steps, step_losses, marker='o',
        linewidth=5,
        markersize=10,
        label='steps')
    
    ax.set_xlabel('$c$', fontsize=26)
    ax.set_ylabel('$MAE$', fontsize=26)
    ax.xaxis.set_tick_params(labelsize=26)
    ax.yaxis.set_tick_params(labelsize=26)
    ax.set_title(title, fontsize=36)

if __name__ == "__main__":
    main()
