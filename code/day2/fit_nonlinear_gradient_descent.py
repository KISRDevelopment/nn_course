import numpy as np 
import matplotlib.pyplot as plt 
import numpy.random as rng
def main():

    data = np.load('fake_data.npz')
    X, y = data['X'], data['y']

    theta = search(X, y, 0.01, 10000)
    plot(X, y, theta)

def search(X, y, eta, iterations):

    # randomly initialize beta0 and beta1
    theta = rng.randn(4)

    # repeat N times
    for i in range(iterations):
        
        # compute gradient 
        grad = grad_loss(X, y, theta)

        # update 
        theta = theta - eta * grad

        _, ypred = predict(X, theta)

        mae = np.mean(np.abs(ypred - y))

        print("%d Loss: %f" % (i, mae))

    # final prediction
    _, ypred = predict(X, theta)

    mae = np.mean(np.abs(ypred - y))

    beta0, beta1, gamma0, gamma1 = theta 

    print("Best beta0: %0.3f, beta1: %0.3f, gamma0: %0.3f, gamma1: %0.3f, MAE: %0.3f" % (beta0, beta1, 
        gamma0, gamma1, mae))
    
    return theta


def plot(X, y, theta):

    f, ax = plt.subplots(1, 1, figsize=(20, 10))

    ax.plot(X, y, marker='o', linestyle='', color='gold', 
        markersize=30, markeredgewidth=4, markeredgecolor='black')

    beta0 = theta[0]
    beta1 = theta[1]
    gamma0 = theta[2]
    gamma1 = theta[3]

    xs = np.linspace(0, 10, 1000)
    ys = gamma0 + gamma1 * np.exp(beta0 + beta1*xs)
    ax.plot(xs, ys, linestyle='dashed', color='purple', 
        linewidth=5)
    ax.set_xlabel('$x$', fontsize=26)
    ax.set_ylabel('$y$', fontsize=26)

    plt.savefig('tmp/best_nonlinear.png', trasparent=False, pad_inches=0., bbox_inches='tight')

    plt.show()

def predict(X, theta):
    beta0, beta1, gamma0, gamma1 = theta 

    g = np.exp(beta0 + beta1 * X)
    ypred = gamma0 + gamma1 * g 

    return g, ypred 

def grad_loss(X, y, theta):
    beta0, beta1, gamma0, gamma1 = theta 

    g, ypred = predict(X, theta)

    diff = ypred - y 

    # gradients
    d_beta0 = gamma1 * g * (diff > 0) - gamma1 * g * (diff < 0)
    d_beta1 = gamma1 * X * g * (diff > 0) - gamma1 * X * g * (diff < 0)
    d_gamma0 = (diff > 0) * 1 -  1 * (diff < 0)
    d_gamma1 = (diff > 0) * g - (diff < 0) * g 

    return np.array([
        np.mean(d_beta0),
        np.mean(d_beta1),
        np.mean(d_gamma0),
        np.mean(d_gamma1)
    ])

if __name__ == "__main__":
    main()
