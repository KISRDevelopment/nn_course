import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
import numpy.random as rng 
from mpl_toolkits.mplot3d import Axes3D

def main():

    data = np.load('fake_data.npz')
    X, y = data['X'], data['y']


    B0, B1, Z = compute_loss(X, y)
    
    
    
    #plot_loss(B0, B1, Z , steps)
    
    f, axes = plt.subplots(1, 3, figsize=(20, 5))

    cp = plot_steps(axes[0], B0, B1, Z, search(X, y, 0.01, 500), title=r'$\eta = 0.01$')
    cp = plot_steps(axes[1], B0, B1, Z, search(X, y, 0.05, 500), title=r'$\eta = 0.05$')
    cp = plot_steps(axes[2], B0, B1, Z, search(X, y, 0.2, 500), title=r'$\eta = 0.2$')

    cbar = plt.gcf().colorbar(cp, ax=axes)
    cbar.ax.tick_params(labelsize=20) 

    plt.savefig('tmp/linear_gd_steps.png', trasparent=False, pad_inches=0.05, 
        bbox_inches='tight')

    plt.show()
    
def compute_loss(X, y):

    beta0 = np.linspace(-3, 3, 100)
    beta1 = np.linspace(-3, 3, 100)

    B0, B1 = np.meshgrid(beta0, beta1)
    Z = np.zeros_like(B0)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i,j] = np.mean(np.abs(B0[i,j] + B1[i,j] * X - y))
    
    return B0, B1, Z 

def plot_loss(B0, B1, Z, steps):
    
    f = plt.figure(figsize=(20, 7.5))

    ax1 = f.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(B0, B1, Z)
    ax1.set_xlabel(r'$\beta_0$', fontsize=26)
    ax1.set_ylabel(r'$\beta_1$', fontsize=26)
    ax1.set_zlabel(r'MAE', fontsize=26)
    ax1.xaxis.set_tick_params(labelsize=20)
    ax1.yaxis.set_tick_params(labelsize=20)

    ax2 = f.add_subplot(1, 2, 2)
    cp = ax2.contourf(B0, B1, Z)

    ax2.plot(steps[:,0], steps[:,1], marker='o',
        linewidth=2,
        markersize=3,
        color='orange',
        label='steps')
    
    ax2.set_xlabel(r'$\beta_0$', fontsize=26)
    ax2.set_ylabel(r'$\beta_1$', fontsize=26)
    ax2.xaxis.set_tick_params(labelsize=26)
    ax2.yaxis.set_tick_params(labelsize=26)
    cbar = f.colorbar(cp, ax=ax2)
    cbar.ax.tick_params(labelsize=20) 


    plt.savefig('tmp/linear_loss.png', trasparent=False, pad_inches=0.05, 
        bbox_inches='tight')

    plt.show()

def plot_steps(ax, B0, B1, Z, steps, title):
    
    cp = ax.contourf(B0, B1, Z)

    ax.plot(steps[:,0], steps[:,1], marker='o',
        linewidth=2,
        markersize=3,
        color='orange',
        label='steps')
    
    
    ax.xaxis.set_tick_params(labelsize=26)
    ax.yaxis.set_tick_params(labelsize=26)
    ax.set_title(title, fontsize=36)
    
    return cp

def search(X, y, eta, iterations):
    
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.float32)

    # randomly initialize beta0 and beta1
    beta = tf.constant([-2, 2.5])

    # repeat N times
    steps = [ beta.numpy() ]
    for i in range(iterations):
        
        # compute model predictions
        ypred = beta[0] + beta[1] * X

        # compute gradient 
        grad = grad_loss(X, y, ypred)

        # update 
        beta = beta - eta * grad
        steps.append(beta.numpy())
    
    mae = tf.reduce_mean(tf.abs(beta[0] + beta[1] * X - y))

    print("beta0 = %0.3f, beta1 = %0.3f, loss = %0.3f" % (beta[0], beta[1], mae))
    
    return np.array(steps) 

def grad_loss(X, y, ypred):

    diff = ypred - y

    cond1 = tf.where(diff > 0, tf.ones_like(diff), -tf.ones_like(diff))
    cond2 = tf.where(diff > 0, X, -X)

    grad_beta0 = tf.reduce_mean(cond1)
    grad_beta1 = tf.reduce_mean(cond2)

    return tf.stack([grad_beta0, grad_beta1])


if __name__ == "__main__":
    main()
