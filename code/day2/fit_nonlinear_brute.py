import numpy as np 
import matplotlib.pyplot as plt 

from mpl_toolkits.mplot3d import Axes3D
def main():

    data = np.load('fake_data.npz')
    X, y = data['X'], data['y']

    B0, B1, Z = compute_loss(X, y)
    plot_loss(B0, B1, Z)

def compute_loss(X, y):

    beta1 = np.linspace(-1, 1, 100)
    gamma1 = np.linspace(-1, 1, 100)

    B0, B1 = np.meshgrid(beta1, gamma1)
    Z = np.zeros_like(B0)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            beta1 = B0[i,j]
            gamma1 = B1[i,j]

            # compute model predictions
            yhat = gamma1 * np.exp(beta1 * X)

            Z[i,j] = np.mean(np.abs(yhat - y))
    
    return B0, B1, Z 
def plot_loss(B0, B1, Z):
    
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

    ax2.set_xlabel(r'$\beta_0$', fontsize=26)
    ax2.set_ylabel(r'$\beta_1$', fontsize=26)
    ax2.xaxis.set_tick_params(labelsize=26)
    ax2.yaxis.set_tick_params(labelsize=26)
    cbar = f.colorbar(cp, ax=ax2)
    cbar.ax.tick_params(labelsize=20) 

    plt.show()

if __name__ == "__main__":
    main()
