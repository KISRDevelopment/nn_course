import numpy as np 
import matplotlib.pyplot as plt 
import numpy.random as rng 

def main():

    mse_vs_xe()
    caliberation()

def mse_vs_xe():
    y_target = 1.0 

    preds = np.linspace(0, 1, 1000)

    f, ax = plt.subplots(1, 1, figsize=(5, 5))

    xe_loss = y_target * np.log(preds) + (1-y_target) * np.log(1-preds)
    ax.plot(preds, -xe_loss)

    mse_loss = np.square(preds - y_target)
    ax.plot(preds, mse_loss)

    plt.show()

def caliberation():

    distrib =  0.25
    samples = rng.binomial(1, distrib, 10)

    mus = np.linspace(0.01, 0.99, 1000)
    xe_losses = []
    mse_losses = []
    for mu in mus:
        xe_loss = samples * np.log(mu) + (1-samples) * np.log(1-mu)
        mse_loss = np.square(samples - mu)
        
        xe_losses.append(np.mean(xe_loss))
        mse_losses.append(np.mean(mse_loss))
    
    ix_best_xe_mu = np.argmax(xe_losses)
    ix_best_mse_mu = np.argmin(mse_losses)

    print("Best by XE: %0.2f" % mus[ix_best_xe_mu])
    print("Best by MSE: %0.2f" % mus[ix_best_mse_mu])
if __name__ == "__main__":
    main()
