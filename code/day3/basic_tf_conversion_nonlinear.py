import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt 
import numpy.random as rng

def main():

    data = np.load('fake_data.npz')
    X, y = data['X'], data['y']

    theta = search(X, y, 0.01, 1000, predict_neuron)
    plot(X, y, theta, predict_neuron)

def search(X, y, eta, iterations, predictor):

    # convert to tensors 
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.float32)

    # randomly initialize beta0 and beta1
    theta = tf.Variable(rng.randn(4), dtype=tf.float32)

    # create optimizer
    optimizer = tf.keras.optimizers.Nadam(learning_rate=eta)

    # repeat N times
    for i in range(iterations):
        
        with tf.GradientTape() as g:
            g.watch(theta)
            ypred = predictor(X, theta)
            
            loss = tf.reduce_mean(tf.abs(ypred - y))

        dloss_dtheta = g.gradient(loss, theta)

        # update with optimizer 
        optimizer.apply_gradients([(dloss_dtheta, theta)])

        # update  without optimizer
        #theta = theta - eta * dloss_dtheta

        ypred = predictor(X, theta)

        loss = tf.reduce_mean(tf.abs(ypred - y))

        print("%d Loss: %f" % (i, loss))

    # final prediction
    ypred = predictor(X, theta)
    loss = tf.reduce_mean(tf.abs(ypred - y))

    theta = theta.numpy()
    beta0, beta1, gamma0, gamma1 = theta 

    print("Best beta0: %0.3f, beta1: %0.3f, gamma0: %0.3f, gamma1: %0.3f, MAE: %0.3f" % (beta0, beta1, 
        gamma0, gamma1, loss))
    
    return theta


def plot(X, y, theta, predictor):

    f, ax = plt.subplots(1, 1, figsize=(20, 10))

    ax.plot(X, y, marker='o', linestyle='', color='gold', 
        markersize=30, markeredgewidth=4, markeredgecolor='black')

    beta0 = theta[0]
    beta1 = theta[1]
    gamma0 = theta[2]
    gamma1 = theta[3]

    xs = np.linspace(0, 10, 1000)
    ys = predictor(xs, theta)
    ax.plot(xs, ys, linestyle='dashed', color='purple', 
        linewidth=5)
    ax.set_xlabel('$x$', fontsize=26)
    ax.set_ylabel('$y$', fontsize=26)

    plt.savefig('tmp/best_nonlinear.png', trasparent=False, pad_inches=0., bbox_inches='tight')

    plt.show()

def predict(X, theta):

    linear_part = theta[0] + theta[1] * X 
    g = tf.cos(linear_part)
    ypred = theta[2]  + theta[3] * g

    return ypred

def predict_neuron(X, theta):

    linear_part = theta[0] + theta[1] * X 
    g = tf.tanh(linear_part)
    ypred = theta[2] + theta[3] * g 

    return ypred 

# def grad_loss(X, y, theta):
#     beta0, beta1, gamma0, gamma1 = theta 

#     linear_part, g, ypred = predict(X, theta)

#     diff = ypred - y 

#     # gradients
#     d_beta0 = -gamma1 * np.sin(linear_part) * (diff > 0) + \
#         gamma1 * np.sin(linear_part) * (diff < 0)
#     d_beta1 = -gamma1 * X * np.sin(linear_part) * (diff > 0) + \
#         gamma1 * X * np.sin(linear_part) * (diff < 0)
#     d_gamma0 = (diff > 0) * 1 -  1 * (diff < 0)
#     d_gamma1 = (diff > 0) * g - (diff < 0) * g 

#     return np.array([
#         np.mean(d_beta0),
#         np.mean(d_beta1),
#         np.mean(d_gamma0),
#         np.mean(d_gamma1)
#     ])

if __name__ == "__main__":
    main()
