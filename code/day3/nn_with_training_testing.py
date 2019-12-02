import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt 
import numpy.random as rng

def main():

    data = np.load('fake_data.npz')
    X, y = data['X'], data['y']

    nn, losses = search(X, y, 0.01, 1000)

    #plot(X, y, nn)
    plot_losses(losses)

def split_data(X, y, prop):

    # shuffle indecies
    ix = rng.permutation(X.shape[0])

    # how many training examples?
    n_train = int(X.shape[0] * (1-prop))

    train_ix = ix[:n_train]
    test_ix = ix[n_train:]

    X_train = X[train_ix,:]
    y_train = y[train_ix,:]

    X_test = X[test_ix, :]
    y_test = y[test_ix, :]

    return X_train, y_train, X_test, y_test

def search(X, y, eta, iterations):

    X_train, y_train, X_test, y_test = split_data(X, y, 0.8)
    
    # convert to tensors 
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

    # create nn 
    trainables, nn = create_nn(50)

    # create optimizer
    optimizer = tf.keras.optimizers.Nadam(learning_rate=eta)

    # repeat N times
    loss_func = mse_loss
    losses = []
    for i in range(iterations):
        
        with tf.GradientTape() as g:
            ypred = nn(X_train)
            loss = loss_func(y_train, ypred)

        grads = g.gradient(loss, trainables)
        optimizer.apply_gradients(zip(grads, trainables))

        ypred_train = nn(X_train)
        train_loss = loss_func(y_train, ypred_train)

        ypred_test = nn(X_test)
        test_loss = loss_func(y_test, ypred_test)

        losses.append((train_loss.numpy(), test_loss.numpy()))
        print("%d Training Loss: %f, Testing Loss: %f" % (i, train_loss, test_loss))

    # final prediction
    ypred_train = nn(X_train)
    train_loss = loss_func(y_train, ypred_train)
    print("Best training loss: %0.3f" % (train_loss))
    
    losses = np.array(losses)
    return nn, losses

def plot(X, y, nn):

    f, ax = plt.subplots(1, 1, figsize=(20, 10))

    ax.plot(X, y, marker='o', linestyle='', color='gold', 
        markersize=30, markeredgewidth=4, markeredgecolor='black')

    xs = np.linspace(0, 10, 1000)[:,None].astype(np.float32)
    ys = nn(xs)
    ax.plot(xs, ys, linestyle='dashed', color='purple', 
        linewidth=5)
    ax.set_xlabel('$x$', fontsize=26)
    ax.set_ylabel('$y$', fontsize=26)
    ax.xaxis.set_tick_params(labelsize=26)
    ax.yaxis.set_tick_params(labelsize=26)

    plt.savefig('tmp/best_nn.png', trasparent=False, pad_inches=0., bbox_inches='tight')

    plt.show()

def plot_losses(losses):

    f, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax.plot(np.arange(losses.shape[0]), losses[:,0], linewidth=5, label='training')
    ax.plot(np.arange(losses.shape[0]), losses[:,1], linewidth=5, label='testing')
    ax.set_xlabel('Iteration', fontsize=26)
    ax.set_ylabel('MAE', fontsize=26)
    ax.xaxis.set_tick_params(labelsize=26)
    ax.yaxis.set_tick_params(labelsize=26)
    ax.legend(fontsize=26)
    plt.show()

def create_nn(n_hidden):

    # hidden layer weight matrix 
    Wh = tf.Variable(rng.random((1, n_hidden)), dtype=tf.float32)

    # hidden layer bias vector
    bh = tf.Variable(rng.random((n_hidden, 1)), dtype=tf.float32)

    # output layer weight matrix 
    Wo = tf.Variable(rng.random((n_hidden, 1)), dtype=tf.float32)

    # output layer bias vector
    bo = tf.Variable(rng.random((1,1)), dtype=tf.float32)

    # this is a list of things that can be trained by the 
    # SGD optimizer
    trainables = [Wh, bh, Wo, bo]

    def call(X):
        """ X is [n_samples, 1] """

        # compute hidden activations
        # hidden_input and hidden_activation are [n_samples, n_hidden]
        hidden_input = tf.matmul(X, Wh) + tf.transpose(bh)
        hidden_activation = tf.tanh(hidden_input)

        # compute output activation [n_samples, 1]
        output = tf.matmul(hidden_activation, Wo) + tf.transpose(bo)

        return output 
    
    return trainables, call

def mae_loss(ytrue, ypred):
    return tf.reduce_mean(tf.abs(ypred - ytrue))

def mse_loss(ytrue, ypred):
    return tf.reduce_mean(tf.square(ypred - ytrue))

if __name__ == "__main__":
    main()
