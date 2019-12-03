import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt 
import numpy.random as rng
import tensorflow.keras.utils 

def main():

    data = np.load('classification_data.npz')
    X, y = data['X'], data['y']

    # convert class number to one-hot-encoded vector
    y = tf.keras.utils.to_categorical(y)
    
    nn, losses = search(X, y, 0.01, 2000)

    # plot(X, y, nn)
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

    X_train, y_train, X_test, y_test = split_data(X, y, 0.2)
    
    # convert to tensors 
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

    # create nn 
    trainables, nn = create_nn(2, y.shape[1])

    # create optimizer
    optimizer = tf.keras.optimizers.Nadam(learning_rate=eta)

    # repeat N times
    loss_func = xe_loss
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

def plot_losses(losses):

    f, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax.plot(np.arange(losses.shape[0]), losses[:,0], linewidth=5, label='training')
    ax.plot(np.arange(losses.shape[0]), losses[:,1], linewidth=5, label='testing')
    ax.set_xlabel('Iteration', fontsize=26)
    ax.set_ylabel('Loss', fontsize=26)
    ax.xaxis.set_tick_params(labelsize=26)
    ax.yaxis.set_tick_params(labelsize=26)
    ax.legend(fontsize=26)
    plt.show()

def create_nn(n_hidden, n_classes):

    # hidden layer weight matrix 
    Wh = tf.Variable(rng.random((2, n_hidden)), dtype=tf.float32)

    # hidden layer bias vector
    bh = tf.Variable(rng.random((n_hidden, 1)), dtype=tf.float32)

    # output layer weight matrix 
    Wo = tf.Variable(rng.random((n_hidden, n_classes)), dtype=tf.float32)

    # output layer bias vector
    bo = tf.Variable(rng.random((n_classes,1)), dtype=tf.float32)

    # this is a list of things that can be trained by the 
    # SGD optimizer
    trainables = [Wh, bh, Wo, bo]

    def call(X):
        """ X is [n_samples, 2] """

        # compute hidden activations
        # hidden_input and hidden_activation are [n_samples, n_hidden]
        hidden_input = tf.matmul(X, Wh) + tf.transpose(bh)
        hidden_activation = tf.tanh(hidden_input)

        # compute output activation [n_samples, n_classes]
        logits = tf.matmul(hidden_activation, Wo) + tf.transpose(bo)

        # finally we apply softmax to transform logits 
        # into probabilities
        output = tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis=1, keepdims=True)

        return output 
    
    return trainables, call

def xe_loss(ytrue, ypred):
    """ ytrue: [n_samples, n_classes]
        ypred: [n_samples, n_classes] """
    loss = tf.reduce_sum(ytrue * ypred, axis=1)
    return tf.reduce_mean(loss)

if __name__ == "__main__":
    main()
