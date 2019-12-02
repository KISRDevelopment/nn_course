import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt 
import numpy.random as rng

def main():

    data = np.load('fake_data.npz')
    X, y = data['X'], data['y']

    nn = search(X, y, 0.1, 10000)
    plot(X, y, nn)
def search(X, y, eta, iterations):

    # convert to tensors 
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.float32)

    # create nn 
    trainables, nn = create_nn(20)

    # create optimizer
    optimizer = tf.keras.optimizers.Nadam(learning_rate=eta)

    # repeat N times
    for i in range(iterations):
        
        with tf.GradientTape() as g:
            ypred = nn(X)
            loss = tf.reduce_mean(tf.abs(ypred - y))

        grads = g.gradient(loss, trainables)
        optimizer.apply_gradients(zip(grads, trainables))

        ypred = nn(X)
        loss = tf.reduce_mean(tf.abs(ypred - y))
        print("%d Loss: %f" % (i, loss))

    # final prediction
    ypred = nn(X)
    loss = tf.reduce_mean(tf.abs(ypred - y))

    print("Best MAE: %0.3f" % (loss))
    
    return nn

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

    plt.savefig('tmp/best_nn.png', trasparent=False, pad_inches=0., bbox_inches='tight')

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

if __name__ == "__main__":
    main()
