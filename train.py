from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from util import y2indicator, error_rate, init_weight_and_bias
from sklearn.utils import shuffle
import pandas as pd
from keras.utils import np_utils


class HiddenLayer(object):
    def __init__(self, M1, M2, an_id):
        self.id = an_id
        self.M1 = M1
        self.M2 = M2
        W, b = init_weight_and_bias(M1, M2)
        self.W = tf.Variable(W.astype(np.float32))
        self.b = tf.Variable(b.astype(np.float32))
        self.params = [self.W, self.b]

    def forward(self, X):
        return tf.nn.relu(tf.matmul(X, self.W) + self.b)


class ANN(object):
    def __init__(self, hidden_layer_sizes):
        self.hidden_layer_sizes = hidden_layer_sizes

    def fit(self, X, Y,class_weights, learning_rate=1e-2, mu=0.99, decay=0.999, reg=1e-3, epochs=10, batch_sz=100, show_fig=False):
        K = 5 # won't work later b/c we turn it into indicator

        # make a validation set
        X, Y = shuffle(X, Y)
        X = X.astype(np.float32)
        Y = y2indicator(Y).astype(np.float32)
        # Y = Y.astype(np.int32)
        Xvalid, Yvalid = X[-1000:], Y[-1000:]
        Yvalid_flat = np.argmax(Yvalid, axis=1) # for calculating error rate
        X, Y = X[:-1000], Y[:-1000]

        # initialize hidden layers
        N, D = X.shape
        
        self.hidden_layers = []
        M1 = D
        count = 0
        for M2 in self.hidden_layer_sizes:
            h = HiddenLayer(M1, M2, count)
            self.hidden_layers.append(h)
            M1 = M2
            count += 1
        W, b = init_weight_and_bias(M1, K)
        self.W = tf.Variable(W.astype(np.float32))
        self.b = tf.Variable(b.astype(np.float32))

        # collect params for later use
        self.params = [self.W, self.b]
        for h in self.hidden_layers:
            self.params += h.params

        # set up theano functions and variables
        tfX = tf.placeholder(tf.float32, shape=(None, D), name='X')
        tfT = tf.placeholder(tf.float32, shape=(None, K), name='T')
        act = self.forward(tfX)

        rcost = reg*sum([tf.nn.l2_loss(p) for p in self.params])
        weights = tf.reduce_sum(class_weights * tfT, axis=1)
        cost = tf.nn.softmax_cross_entropy_with_logits(
                logits=act,
                labels=tfT) 
        weighted_losses = cost * weights
        loss = tf.reduce_mean(weighted_losses) + rcost

        prediction = self.predict(tfX)
        train_op = tf.train.RMSPropOptimizer(learning_rate, decay=decay, momentum=mu).minimize(loss)

        n_batches = N // batch_sz
        costs = []
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            for i in range(epochs):
                X, Y = shuffle(X, Y)
                for j in range(n_batches):
                    Xbatch = X[j*batch_sz:(j*batch_sz+batch_sz)]
                    Ybatch = Y[j*batch_sz:(j*batch_sz+batch_sz)]

                    session.run(train_op, feed_dict={tfX: Xbatch, tfT: Ybatch})

                    if j % 20 == 0:
                        c = session.run(cost, feed_dict={tfX: Xvalid, tfT: Yvalid})
                        costs.append(c)

                        p = session.run(prediction, feed_dict={tfX: Xvalid, tfT: Yvalid})
                        e = error_rate(Yvalid_flat, p)
                        print("i:", i, "j:", j, "nb:", n_batches, "cost:", c, "error rate:", e)
        
        if show_fig:
            plt.plot(costs)
            plt.show()

    def forward(self, X):
        Z = X
        for h in self.hidden_layers:
            Z = h.forward(Z)
        return tf.matmul(Z, self.W) + self.b

    def predict(self, X):
        act = self.forward(X)
        return tf.argmax(act, 1)


def main():
    X_es = np.loadtxt('es_train.txt')
    X_fr = np.loadtxt('fr_train.txt')
    X_en = np.loadtxt('en_train.txt')

    y_es = np.loadtxt('y_es.txt')
    y_fr = np.loadtxt('y_fr.txt')
    y_en = np.loadtxt('y_en.txt')

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = pd.read_csv('gdrive/My Drive/HackerEarth/Soc-Gen/train.csv')['Complaint-Status'].iloc[:].values
    y = le.fit_transform(y)
    y = np_utils.to_categorical(y)
    total = len(y)/np.sum(y, axis = 0)
    class_weights = tf.constant([list(total)])



    esmodel = ANN([2000, 2000, 2000, 1000])
    esmodel.fit(X_es, y_es, class_weights, show_fig=False)
    frmodel = ANN([2000, 2000, 2000, 1000])
    frmodel.fit(X_fr, y_fr, class_weights, show_fig=False)
    enmodel = ANN([2000, 2000, 2000, 1000])
    enmodel.fit(X_en, y_en, class_weights, show_fig=False)
    X_test = np.loadtxt('X_test.txt').astype(np.float32)
    languages = X_test[:, 0]
    X_test = X_test[:, 1:]
    z = np.ones((len(X_test), 1))
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(X_test.shape[0]):
        if languages[i] == 3:
            z[i] = sess.run(enmodel.predict(X_test[i, :].reshape((1, 300))))
        elif langauges[i] ==4:
            z[i] = sess.run(esmodel.predict(X_test[i, :].reshape((1, 300))))
        elif languages[i] ==5:
            z[i] = sess.run(frmodel.predict(X_test[i, :].reshape((1, 300))))

    z1 = pd.DataFrame(columns = ['Complaint-ID', 'Complaint-Status'], index = None)
    z1['Complaint-ID'] = pd.read_csv(os.path.join('test.csv'))['Complaint-ID'].iloc[:].values
    z1['Complaint-Status'] = z
    z1.to_csv('weighted_submission.csv', index = None)


if __name__ == '__main__':
    main()
