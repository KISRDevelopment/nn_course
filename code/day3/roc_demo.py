import numpy as np 
import numpy.random as rng
import sklearn.metrics 

def main():


    y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0])
    y_pred = generate_ypred(y_true, 3)

    auc = sklearn.metrics.roc_auc_score(y_true, y_pred)

    for thres in np.linspace(0, 1, 9):
        hard_pred = threshold(y_pred, thres)
        acc = sklearn.metrics.accuracy_score(y_true, hard_pred)
        print("Accuracy at %0.2f: %0.2f" % (thres, acc))

    print("AUC: %0.2f" % auc)
def generate_ypred(y_true, kappa):

    y_pred = np.zeros_like(y_true, dtype=float)

    omega_1 = 0.8
    kappa_1 = kappa

    a = omega_1 * (kappa_1 - 2) + 1 
    b = (1-omega_1) * (kappa_1 - 2) + 1 

    one_ix = y_true == 1
    y_pred[one_ix] = rng.beta(a, b, size=np.sum(one_ix))

    omega_0 = 0.2 
    kappa_0 = kappa

    a = omega_0 * (kappa_0 - 2) + 1 
    b = (1 - omega_0) * (kappa_0 - 2) + 1 
    zero_ix = y_true == 0 
    y_pred[zero_ix] = rng.beta(a, b, size=np.sum(zero_ix))

    return y_pred
    
def threshold(y_pred, thres):

    min_pred = np.min(y_pred)
    max_pred = np.max(y_pred)

    y_pred = (y_pred - min_pred) / (max_pred - min_pred)

    return 0 * (y_pred < thres) + 1 * (y_pred >= thres)

if __name__ == "__main__":
    main()
