import numpy as np
import copy
import math


def load_data(file):
    data = np.loadtxt(file, delimiter=',')
    X = data[:, 0]
    y = data[:, 1]
    return X, y


def load_data_multi(file):
    data = np.loadtxt(file, delimiter='\t',
                      skiprows=1, usecols=(1, 2, 3, 4, 5, 6, 7))
    X = data[:, :6]
    y = data[:, 6]
    return X, y


def load_data_header(file):

    with open(file, 'r', encoding='utf-8') as f:  # 打开文件
        lines = f.readlines()  # 读取所有行
        first_line = lines[0]  # 取第一行
        # print(first_line)
        # print(first_line.split("\t")[1:-1])
        return first_line.split("\t")[1:-1]


# data = pd.read_csv('./data/ex1data1.txt', sep='\t')


# train=pd.read_csv('test.tsv', sep='\t', header=0)


# train=pd.read_csv('test.tsv', sep='\t', header=0, index_col='id')


def zscore_normalize_features(X, rtn_ms=False):
    """
    returns z-score normalized X by column
    Args:
      X : (numpy array (m,n)) 
    Returns
      X_norm: (numpy array (m,n)) input normalized by column
    """
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu)/sigma

    if rtn_ms:
        return (X_norm, mu, sigma)
    else:
        return (X_norm)


def compute_cost_matrix(X, y, w, b, lambda_=0):
    """
    Computes the cost using  using matrices
    Args:
      X : (ndarray, Shape (m,n))          matrix of examples
      y : (ndarray  Shape (m,) or (m,1))  target value of each example
      w : (ndarray  Shape (n,) or (n,1))  Values of parameter(s) of the model
      b : (scalar )                       Values of parameter of the model
      verbose : (Boolean) If true, print out intermediate value f_wb
    Returns:
      total_cost: (scalar)                cost
    """
    m = X.shape[0]
    y = y.reshape(-1, 1)             # ensure 2D

    # (m,n)(n,1) = (m,1)
    f = X @ w + b

    cost = (1/(2*m)) * np.sum((f - y)**2)

    # scalar
    reg_cost = (lambda_/(2*m)) * np.sum(w**2)

    total_cost = cost + reg_cost                                                # scalar

    # scalar
    return total_cost


def compute_gradient_matrix(X, y, w, b, lambda_=0):
    """
    Computes the gradient using matrices

    Args:
      X : (ndarray, Shape (m,n))          matrix of examples
      y : (ndarray  Shape (m,) or (m,1))  target value of each example
      w : (ndarray  Shape (n,) or (n,1))  Values of parameters of the model
      b : (scalar )                       Values of parameter of the model
      lambda_:  (float)                   applies regularization if non-zero
    Returns
      dj_dw: (array_like Shape (n,1))     The gradient of the cost w.r.t. the parameters w
      dj_db: (scalar)                     The gradient of the cost w.r.t. the parameter b
    """
    m = X.shape[0]
    y = y.reshape(-1, 1)             # ensure 2D
    w = w.reshape(-1, 1)             # ensure 2D

    # (m,n)(n,1) = (m,1)
    f_wb = X @ w + b
    err = f_wb - y                                              # (m,1)
    # (n,m)(m,1) = (n,1)
    dj_dw = (1/m) * (X.T @ err)
    dj_db = (1/m) * np.sum(err)                                   # scalar

    dj_dw += (lambda_/m) * w        # regularize                  # (n,1)

    # scalar, (n,1)
    return dj_dw, dj_db


def gradient_descent(X, y, w_in, b_in, alpha, num_iters, lambda_=0, verbose=True):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking
    num_iters gradient steps with learning rate alpha

    Args:
      X (ndarray):    Shape (m,n)         matrix of examples
      y (ndarray):    Shape (m,) or (m,1) target value of each example
      w_in (ndarray): Shape (n,) or (n,1) Initial values of parameters of the model
      b_in (scalar):                      Initial value of parameter of the model
      lambda_:  (float)                   applies regularization if non-zero
      alpha (float):                      Learning rate
      num_iters (int):                    number of iterations to run gradient descent

    Returns:
      w (ndarray): Shape (n,) or (n,1)    Updated values of parameters; matches incoming shape
      b (scalar):                         Updated value of parameter
    """
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  # avoid modifying global w within function
    b = b_in
    w = w.reshape(-1, 1)  # prep for matrix operations
    y = y.reshape(-1, 1)

    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_dw, dj_db = compute_gradient_matrix(X, y, w, b, lambda_)

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Save cost J at each iteration
        if i < 100000:      # prevent resource exhaustion
            J_history.append(compute_cost_matrix(X, y, w, b, lambda_))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0:
            if verbose:
                print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")

    # return final w,b and J history for graphing
    return w.reshape(w_in.shape), b, J_history
