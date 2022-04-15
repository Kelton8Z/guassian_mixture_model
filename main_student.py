import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def get_mu_core(X, r, n_cluster, n_dim):
    #############
    # Your Code #
    #############

def get_sigma_core(X, r, n_cluster, n_dim):
    #############
    # Your Code #
    #############

def get_w_core(r, n_cluster, n_dim):
    #############
    # Your Code #
    #############

def get_normal_pdf_core(x, mu_scalar, sigma_scalar):
    #############
    # Your Code #
    #############

def get_initial_r(X, n_cluster):
    n_sample = len(lines)
    r = np.zeros((n_sample, n_cluster))
    for i in range(n_sample):
        if X[i,0] < -3:
            r[i,0] = 1
        elif X[i,0] < 0:
            r[i,1] = 1
        else:
            r[i,2] = 1
    return r

def get_mu(X, r):
    assert len(X.shape) == 2 and len(r.shape) == 2
    assert X.shape[0] == r.shape[0]
    n_dim = X.shape[1]
    n_cluster = r.shape[1]
    return get_mu_core(X, r, n_cluster, n_dim)

def get_sigma(X, r):
    assert len(X.shape) == 2 and len(r.shape) == 2
    assert X.shape[0] == r.shape[0]
    n_dim = X.shape[1]
    n_sample, n_cluster = r.shape
    return get_sigma_core(X, r, n_cluster, n_dim)

def get_mu_sigma(X, r):
    assert len(X.shape) == 2 and len(r.shape) == 2
    assert X.shape[0] == r.shape[0]
    mu = get_mu(X,r)
    sigma = get_sigma(X,r)
    return mu, sigma

def get_w(r):
    assert len(r.shape) == 2
    n_dim, n_cluster = r.shape
    return get_w_core(r, n_cluster, n_dim)

def get_normal_pdf(x, mu, sigma):
    assert x.size == 1
    assert mu.size == 1
    assert sigma.size == 1
    return get_normal_pdf_core(x, mu, sigma)

with open('data.txt') as f:
    lines = f.readlines()

n_cluster = 3
n_sample = len(lines)
X = np.array([[float(x) for x in l[:-2].split(' ')][:5] for l in lines])
assert len(X.shape) == 2

# Unit-test
mu_given, sigma_given = 5, 2
X_sample = np.random.normal(mu_given, sigma_given, n_sample).reshape(n_sample, -1)
r = np.ones((n_sample, 1))
mu, sigma = get_mu_sigma(X_sample, r)
print(mu, sigma)

# Univariate
X0 = X[:,:1]
print(X0.shape)
r = get_initial_r(X, n_cluster)
for k in range(100):

    # M-step
    mu, sigma = get_mu_sigma(X0, r)
    w = get_w(r)

    # E-step
    prob_normal = np.zeros((n_sample,n_cluster))
    for i in range(n_sample):
        for j in range(n_cluster):
            prob_normal[i,j] = w[j] * get_normal_pdf(X0[i,:], mu[j,:], sigma[j,:]) #numerator / deno
    r = prob_normal / prob_normal.sum(1).reshape(-1,1)

    print("mu = ", mu.tolist())
    print("sigma = ", sigma.tolist())

# Multivariate
r = get_initial_r(X, n_cluster)
for k in range(100):
    mu, sigma = get_mu_sigma(X, r)
    w = get_w(r)

    prob_normal = np.zeros((n_sample,n_cluster))
    for i in range(n_sample):
        for j in range(n_cluster):
            prob_normal[i,j] = w[j] * get_normal_pdf(X[i,:], mu[j,:], sigma[j,:]) #numerator / deno
    r = prob_normal / prob_normal.sum(1).reshape(-1,1)

    print("mu = ", mu.tolist())
    print("sigma = ", sigma.tolist())
