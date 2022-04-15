import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

'''
    Initialize θ with some values (random or otherwise).
     The parameters θ = {ωk,μk,σk2}Kk=1 includes the weights ωk, means μk and variance σk2. 
     For unit test purpose, given μ = 2,σ = 2, generate a normal distribution with 3000 sample points, i.e. np.random.normal(mu, sigma, 3000) 
    and implement a function get_mu_sigma(X) to estimate parameters μ and σ2 for your 3000 sample points.
'''

def get_mu_core(X, r, n_cluster, n_dim):
    mu = np.zeros(n_cluster)
    for j in range(n_cluster):
        mu[j] = sum(r[i][j]*X[i] for i in range(n_dim)) / sum(r[i][j] for i in range(n_dim))
    return mu

def get_sigma_core(X, r, n_cluster, n_dim, mu):
    sigma = np.zeros(n_cluster)
    for j in range(n_cluster):
        sigma[j] = sum((mu[j]-X[i])**2 for i in range(n_dim)) / n_dim
    return sigma

def get_w_core(r, n_cluster, n_dim):
    n_sample = r.shape[0]
    w = np.zeros(r.shape)
    for k in range(n_cluster):
        w[k] = np.sum(r[i][k] for i in range(n_sample)) / np.sum(np.sum(r[i][kk] for i in range(n_sample)) for kk in range(n_cluster))
    return w

def get_normal_pdf_core(x, mu_scalar, sigma_scalar):
    # normal distribution value of x given mu and sigma   ``
    return 1 / (sigma_scalar*(2*np.pi)**0.5)*np.exp(-0.5*((x-mu_scalar)/sigma_scalar)**2)

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

def get_sigma(X, r,mu):
    assert len(X.shape) == 2 and len(r.shape) == 2
    assert X.shape[0] == r.shape[0]
    n_dim = X.shape[1]
    n_sample, n_cluster = r.shape
    return get_sigma_core(X, r, n_cluster, n_dim,mu)

def get_mu_sigma(X, r):
    assert len(X.shape) == 2 and len(r.shape) == 2
    assert X.shape[0] == r.shape[0]
    mu = get_mu(X,r)
    sigma = get_sigma(X,r,mu)
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

# def lower_bound(r, )

def cov_matrix(k, r, mu, X, n_cluster, n_samples):
    return 1 / sum(r[i][k] for i in range(n_sample)) * np.sum(r[n][k]*np.matmul(X-mu[k], (X-mu[k]).T))

with open('data.txt') as f:
    lines = f.readlines()

n_cluster = 3
n_sample = len(lines)
X = np.array([[float(x) for x in l[:-2].split(' ')][:5] for l in lines])
assert len(X.shape) == 2

# Unit-test
mu_given, sigma_given = 2, 2
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
    print(w[0])
    print(mu[0,:])
    print(sigma[0,:])
    # E-step
    prob_normal = np.zeros((n_sample,n_cluster))
    for i in range(n_sample):
        for j in range(n_cluster):
            prob_normal[i][j] = w[j] * get_normal_pdf(X0[i,:], mu[j,:], sigma[j,:])
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

# comparing with sklearn 
# from sklearn.mixture import GaussianMixtur

# gm = GaussianMixture(n_components=2, random_state=0).fit(X)
# gm.means_
# gm.predict([[0, 0], [12, 3]])
