# -*- coding:utf-8 -*-

import pickle
import hashlib
import os
import pystan
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os.path import basename, dirname
import numpy as np
from scipy import stats
import seaborn as sns

model_code = """
data {
    int     N;
    int     M;
    real    x[N];
}

parameters {
    vector[M]           mu;
    vector<lower=0>[M]  sigma;
    simplex[M]          PI;
}

model {
    real ps[M];
    for(n in 1:N){
        for(m in 1:M){
            ps[m] = log(PI[m]) + normal_lpdf(x[n] | mu[m], sigma[m]);
        }
        target += log_sum_exp(ps);
    }
}
"""

"""
data {
  int N;
  vector[N] Y;
}

parameters {
  real<lower=0, upper=1> a;
  ordered[2] mu;
  vector<lower=0>[2] sigma;
}

model {
  for (n in 1:N)
    target += log_sum_exp(
      log(a)   + normal_lpdf(Y[n] | mu[1], sigma[1]),
      log1m(a) + normal_lpdf(Y[n] | mu[2], sigma[2])
    );
}
"""

def model_cache(model_code, path='./tmp'):
    if not os.path.exists(path):
        os.makedirs(path)

    hashname = hashlib.md5(model_code.encode('utf-8')).hexdigest()
    filename = '%s/%s.pkl' % (path, hashname) 

    if os.path.exists(filename):
        with open(filename, 'rb') as fp:
            model = pickle.load(fp)
    else:
        model = pystan.StanModel(model_code=model_code)
        with open(filename, 'wb') as fp:
            pickle.dump(model, fp)

    return model

if __name__ == '__main__':
    #サンプル数
    N = 4200
    #混合数
    K = 2
    #混合係数
    PI = 0.7
    #乱数の種
    np.random.seed(42)

    #混合係数から各分布のサンプリング数を決める
    N_k1 = int(N * PI)
    N_k2 = N - N_k1

    #真のパラメータ
    mu_1 = -5
    sigma_1 = np.sqrt(25)
    mu_2 = 5
    sigma_2 = np.sqrt(1)

    x1 = np.random.normal(mu_1,sigma_1,N_k1)
    x2 = np.random.normal(mu_2,sigma_2,N_k2)

    #観測変数
    x = np.hstack((x1,x2))
    base=np.linspace(np.min(x),np.max(x),1000)
    plt.hist(x,bins=100)
    plt.plot(base,PI*stats.norm.pdf(base,mu_1,sigma_1))
    plt.plot(base,(1-PI)*stats.norm.pdf(base,mu_2,sigma_2))
    plt.plot(base,PI*stats.norm.pdf(base,mu_1,sigma_1)+(1-PI)*stats.norm.pdf(base,mu_2,sigma_2))
    plt.savefig("%s/%s.png" % (dirname(__file__), basename(__file__)), dpi=90)

    #Stan
    stan_data = {'N': N, 'M': K, 'x': x}

    model = model_cache(model_code=model_code)

    fitchan = model.optimizing(data=stan_data)
    #fitchan = model.sampling(data=stan_data, iter=20000)
    #pdata = fitchan.extract()
    print(fitchan)
    #print(pdata)

