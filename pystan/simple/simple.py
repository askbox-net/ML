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

model_code = """
data {
    int     N;
    real    x[N];
}

parameters {
    real            mu;
    real<lower=0>   sigma;
}

model {
    x ~ normal(mu, sigma);
}

generated quantities{
    real xaste;
    real log_lik;
    xaste = normal_rng(mu,sigma);
    log_lik = normal_lpdf(x|mu,sigma);
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
    np.random.seed(42)
    N = 100
    mu = 10.5
    sigma = 5.5
    x = np.random.normal(mu, sigma , N).tolist()
    print("mean = %0.3f, std = %0.3f" % (np.mean(x), np.std(x)))

    model = model_cache(model_code)
    fit = model.sampling(data={ 'x': x, 'N': len(x) }, chains=5, iter=21000, warmup=1000, thin=1)
    pdata = fit.extract()

    fit.plot()
    plt.savefig("%s/%s.png" % (dirname(__file__), basename(__file__)), dpi=90)
    plt.show()
    sigma = pdata['sigma']
    mu = pdata['mu']

    var = sigma**2 #分散(variance)
    cv = sigma / mu #変動係数(coefficient of variation)

    prob = np.array([0.025, 0.05, 0.5, 0.95, 0.975])*100

    desc = [ "", "EAP", "post.sd" ] + prob.tolist()
    varinfo = [ "分散　　", np.mean(var), np.std(var) ] + np.percentile(var, prob).tolist()
    cvinfo = [ "変動係数", np.mean(cv), np.std(cv) ] + np.percentile(cv, prob).tolist()
    fmt = "{:^4}" + "{:>10.2f}" * (len(varinfo) - 1)
    print("")
    print(("{:^8}{:>10}{:>10}" + "{:>9.1f}%"*(len(varinfo)-3)).format(*desc))
    print(fmt.format(*varinfo))
    print(fmt.format(*cvinfo))
