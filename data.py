import numpy as np
import geomstats.backend as gs
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.learning.pca import TangentPCA
from config import N_SAMPLES, KAPPA,PRECISION

def generate_data(n_samples=N_SAMPLES, kappa=KAPPA):
    sphere = Hypersphere(dim=2)
    random_mu = np.random.randn(3)
    random_mu /= np.linalg.norm(random_mu)
    data = sphere.random_von_mises_fisher(kappa=kappa, n_samples=n_samples, mu=gs.array(random_mu))
    return sphere, data

def generate_data_normal(n_samples=N_SAMPLES, precision=PRECISION):
    sphere = Hypersphere(dim=2)
    random_mu = np.random.randn(3)
    random_mu /= np.linalg.norm(random_mu)
    data = sphere.random_riemannian_normal(mean=random_mu,precision=precision, n_samples=n_samples)
    return sphere, data

def compute_frechet_mean(sphere, data):
    mean = FrechetMean(sphere)
    mean.fit(data)
    return mean.estimate_

def compute_tangent_pca(sphere, data, base_point):
    tpca = TangentPCA(sphere, n_components=2)
    tpca.fit(data, base_point=base_point)
    return tpca
