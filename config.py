
N_SAMPLES = 100
import geomstats.backend as gs

KAPPA = 15
PRECISION = gs.array([[20, 11], [1, 20]])
SCALING_FACTOR_PC1 = 0.5
SCALING_FACTOR_PC2 = 0.25

FIG_WIDTH = 800
FIG_HEIGHT = 800

COLORS = {
    "sphere": "Blues",
    "data": "black",
    "frechet_mean": "red",
    "tangent_plane": "gray",
    "pc1": "red",
    "pc2": "green",
    "reconstructed": "orange",
    "full_reconstruction": "blue",
}

MARKER_SIZES = {
    "data": 4,
    "frechet_mean": 10,
    "reconstructed": 4,
    "full_reconstruction": 4,
}

LINE_WIDTHS = {
    "pc1": 5,
    "pc2": 5,
}

OPACITIES = {
    "sphere": 0.3,
    "tangent_plane": 0.5,
}
