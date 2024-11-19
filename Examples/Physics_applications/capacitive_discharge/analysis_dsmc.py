#!/usr/bin/env python3

# 2023 TAE Technologies

import os
import sys

import numpy as np

sys.path.insert(1, "../../../../warpx/Regression/Checksum/")
from checksumAPI import evaluate_checksum

# fmt: off
ref_density = np.array([
    1.27942709e+14, 2.23579371e+14, 2.55384387e+14, 2.55660663e+14,
    2.55830911e+14, 2.55814337e+14, 2.55798906e+14, 2.55744891e+14,
    2.55915585e+14, 2.56083194e+14, 2.55942354e+14, 2.55833026e+14,
    2.56036175e+14, 2.56234141e+14, 2.56196179e+14, 2.56146141e+14,
    2.56168022e+14, 2.56216909e+14, 2.56119961e+14, 2.56065167e+14,
    2.56194764e+14, 2.56416398e+14, 2.56465239e+14, 2.56234337e+14,
    2.56234503e+14, 2.56316003e+14, 2.56175023e+14, 2.56030269e+14,
    2.56189301e+14, 2.56286379e+14, 2.56130396e+14, 2.56295225e+14,
    2.56474082e+14, 2.56340375e+14, 2.56350864e+14, 2.56462330e+14,
    2.56469391e+14, 2.56412726e+14, 2.56241788e+14, 2.56355650e+14,
    2.56650599e+14, 2.56674748e+14, 2.56642480e+14, 2.56823508e+14,
    2.57025029e+14, 2.57110614e+14, 2.57042364e+14, 2.56950884e+14,
    2.57051822e+14, 2.56952148e+14, 2.56684016e+14, 2.56481130e+14,
    2.56277073e+14, 2.56065774e+14, 2.56190033e+14, 2.56411074e+14,
    2.56202418e+14, 2.56128368e+14, 2.56227002e+14, 2.56083004e+14,
    2.56056768e+14, 2.56343831e+14, 2.56443659e+14, 2.56280541e+14,
    2.56191572e+14, 2.56147304e+14, 2.56342794e+14, 2.56735473e+14,
    2.56994680e+14, 2.56901500e+14, 2.56527131e+14, 2.56490824e+14,
    2.56614730e+14, 2.56382744e+14, 2.56588214e+14, 2.57160270e+14,
    2.57230435e+14, 2.57116530e+14, 2.57065771e+14, 2.57236507e+14,
    2.57112865e+14, 2.56540177e+14, 2.56416828e+14, 2.56648954e+14,
    2.56625594e+14, 2.56411003e+14, 2.56523754e+14, 2.56841108e+14,
    2.56856368e+14, 2.56757912e+14, 2.56895134e+14, 2.57144419e+14,
    2.57001944e+14, 2.56371759e+14, 2.56179404e+14, 2.56541905e+14,
    2.56715727e+14, 2.56851681e+14, 2.57114458e+14, 2.57001739e+14,
    2.56825690e+14, 2.56879682e+14, 2.56699673e+14, 2.56532841e+14,
    2.56479582e+14, 2.56630989e+14, 2.56885996e+14, 2.56694637e+14,
    2.56250819e+14, 2.56045278e+14, 2.56366075e+14, 2.56693733e+14,
    2.56618530e+14, 2.56580918e+14, 2.56812781e+14, 2.56754216e+14,
    2.56444736e+14, 2.56473391e+14, 2.56538398e+14, 2.56626551e+14,
    2.56471950e+14, 2.56274969e+14, 2.56489423e+14, 2.56645266e+14,
    2.56611124e+14, 2.56344324e+14, 2.56244156e+14, 2.24183727e+14,
    1.27909856e+14
])
# fmt: on

density_data = np.load("ion_density_case_1.npy")
print(repr(density_data))
assert np.allclose(density_data, ref_density)

# compare checksums
evaluate_checksum(
    test_name=os.path.split(os.getcwd())[1],
    output_file=sys.argv[1],
)
