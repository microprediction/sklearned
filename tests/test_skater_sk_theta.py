import os
from timemachines.skaters.localskaters import local_skater_from_name
from timemachines.skating import prior
import numpy as np

if __name__=='__main__':
    from timemachines.skaters.sk.skinclusion import using_sktime
    assert using_sktime
    skater_name = __file__.split(os.path.sep)[-1].replace('test_skater_', '').replace('.py', '')
    print(skater_name)
    f = local_skater_from_name(skater_name)
    assert f is not None
    y = np.random.randn(100)
    prior(f=f, y=y, k=1)


