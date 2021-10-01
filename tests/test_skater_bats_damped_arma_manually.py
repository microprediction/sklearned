import os
from timemachines.skaters.localskaters import local_skater_from_name
from timemachines.skating import prior
import numpy as np
from timemachines.skaters.bats.batsinclusion import using_bats

if __name__=='__main__':
    import tbats
    skater_name = __file__.split(os.path.sep)[-1].replace('test_skater_', '').replace('.py', '')
    print(skater_name)
    f = local_skater_from_name(skater_name)
    y = np.random.randn(100)
    prior(f=f, y=y, k=1)


