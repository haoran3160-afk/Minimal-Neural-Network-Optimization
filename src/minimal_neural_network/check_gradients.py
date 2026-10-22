"""Central finite differences, independent of the backprop implementation."""
import numpy as np
from .model import PARAMETER_NAMES, initialize, loss_and_gradient


def check_gradient(theta,step=1e-5):
    _,analytic=loss_and_gradient(theta)
    numeric=np.zeros(9)
    for j in range(9):
        offset=np.zeros(9);offset[j]=step
        numeric[j]=(loss_and_gradient(theta+offset)[0]-loss_and_gradient(theta-offset)[0])/(2*step)
    absolute=np.abs(analytic-numeric)
    relative=float(np.linalg.norm(analytic-numeric)/max(1e-10,np.linalg.norm(analytic)+np.linalg.norm(numeric)))
    return {'relative_l2_error':relative,'maximum_absolute_error':float(absolute.max()),
            'components':[{'parameter':name,'analytic':float(a),'numeric':float(n),'absolute_error':float(e)}
                          for name,a,n,e in zip(PARAMETER_NAMES,analytic,numeric,absolute)]}


if __name__=='__main__':
    for seed in [0,1,13]:
        for scale in [.2,1.,3.]:
            result=check_gradient(initialize(seed)*scale)
            assert result['maximum_absolute_error']<1e-7
            assert result['relative_l2_error']<1e-5
            print(f'seed={seed}, scale={scale:g}: PASS, relative error={result["relative_l2_error"]:.3g}')
