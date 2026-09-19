import numpy as np
import pytest
from model import X,Y,accuracy,forward,initialize,loss_and_gradient,predict_proba,sigmoid
from train import train
from check_gradients import check_gradient


@pytest.mark.parametrize('seed',[0,1,13])
@pytest.mark.parametrize('scale',[.2,1.,3.])
def test_backprop_against_finite_difference(seed,scale):
    report=check_gradient(initialize(seed)*scale)
    assert report['maximum_absolute_error']<1e-7
    assert report['relative_l2_error']<1e-5


def test_zero_initialization_is_an_exact_stationary_trap():
    loss,gradient=loss_and_gradient(np.zeros(9))
    assert abs(loss-np.log(2))<1e-15
    np.testing.assert_array_equal(gradient,np.zeros(9))
    result=train(initial=np.zeros(9))
    assert not result.success and result.reason=='stationary_before_target'
    assert result.accuracy==.5


def test_large_logits_do_not_overflow_cross_entropy():
    theta=initialize(0);theta[8]=1000
    loss,gradient=loss_and_gradient(theta)
    assert np.isfinite(loss) and np.isfinite(gradient).all()
    np.testing.assert_array_equal(sigmoid([-1000,1000]),[0.,1.])


def test_hidden_unit_permutation_preserves_function():
    theta=initialize(13);permuted=theta.copy()
    permuted[:4]=theta[:4].reshape(2,2)[:,::-1].ravel()
    permuted[4:6]=theta[4:6][::-1];permuted[6:8]=theta[6:8][::-1]
    np.testing.assert_allclose(predict_proba(theta),predict_proba(permuted),atol=1e-15)


def test_adam_reaches_declared_target_on_predeclared_seed():
    result=train('adam',seed=0)
    assert result.success and result.accuracy==1 and result.loss<=.02
    assert check_gradient(result.theta)['relative_l2_error']<1e-5


@pytest.mark.parametrize('bad',[np.zeros(8),np.full(9,np.nan)])
def test_invalid_parameters_rejected(bad):
    with pytest.raises(ValueError):forward(bad)
