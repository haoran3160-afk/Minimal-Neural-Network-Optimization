"""Reproduce XOR optimizer comparisons and explicitly labeled failure cases."""
from pathlib import Path
import csv
import json
import platform
from time import perf_counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from .model import X,Y,accuracy,forward,initialize,loss_and_gradient,predict_proba
from .train import DEFAULT_LEARNING_RATES,train
from .check_gradients import check_gradient

from .paths import PROJECT_ROOT as ROOT, result_path, prepare_results


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def run(output=None,seeds=range(20)):
    output=Path(output) if output else ROOT/'results';prepare_results(output)
    grad_reports=[]
    for seed in [0,1,13]:
        for scale in [.2,1.,3.]:
            report=check_gradient(initialize(seed)*scale)
            assert report['maximum_absolute_error']<1e-7 and report['relative_l2_error']<1e-5
            grad_reports.append({'seed':seed,'scale':scale,**report})
    (result_path('gradient_checks.json', output)).write_text(json.dumps(grad_reports,indent=2),encoding='utf-8')
    rows=[];main_results={}
    for seed in seeds:
        initial=initialize(seed)
        for method in ['gd','momentum','adam']:
            start=perf_counter();result=train(method,initial=initial);elapsed=perf_counter()-start
            rows.append(dict(seed=seed,method=method,learning_rate=DEFAULT_LEARNING_RATES[method],
                             loss=result.loss,accuracy=result.accuracy,iterations=result.iterations,
                             success=result.success,reason=result.reason,elapsed_seconds=elapsed))
            if seed==0:main_results[method]=result
        print(f'XOR seed {seed}: complete',flush=True)
    write_csv(result_path('optimizer_runs.csv', output),rows)
    summary=[]
    for method in ['gd','momentum','adam']:
        group=[r for r in rows if r['method']==method];ok=[r for r in group if r['success']]
        summary.append(dict(method=method,seeds=len(group),successes=len(ok),failures=len(group)-len(ok),
                            mean_final_loss=float(np.mean([r['loss'] for r in group])),
                            median_iterations_successes=float(np.median([r['iterations'] for r in ok])) if ok else None,
                            mean_accuracy=float(np.mean([r['accuracy'] for r in group]))))
    write_csv(result_path('optimizer_summary.csv', output),summary)
    # SciPy is an independent optimization reference, not the training implementation.
    reference=minimize(lambda theta:loss_and_gradient(theta),initialize(0),jac=True,method='BFGS',
                       options={'gtol':1e-8,'maxiter':5000})
    ref_loss,ref_grad=loss_and_gradient(reference.x)
    assert accuracy(reference.x)==1. and ref_loss<.02
    reference_info={'success':bool(reference.success),'message':str(reference.message),
                    'loss':ref_loss,'gradient_norm':float(np.linalg.norm(ref_grad)),
                    'accuracy':accuracy(reference.x),'iterations':int(reference.nit)}
    # Predeclared stress cases. They are not hidden in the 20 ordinary starts.
    zero=train('adam',initial=np.zeros(9))
    saturated_initial=initialize(0);saturated_initial[:6]*=100
    saturated=train('adam',initial=saturated_initial,max_iterations=3000)
    high_rate=train('gd',initial=initialize(0),learning_rate=50,max_iterations=3000)
    stress=[dict(case=name,success=r.success,loss=r.loss,accuracy=r.accuracy,
                 iterations=r.iterations,reason=r.reason) for name,r in
            [('all_zero',zero),('saturated_hidden_x100',saturated),('gd_learning_rate_50',high_rate)]]
    write_csv(result_path('stress_cases.csv', output),stress)
    chosen=main_results['adam'];assert chosen.success
    trained_gradcheck=check_gradient(chosen.theta)
    assert trained_gradcheck['maximum_absolute_error']<1e-7
    assert trained_gradcheck['relative_l2_error']<1e-5
    grad_reports.append({'seed':0,'scale':1.,'state':'trained_adam',**trained_gradcheck})
    (result_path('gradient_checks.json', output)).write_text(json.dumps(grad_reports,indent=2),encoding='utf-8')
    np.savez_compressed(result_path('trained_model.npz', output),theta=chosen.theta,initial=initialize(0))
    predictions=[dict(x1=x[0],x2=x[1],target=int(y),probability=float(p),prediction=int(p>=.5))
                 for x,y,p in zip(X,Y,predict_proba(chosen.theta))]
    write_csv(result_path('predictions.csv', output),predictions)
    np.savez_compressed(result_path('training_histories.npz', output),**{name:r.history for name,r in main_results.items()},
                        all_zero=zero.history,saturated=saturated.history,high_rate=high_rate.history)

    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for method,result in main_results.items():
        axes[0].semilogy(result.history[:,0],result.history[:,1],label=method)
        axes[1].semilogy(result.history[:,0],np.maximum(result.history[:,2],1e-16),label=method)
    axes[0].axhline(.02,color='grey',ls=':');axes[0].set(xlabel='Full-batch iteration',ylabel='Mean binary cross-entropy',title='Same initial parameters, seed 0')
    axes[1].set(xlabel='Full-batch iteration',ylabel='Gradient norm',title='Hand-written backprop')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.savefig(result_path('training_curves.png', output),dpi=170);plt.close(fig)
    grid=np.linspace(-1.6,1.6,160);xx,yy=np.meshgrid(grid,grid)
    probability=predict_proba(chosen.theta,np.column_stack([xx.ravel(),yy.ravel()])).reshape(xx.shape)
    fig,ax=plt.subplots(figsize=(5.5,4.8),constrained_layout=True)
    contour=ax.contourf(xx,yy,probability,levels=np.linspace(0,1,21),cmap='RdBu_r',vmin=0,vmax=1)
    ax.contour(xx,yy,probability,levels=[.5],colors='black',linewidths=1)
    ax.scatter(X[:,0],X[:,1],c=Y,cmap='RdBu_r',vmin=0,vmax=1,s=100,edgecolors='black')
    fig.colorbar(contour,ax=ax,label='Predicted P(y=1)')
    ax.set(xlabel='x1',ylabel='x2',title='2-2-1 network | 9 parameters | Adam seed 0')
    fig.savefig(result_path('decision_boundary.png', output),dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
    for name,r in [('All-zero init',zero),('Saturated hidden',saturated),('GD rate=50',high_rate)]:
        ax.plot(r.history[:,0],r.history[:,1],label=name,marker='o' if len(r.history)==1 else None)
    ax.set(xlabel='Iteration',ylabel='Mean binary cross-entropy',title='Explicit stress cases; unsuccessful runs retained')
    ax.legend();ax.grid(alpha=.2);fig.savefig(result_path('stress_curves.png', output),dpi=170);plt.close(fig)
    info={'architecture':[2,2,1],'parameter_count':9,'primary_seed':0,
          'primary_method':'adam','primary_loss':chosen.loss,'primary_accuracy':chosen.accuracy,
          'primary_iterations':chosen.iterations,'primary_theta':chosen.theta.tolist(),
          'target_loss':.02,'iteration_cap':10000,'optimizer_summary':summary,
          'gradient_max_absolute_error':max(r['maximum_absolute_error'] for r in grad_reports),
          'gradient_max_relative_error':max(r['relative_l2_error'] for r in grad_reports),
          'reference_bfgs':reference_info,'stress_cases':stress,'python':platform.python_version(),
          'numpy':np.__version__,
          'limits':'Four XOR truth-table points; no held-out generalization claim; finite loss target, not proof of finite global minimizer.'}
    (result_path('summary.json', output)).write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(json.dumps({'primary_loss':chosen.loss,'primary_accuracy':chosen.accuracy,
                      'optimizer_summary':summary},indent=2))
    return info


if __name__=='__main__':run()
