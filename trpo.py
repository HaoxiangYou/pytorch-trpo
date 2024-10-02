import numpy as np

import torch
from torch.autograd import Variable
from utils import *


def conjugate_gradients(Avp, b, nsteps, residual_tol=1e-10):
    x = torch.zeros(b.size())
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        _Avp = Avp(p)
        alpha = rdotr / torch.dot(p, _Avp)
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x


def linesearch(model,
               f,
               compute_kl,
               x,
               fullstep,
               expected_improve_rate,
               max_kl,
               max_backtracks=10,
               accept_ratio=.1):
    fval = f(True).data
    print("fval before", fval.item())
    for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        set_flat_params_to(model, xnew)
        newfval = f(True).data
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        print("a/e/r", actual_improve.item(), expected_improve.item(), ratio.item())
        print("average kl/kl tolerance", compute_kl().data, max_kl)
        if ratio.item() > accept_ratio and actual_improve.item() > 0:
            print("fval after", newfval.item())
            return True, xnew
    return False, x


def trpo_step(model, get_loss, get_kl, compute_kl, step_size, damping, is_trpo):
    loss = get_loss()
    grads = torch.autograd.grad(loss, model.parameters())
    loss_grad = torch.cat([grad.view(-1) for grad in grads]).data

    def Fvp(v):
        kl = get_kl()
        kl = kl.mean()

        grads = torch.autograd.grad(kl, model.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        kl_v = (flat_grad_kl * Variable(v)).sum()
        grads = torch.autograd.grad(kl_v, model.parameters())
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data

        # v * damping is adding small identity to fisher information matrix
        return flat_grad_grad_kl + v * damping
    
    stepdir = conjugate_gradients(Fvp, -loss_grad, 10)

    prev_params = get_flat_params_from(model)

    if is_trpo:
        shs = 0.5 * (stepdir * Fvp(stepdir)).sum(0, keepdim=True)
        lm = torch.sqrt(shs / step_size)
        fullstep = stepdir / lm[0]
        neggdotstepdir = (-loss_grad * stepdir).sum(0, keepdim=True)
        print(("lagrange multiplier:", lm[0], "grad_norm:", loss_grad.norm()), "fullstep_norm:", fullstep.norm())
        success, new_params = linesearch(model, get_loss, compute_kl, prev_params, fullstep,
                                        neggdotstepdir / lm[0], step_size)
    else:
        new_params = prev_params + step_size * stepdir
        print(("lagrange multiplier:", 1/step_size, "grad_norm:", loss_grad.norm()))

    set_flat_params_to(model, new_params)

    return loss
