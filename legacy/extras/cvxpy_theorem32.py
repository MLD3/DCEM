import cvxpy as cp
import numpy as np

y_hat = cp.Variable(1)
t_hat = np.random.rand()
qy = np.random.rand()
inv_qy = 1 - qy
mstep_0 = -qy * cp.log(y_hat) - inv_qy * cp.log(1 - y_hat) - qy * cp.log(1 - y_hat * t_hat)

obj =  cp.Minimize(mstep_0)
prob = cp.Problem(obj, [y_hat >= 0, y_hat <= 1])

print("Params:", t_hat, qy)

prob.solve()

def eval_obj(y_hat):
    return -qy * np.log(y_hat) - inv_qy * np.log(1 - y_hat) - qy * np.log(1 - y_hat * t_hat)

b = 1 + 2 * qy * t_hat
d = b ** 2 - 4 * (qy * t_hat + t_hat) * qy
soln = (b - np.sqrt(d)) / (2 * (qy * t_hat + t_hat))

print("Solutions:", soln, y_hat.value.item())
analytic_obj = eval_obj(soln)
cvxpy_obj = eval_obj(y_hat.value.item())
print(analytic_obj, cvxpy_obj)
print("Rel error:", f"{(cvxpy_obj - analytic_obj) / analytic_obj * 100:.2f}%")
print("abs error:", abs(cvxpy_obj - analytic_obj))
