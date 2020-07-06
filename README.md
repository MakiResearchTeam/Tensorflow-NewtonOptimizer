# Tensorflow-NewtonOptimizer
This repository contains an implementation of the Newton optimization wrapped using Tensorflow.
The optimizer does not have all the arguments that other TensorFlow optimizers do (gating, distributed training), so it may be needed to modify the code a bit.

### Use it as follows:

`optimizer = NewtonOptimizer(...)`

`train_op = optimizer.minimize(loss)`

`sess.run(train_op)`

### Caching

The optimizer performs caching of the hessian in order to use it in multiple steps during optimization. This helps to accelerate the optimization procedure.
Use `hessian_period` argument to determine how often the hessian will be calculated.
If `hessian_period=1` the hessian will be calculated on each step. If `hessian_period=3` the hessian will be calculated after each 3rd step.

### On improving stability

Since in some points of the objective surface the inverse hessian's eigen values may become negative, it may lead to pure optimization result of even increase the value of the objective. In order to improve this issue an identity matrix is added to the hessian.
The 'final' hessian is computed as follows:

`hessian = hessian + alpha * eye`, where `eye` is the identity matrix. `alpha` is the argument of the optimizer's constructor.
You may try values as 0.1, 0.3, 0.6 and see which works for you. 

With `alpha=0.0` the optimizer does vanilla Newton optimization.

For more info on Newton's method check out Ian Goodfellow's DeepLearning Book (page 310).

### If the hessian is degenerate

In the case when the hessian's determinant is zero a gradient descent step is performed. You can regulate both the step size for Newton's updates and for gradient descent by setting appropriate values in the constructor.