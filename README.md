# CudaExpressions
Evaluate symbolic expressions on GPU using numba.cuda

The expression can be provided as string and the prameters as numpy array:
```
expression = 'H0*k1/(k2 + k3) - H1*k3/(k1 - k2) - k0'
R = np.random.uniform(0,1,(1000000,6))
```

After creating the expression object, the eval method can be applied on the parameter set. 
```
gpu_expr = GPUExpression(expression)
gpu_results = gpu_expr.eval(R)
```

Alternatively, a tensor representation can be created to use the expression as an inline function from within a kernel:
```
cuTensor = gpu_expr.toTensor(on_device=True)

@cuda.jit
def kernel(params,expression_tensor,eval_buffer,results):
    i = cuda.grid(1)
    if i < params.shape[0]:
         results[i] = ce.eval_inline(expression_tensor,params[i],eval_buffer[i])
```
