# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 16:30:26 2023

@author: Lorenz Hexemer
"""
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from numba import cuda
import numpy as np

class GPUExpression:
    '''
    Creates a tensor representation of symbolic expressions that can be evaluated parallel on the GPU.
    '''
    operations = {sp.core.Number:1,sp.core.Symbol:2,sp.core.Add:3,sp.core.Mul:4,sp.core.Pow:5}

    def __init__(self,expression:str,params_order=None):
        '''
        Creates a tensor representation of symbolic expressions that can be evaluated parallel on the GPU.
        '''
        self.base_expr = parse_expr(expression).simplify()
        self.params = list(self.base_expr.atoms(sp.Symbol))
        if type(params_order) != type(None):
            self.setParameterOrder(params_order)

        self.command_chain = []
        self.parseSubexpr(self.base_expr)
        self.tensor = np.array(self.command_chain,dtype=np.int64)
        
    def toTensor(self,on_device=True):
        '''
        Returns the tensor representation.
        '''
        if on_device:
            return cuda.to_device(self.tensor)
        else:
            return self.tensor 

    def toNumpy(self,on_array=True):
        """
        Create a numpy-based function representing the expression.

        Args:
            on_arrax (bool):
                If True: expecting an array of shape n x d where d is the number of parameters
                If False: expecting an array of d parameters

        Return:
            function handle
        """
        numpy_single = sp.lambdify((self.params,),self.base_expr,'numpy')
        if not on_array:
            return numpy_single
        else:
            def numpy_array(in_array):
                return np.array([numpy_single(r) for r in in_array])
            return numpy_array
            
    def parseSubexpr(self,subexpr):
        """
        Internal functioN: Recursively used to parse the expression.
        """
        if isinstance(subexpr,sp.core.Number):
            if int(subexpr) != subexpr:
                raise ValueError('The expression contains non-intger values. These should be provided as parameters!')
            as_leaf = [1,int(subexpr),-1]
            if not as_leaf in self.command_chain:
                self.command_chain.append(as_leaf)
            return self.command_chain.index(as_leaf)
        
        elif isinstance(subexpr,sp.core.Symbol):
            as_leaf = [2,self.params.index(subexpr),-1]
            if not as_leaf in self.command_chain:
                self.command_chain.append(as_leaf)
            return self.command_chain.index(as_leaf)
        
        else:
            if len(subexpr.args) == 2:
                a,b = subexpr.args
                id_a = self.parseSubexpr(a)
                id_b = self.parseSubexpr(b)
            else:
                a,*b = subexpr.args
                id_a = self.parseSubexpr(a)
                new_b = subexpr.func(*b)
                id_b = self.parseSubexpr(new_b)
                
            self.command_chain.append([self.operations[subexpr.func],id_a,id_b])
            return len(self.command_chain)-1
        
    def setParameterOrder(self,parameters:list, allow_unused=True):
        """
        Define the order in which parameters are provided to the function.

        Args:
            parameters (list of symbols): Symbols used in the expression in the right order.
        """
        if not allow_unused:
            unexpected_params = [p for p in parameters if p not in self.params]
            if unexpected_params:
                unexpected = ', '.join([str(p) for p in unexpected_params])
                raise Exception('Parameter(s) %s were not defined in the expression!'%unexpected)
        
        missing_params = [p for p in self.params if p not in parameters] 
        if missing_params:
            missing = ', '.join([str(p) for p in missing_params])
            raise Exception('Parameter(s) %s were not listed in the provided ordering!'%missing)
        
        self.params = parameters

    def eval(self,params,threadsperblock=1024):
        """
        Evaluate the expression on the GPU for a collection of parameters.
        
        Args:
            params (n x p array): numpy array of parameters.

            threadsperblock: Numberof threads used.
        """
        cuR = cuda.to_device(params)
        cuTensor = self.toTensor(on_device=True)

        buffer = np.zeros((params.shape[0],cuTensor.shape[0]))
        cuBuffer = cuda.to_device(buffer)
        results = np.zeros(params.shape[0])
        cuResults = cuda.to_device(results)

        blockspergrid = (params.shape[0] + (threadsperblock - 1)) // threadsperblock
        expression_kernel[blockspergrid,threadsperblock](cuR,cuTensor,cuBuffer,cuResults)
        cuda.synchronize()
        results = cuResults.copy_to_host()
        return results

class GPUExpressionVector(GPUExpression):
    def __init__(self,expression_list:list,params_order=None):
        self.out_dim = len(expression_list)
        self.base_expr = sp.Matrix(expression_list)

        self.params = list(self.base_expr.atoms(sp.Symbol))
        if type(params_order) != type(None):
            self.setParameterOrder(params_order)

        self.parseExpressions(self.base_expr[:])
        #self.shortenCommandChains()
        self.tensor = np.array(self.command_chain,dtype=np.int64)

    def parseExpressions(self,expression_list):
        self.command_chain = []
        for k,expr in enumerate(expression_list):
            if expr != 0:
                gpu_expr = GPUExpression(str(expr),params_order=self.params)
                # command_chain imported from GPUExpression objects assume starting at 0 => shift
                self.command_chain += self._shiftBufferRef(gpu_expr.command_chain,len(self.command_chain))
                self.command_chain.append([6,len(self.command_chain)-1,k])

    def _shiftBufferRef(self,command_chain,offset):
        """
        When multiple command_chains are concattenated, the indices from later expressions don't start with 0.
        This function shifts the the refferences (buffer positions) to the absolute position. 
        """
        for k,(com,*_) in enumerate(command_chain):
            if com not in [1,2]:
                command_chain[k][1] += offset 
                command_chain[k][2] += offset
            if com == 6:
                command_chain[k][1] += offset

        return command_chain

    def _shortenCommandChains(self):
        """
        In several components the same parameters or constants might be loaded to the buffer
          or the same calculation step might be applied. This function seaches for duplicates and 
          changes the refferences to the first element in the chain containing the same statement.
        """
        second,first = self._findCommandDuplicates(0)
        while second > 0:
            self.command_chain.pop(second)
            self._redirectValueLink(second,first)
            second,first = self._findCommandDuplicates(second)

    def _findCommandDuplicates(self,offset):
        """
        In several components the same parameters or constants might be loaded to the buffer
          or the same calculation step might be applied. This function finds theses duplicates 
          and returns the respective inices.

        """
        for k,line in enumerate(self.command_chain[offset:]):
            if line in self.command_chain[:offset+k]:
                first = self.command_chain.index(line)
                return offset+k,first
        return -1,-1

    def _redirectValueLink(self,old,new):
        """
        When duplicates are removed from the command_chain, all references to the deleted duplicate must be 
        redirected to the first instance of the same command.
        """
        #self.command_chain.pop(old)
        for k,(com,a,b) in enumerate(self.command_chain):
            # dont 'shift' values / parameters
            if com in [1,2]:
                continue
            if a==old:
                self.command_chain[k][1] = new
            if b==old:
                self.command_chain[k][2] = new

    def eval(self,params,threadsperblock=1024):
        """
        Evaluate the expression on the GPU for a collection of parameters.
        
        Args:
            params (n x p array): numpy array of parameters.

            threadsperblock: Numberof threads used.
        """
        cuR = cuda.to_device(params)
        cuTensor = self.toTensor(on_device=True)

        buffer = np.zeros((params.shape[0],cuTensor.shape[0]))
        cuBuffer = cuda.to_device(buffer)
        results = np.zeros((params.shape[0],self.out_dim))
        cuResults = cuda.to_device(results)

        blockspergrid = (params.shape[0] + (threadsperblock - 1)) // threadsperblock
        vector_kernel[blockspergrid,threadsperblock](cuR,cuTensor,cuBuffer,cuResults)
        cuda.synchronize()
        results = cuResults.copy_to_host()
        return results
        
@cuda.jit(device=True)
def eval_inline(expression_tensor,parameters,buffer):
    """
    Evaluate the symbolic expression represented by expression_tensor using the defined parameters.

    Arugs:
        expression_tensor (expression tensor on device): 
            Tensor representation of the expression generated by GPUExpression.toTensor().

        parameters (n x p array on device):
            Matrix containing the parameters for each case.

        buffer: (array of length n on device): 
            Empty array which is filled with the results by the kernel.
    """
    for k,(com,a,b) in enumerate(expression_tensor):
        if com == 1:
            buffer[k] = expression_tensor[k,1]
        elif com == 2:
            buffer[k] = parameters[int(a)]
        elif com == 3:
            buffer[k] = buffer[int(a)] + buffer[int(b)]
        elif com == 4:
            buffer[k] = buffer[int(a)]*buffer[int(b)]  
        elif com == 5:
            buffer[k] = buffer[int(a)]**buffer[int(b)]

    return buffer[-1]

@cuda.jit
def expression_kernel(params,expression_tensor,eval_buffer,results):
    """
    Generic kernel calling the inline function. This function is used by the eval() method of the GPUExpression class.
    """
    i = cuda.grid(1)
    if i < params.shape[0]:
         results[i] = eval_inline(expression_tensor,params[i],eval_buffer[i])

@cuda.jit(device=True)
def eval_vector_inline(expression_tensor,parameters,buffer,result_vector):
    """
    Evaluate the symbolic expression represented by expression_tensor using the defined parameters.

    Arugs:
        expression_tensor (expression tensor on device): 
            Tensor representation of the expression generated by GPUExpression.toTensor().

        parameters (n x p array on device):
            Matrix containing the parameters for each case.

        buffer (n x array on device):
            Matrix containing intermediate results.

        result_vector: (array of length n on device): 
            Empty array which is filled with the results by the kernel.
    """

    for k,(com,a,b) in enumerate(expression_tensor):
        if com == 1:
            buffer[k] = expression_tensor[k,1]
        elif com == 2:
            buffer[k] = parameters[int(a)]
        elif com == 3:
            buffer[k] = buffer[int(a)] + buffer[int(b)]
        elif com == 4:
            buffer[k] = buffer[int(a)]*buffer[int(b)]  
        elif com == 5:
            buffer[k] = buffer[int(a)]**buffer[int(b)]
        elif com == 6:
            result_vector[int(b)] = buffer[int(a)]
        
@cuda.jit
def vector_kernel(params,expression_tensor,eval_buffer,results):
    """
    Generic kernel calling the inline function. This function is used by the eval() method of the GPUExpression class.
    """
    i = cuda.grid(1)
    if i < params.shape[0]:
        eval_vector_inline(expression_tensor,params[i],eval_buffer[i],results[i])