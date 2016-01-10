from abc import ABCMeta, abstractmethod

import numpy as np

# Ignore division by zero which will happen CrossEntropyLoss.fprop()
# when Softmax is not included at the end layer.
np.seterr(divide='ignore')


class Module(object):
    """
    Abstract Module class for neural net
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        self.output     = None
        self.grad_input = None

    @abstractmethod
    def fprop(self, input_data):
        self.output = input_data
        return self.output

    @abstractmethod
    def bprop(self, input_data, grad_output):
        self.grad_input = grad_output
        return self.grad_input

    @abstractmethod
    def update(self, params):
        pass

    @abstractmethod
    def share(self, m):
        pass


class Container(Module):
    """
    Container
    """
    def __init__(self):
        super(Container, self).__init__()
        self.modules = []

    def add(self, m):
        self.modules.append(m)

    def update(self, params):
        for module in self.modules:
            module.update(params)

    def share(self, m):
        for c_module, m_module in zip(self.modules, m.modules):
            c_module.share(m_module)


class Sequential(Container):

    def fprop(self, input_data):
        temp = input_data
        for module in self.modules:
            temp = module.fprop(temp)

        self.output = temp
        return self.output

    def bprop(self, input_data, grad_output):
        for i in range(len(self.modules) - 1, 0, -1):
            grad_input = self.modules[i].bprop(self.modules[i - 1].output, grad_output)
            grad_output = grad_input
        grad_input = self.modules[0].bprop(input_data, grad_output)

        self.grad_input = grad_input
        return self.grad_input


class Parallel(Container):
    """
    Computes forward and backward propagations for all modules at once.
    """
    def fprop(self, input_data):
        self.output = [module.fprop(input_elem)
                       for module, input_elem in zip(self.modules, input_data)]
        return self.output

    def bprop(self, input_data, grad_output):
        self.grad_input = [module.bprop(input_elem, grad_output_elem)
                           for module, input_elem, grad_output_elem
                           in zip(self.modules, input_data, grad_output)]
        return self.grad_input


class AddTable(Module):
    """
    Module for sum operator which sums up all elements in input data
    """
    def __init__(self):
        super(AddTable, self).__init__()

    def fprop(self, input_data):
        self.output = input_data[0]
        for elem in input_data[1:]:
            # Expand to the same ndim as self.output
            # TODO: Code improvement
            if elem.ndim == self.output.ndim - 1:
                elem = np.expand_dims(elem, axis=elem.ndim + 1)
            self.output += elem
        return self.output

    def bprop(self, input_data, grad_output):
        self.grad_input = [grad_output for _ in range(len(input_data))]
        return self.grad_input

    def share(self, m):
        pass

    def update(self, params):
        pass


class ConstMult(Module):
    """
    Module for multiplying with a constant
    """
    def __init__(self, c):
        super(ConstMult, self).__init__()
        self.c = c

    def fprop(self, input_data):
        self.output = self.c * input_data
        return self.output

    def bprop(self, input_data, grad_output):
        self.grad_input = self.c * grad_output
        return self.grad_input

    def share(self, m):
        pass

    def update(self, params):
        pass


class Duplicate(Module):
    """
    Duplicate module which essentially makes a clone for input data
    """
    def __init__(self):
        super(Duplicate, self).__init__()

    def fprop(self, input_data):
        self.output = [input_data, input_data]
        return self.output

    def bprop(self, input_data, grad_output):
        self.grad_input = grad_output[0] + grad_output[1]
        return self.grad_input

    def share(self, m):
        pass

    def update(self, params):
        pass


class ElemMult(Module):
    """
    Module for element-wise product
    """
    def __init__(self, weight):
        super(ElemMult, self).__init__()
        self.weight = weight

    def fprop(self, input_data):
        # TODO: Rewrite these checkings!!!
        if input_data.ndim == 2:
            self.output = input_data * self.weight
        elif input_data.ndim == 3:
            self.output = input_data * self.weight[:, :, None]  # broadcasting
        elif input_data.ndim == 4:
            self.output = input_data * self.weight[:, :, None, None]  # broadcasting
        else:
            raise Exception("input_data has large dimension = %d" % input_data.ndim)
        return self.output

    def bprop(self, input_data, grad_output):
        # TODO: Same as above.
        if input_data.ndim == 2:
            self.grad_input = grad_output * self.weight
        elif input_data.ndim == 3:
            self.grad_input = grad_output * self.weight[:, :, None]  # broadcasting
        elif input_data.ndim == 4:
            self.grad_input = grad_output * self.weight[:, :, None, None]  # broadcasting
        else:
            raise Exception("input_data has large dimension = %d" % input_data.ndim)
        return self.grad_input

    def share(self, m):
        pass

    def update(self, params):
        pass


class Identity(Module):
    """
    Identical forward and backward propagations
    """
    def __init__(self):
        super(Identity, self).__init__()

    def fprop(self, input_data):
        self.output = input_data
        return self.output

    def bprop(self, input_data, grad_output):
        self.grad_input = grad_output
        return self.grad_input

    def share(self, m):
        pass

    def update(self, params):
        pass


class Linear(Module):
    """
    Linear Layer
    """
    def __init__(self, in_dim, out_dim):
        super(Linear, self).__init__()
        self.in_dim  = in_dim
        self.out_dim = out_dim
        self.weight  = Weight((out_dim, in_dim))
        self.bias    = Weight((out_dim, 1))

    def fprop(self, input_data):
        high_dimension_input = input_data.ndim > 2

        # Reshape input
        if high_dimension_input:
            input_data = input_data.reshape(input_data.shape[0], -1)

        self.output = np.dot(self.weight.D, input_data) + self.bias.D

        # Reshape output
        if high_dimension_input:
            self.output = self.output.reshape(self.output.shape[0], -1)

        return self.out_dim

    def bprop(self, input_data, grad_output):
        orig_input_data_shape = input_data.shape
        high_dimension_input = input_data.ndim > 2

        # Reshape input and grad_output
        if high_dimension_input:
            input_data  = input_data.reshape(input_data.shape[0], -1)
            grad_output = grad_output.reshape(grad_output.shape[0], -1)

        self.weight.grad = self.weight.grad + np.dot(grad_output, input_data.T)
        self.bias.grad   = self.bias.grad + grad_output.sum(axis=1)
        self.grad_input  = np.dot(self.weight.D.T, grad_output)

        if high_dimension_input:
            self.grad_input = self.grad_input.reshape(orig_input_data_shape)

        return self.grad_input

    def update(self, params):
        self.weight.update(params)
        self.bias.update(params)

    def share(self, m):
        self.weight = m.weight
        self.bias = m.bias


class LinearNB(Module):
    """
    Linear layer with no bias
    """
    def __init__(self, in_dim, out_dim, do_transpose=False):
        super(LinearNB, self).__init__()
        self.in_dim       = in_dim
        self.out_dim      = out_dim
        self.do_transpose = do_transpose

        if do_transpose:
            self.weight = Weight((in_dim, out_dim))
        else:
            self.weight = Weight((out_dim, in_dim))

    def fprop(self, input_data):
        high_dimension_input = input_data.ndim > 2

        if high_dimension_input:
            input_data = input_data.reshape(input_data.shape[0], -1)

        if self.do_transpose:
            self.output = np.dot(self.weight.D.T, input_data)
        else:
            self.output = np.dot(self.weight.D, input_data)

        if high_dimension_input:
            self.output = self.output.reshape(self.output.shape[0], -1)

        return self.output

    def bprop(self, input_data, grad_output):
        orig_input_data_shape = input_data.shape
        high_dimension_input = input_data.ndim > 2

        # Reshape input and grad_output
        if high_dimension_input:
            input_data = input_data.reshape(input_data.shape[0], -1)
            grad_output = grad_output.reshape(grad_output.shape[0], -1)

        if self.do_transpose:
            self.weight.grad = self.weight.grad + np.dot(input_data, grad_output.T)
            self.grad_input  = np.dot(self.weight.D, grad_output)
        else:
            self.weight.grad = self.weight.grad + np.dot(grad_output, input_data.T)
            self.grad_input  = np.dot(self.weight.D.T, grad_output)

        if high_dimension_input:
            self.grad_input = self.grad_input.reshape(orig_input_data_shape)

        return self.grad_input

    def update(self, params):
        self.weight.update(params)

    def share(self, m):
        self.weight = m.weight


class LookupTable(Module):
    """
    Lookup table
    """
    def __init__(self, voc_sz, out_dim):
        """
        Constructor

        Args:
            voc_sz (int): vocabulary size
            out_dim (int): output dimension
        """
        super(LookupTable, self).__init__()
        self.sz      = voc_sz
        self.out_dim = out_dim
        self.weight  = Weight((out_dim, voc_sz))

    def fprop(self, input_data):
        self.output = self.weight.D[:, input_data.T.astype(np.int).flatten()]
        # Matlab's reshape uses Fortran order (i.e. column first)
        self.output = np.squeeze(self.output.reshape((self.out_dim,) + input_data.shape, order='F'))
        return self.output

    def bprop(self, input_data, grad_output):
        # Make sure input_data has one dim lower than grad_output (see the index below)
        if input_data.ndim == grad_output.ndim:
            input_data = np.squeeze(input_data)  # TODO: Seems clumsy!

        input_data = input_data.astype(int)
        c = np.unique(input_data.flatten())
        for i in np.nditer(c):
            self.weight.grad[:, i] += np.sum(grad_output[:, input_data == i], axis=1)

        self.grad_input = []
        return self.grad_input

    def update(self, params):
        self.weight.update(params)

    def share(self, m):
        self.weight = m.weight


class MatVecProd(Module):
    """
    Product of matrix and vector in batch, where
        matrix's shape is [:, :, batch] and vectors is [:, batch]
    Result is a vector of size [:, batch]
    """
    def __init__(self, do_transpose):
        super(MatVecProd, self).__init__()
        self.do_transpose = do_transpose

    def fprop(self, input_data):
        M = input_data[0]
        V = input_data[1]

        # Expand M to 3-dimension and V to 2-dimension
        if M.ndim == 2:
            M = np.expand_dims(M, axis=2)
        if V.ndim == 1:
            V = np.expand_dims(V, axis=1)

        batch_size = M.shape[2]

        if self.do_transpose:
            self.output = np.zeros((M.shape[1], batch_size), np.float32)
            for i in range(batch_size):
                self.output[:, i] = np.dot(M[:, :, i].T, V[:, i])
        else:
            self.output = np.zeros((M.shape[0], batch_size), np.float32)
            for i in range(batch_size):
                self.output[:, i] = np.dot(M[:, :, i], V[:, i])

        return self.output

    def bprop(self, input_data, grad_output):
        M = input_data[0]
        V = input_data[1]

        # Expand M to 3-dimension and V to 2-dimension
        if M.ndim == 2:
            M = np.expand_dims(M, axis=2)
        if V.ndim == 1:
            V = np.expand_dims(V, axis=1)

        batch_size = M.shape[2]

        grad_M = np.zeros_like(M, np.float32)
        grad_V = np.zeros_like(V, np.float32)

        for i in range(batch_size):
            if self.do_transpose:
                grad_M[:, :, i] = np.dot(V[:, [i]], grad_output[:, [i]].T)
                grad_V[:, i] = np.dot(M[:, :, i], grad_output[:, i])
            else:
                grad_M[:, :, i] = np.dot(grad_output[:, [i]], V[:, [i]].T)
                grad_V[:, i] = np.dot(M[:, :, i].T, grad_output[:, i])

        self.grad_input = (grad_M, grad_V)
        return self.grad_input

    def update(self, params):
        pass

    def share(self, m):
        pass


class ReLU(Module):
    """ ReLU module """

    def fprop(self, input_data):
        self.output = np.multiply(input_data, input_data > 0)
        return self.output

    def bprop(self, input_data, grad_output):
        self.grad_input = np.multiply(grad_output, input_data > 0)
        return self.grad_input

    def update(self, params):
        pass

    def share(self, m):
        pass


class SelectTable(Module):
    """ SelectTable which slices input data in a specific dimension """

    def __init__(self, index):
        super(SelectTable, self).__init__()
        self.index = index

    def fprop(self, input_data):
        self.output = input_data[self.index]
        return self.output

    def bprop(self, input_data, grad_output):
        self.grad_input = [grad_output if i == self.index
                           else np.zeros_like(input_elem, np.float32)
                           for i, input_elem in enumerate(input_data)]
        return self.grad_input

    def update(self, params):
        pass

    def share(self, m):
        pass


class Sigmoid(Module):

    def fprop(self, input_data):
        self.output = 1. / (1 + np.exp(-input_data))
        return self.output

    def bprop(self, input_data, grad_output):
        return grad_output * self.output * (1. - self.output)

    def update(self, params):
        pass

    def share(self, m):
        pass


class Softmax(Module):

    def __init__(self, skip_bprop=False):
        super(Softmax, self).__init__()
        self.skip_bprop = skip_bprop  # for the output module

    def fprop(self, input_data):
        input_data -= np.max(input_data, axis=0)
        input_data += 1.0

        a = np.exp(input_data)
        sum_a = a.sum(axis=0)

        self.output = a / sum_a[None, :]  # divide by row
        return self.output

    def bprop(self, input_data, grad_output):
        if not self.skip_bprop:
            z = grad_output - np.sum(self.output * grad_output, axis=0)
            self.grad_input = self.output * z
        else:
            self.grad_input = grad_output

        return self.grad_input

    def update(self, params):
        pass

    def share(self, m):
        pass


class Sum(Module):
    """
    Sum module which sums up input data at specified dimension
    """
    def __init__(self, dim):
        super(Sum, self).__init__()
        self.dim = dim

    def fprop(self, input_data):
        self.output = np.squeeze(np.sum(input_data, axis=self.dim))
        return self.output

    def bprop(self, input_data, grad_output):
        # TODO: Seems clumsy!
        sz = np.array(input_data.shape)
        sz[self.dim] = 1
        grad_output = grad_output.reshape(sz)
        sz[:] = 1
        sz[self.dim] = input_data.shape[self.dim]
        self.grad_input = np.tile(grad_output, sz)
        return self.grad_input

    def update(self, params):
        pass

    def share(self, m):
        pass


class Weight(object):
    def __init__(self, sz):
        """
        Initialize weight
        Args:
            sz (tuple): shape
        """
        self.sz   = sz
        self.D    = 0.1 * np.random.standard_normal(sz)
        self.grad = np.zeros(sz, np.float32)

    def update(self, params):
        """
        Update weights
        """
        max_grad_norm = params.get('max_grad_norm')
        if max_grad_norm and max_grad_norm > 0:
            grad_norm = np.linalg.norm(self.grad, 2)
            if grad_norm > max_grad_norm:
                self.grad = self.grad * max_grad_norm / grad_norm

        self.D -= params['lrate'] * self.grad
        self.grad[:] = 0

    def clone(self):
        m      = Weight(self.sz)
        m.D    = np.copy(self.D)
        m.grad = np.copy(self.grad)
        return m


class Loss(object):
    """ Abstract Loss class """
    __metaclass__ = ABCMeta

    @abstractmethod
    def fprop(self, input_data, target_data):
        """ Abstract function for forward propagation """
        pass

    @abstractmethod
    def bprop(self, input_data, target_data):
        """ Abstract function for back-propagation """
        pass


class CrossEntropyLoss(Loss):

    def __init__(self):
        self.do_softmax_bprop = False
        self.eps              = 1e-7
        self.size_average     = True

    def fprop(self, input_data, target_data):
        tmp = [(t, i) for i, t in enumerate(target_data)]
        z = zip(*tmp)  # unzipping trick !
        cost = np.sum(-np.log(input_data[z]))
        if self.size_average:
            cost /= input_data.shape[1]

        return cost

    def bprop(self, input_data, target_data):
        tmp = [(t, i) for i, t in enumerate(target_data)]
        z = zip(*tmp)

        if self.do_softmax_bprop:
            grad_input = input_data
            grad_input[z] -= 1
        else:
            grad_input = np.zeros_like(input_data, np.float32)
            grad_input[z] = -1. / (input_data[z] + self.eps)

        if self.size_average:
            grad_input /= input_data.shape[1]

        return grad_input

    def get_error(self, input_data, target_data):
        y = input_data.argmax(axis=0)
        return np.sum(y != target_data)
