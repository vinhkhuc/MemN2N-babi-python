import numpy as np

from memn2n.nn import ElemMult, Identity, Sequential, LookupTable, Module
from memn2n.nn import Sum, Parallel, Softmax, MatVecProd


class Memory(Module):
    """
    Memory:
        Query module  = Parallel(LookupTable + Identity) + MatVecProd with transpose + Softmax
        Output module = Parallel(LookupTable + Identity) + MatVecProd
    """
    def __init__(self, train_config):
        super(Memory, self).__init__()

        self.sz        = train_config["sz"]
        self.voc_sz    = train_config["voc_sz"]
        self.in_dim    = train_config["in_dim"]
        self.out_dim   = train_config["out_dim"]

        # TODO: Mark self.nil_word and self.data as None since they will be overriden eventually
        # In build.model.py, memory[i].nil_word = dictionary['nil']"
        self.nil_word  = train_config["voc_sz"]
        self.config    = train_config
        self.data      = np.zeros((self.sz, train_config["bsz"]), np.float32)

        self.emb_query = None
        self.emb_out   = None
        self.mod_query = None
        self.mod_out   = None
        self.probs     = None

        self.init_query_module()
        self.init_output_module()

    def init_query_module(self):
        self.emb_query = LookupTable(self.voc_sz, self.in_dim)
        p = Parallel()
        p.add(self.emb_query)
        p.add(Identity())

        self.mod_query = Sequential()
        self.mod_query.add(p)
        self.mod_query.add(MatVecProd(True))
        self.mod_query.add(Softmax())

    def init_output_module(self):
        self.emb_out = LookupTable(self.voc_sz, self.out_dim)
        p = Parallel()
        p.add(self.emb_out)
        p.add(Identity())

        self.mod_out = Sequential()
        self.mod_out.add(p)
        self.mod_out.add(MatVecProd(False))

    def reset(self):
        self.data[:] = self.nil_word

    def put(self, data_row):
        self.data[1:, :] = self.data[:-1, :]  # shift rows down
        self.data[0, :] = data_row            # add the new data row on top

    def fprop(self, input_data):
        self.probs = self.mod_query.fprop([self.data, input_data])
        self.output = self.mod_out.fprop([self.data, self.probs])
        return self.output

    def bprop(self, input_data, grad_output):
        g1 = self.mod_out.bprop([self.data, self.probs], grad_output)
        g2 = self.mod_query.bprop([self.data, input_data], g1[1])
        self.grad_input = g2[1]
        return self.grad_input

    def update(self, params):
        self.mod_out.update(params)
        self.mod_query.update(params)
        self.emb_out.weight.D[:, self.nil_word] = 0

    def share(self, m):
        pass


class MemoryBoW(Memory):
    """
    MemoryBoW:
        Query module  = Parallel((LookupTable + Sum(1)) + Identity) + MatVecProd with transpose + Softmax
        Output module = Parallel((LookupTable + Sum(1)) + Identity) + MatVecProd
    """
    def __init__(self, config):
        super(MemoryBoW, self).__init__(config)
        self.data = np.zeros((config["max_words"], self.sz, config["bsz"]), np.float32)

    def init_query_module(self):
        self.emb_query = LookupTable(self.voc_sz, self.in_dim)
        s = Sequential()
        s.add(self.emb_query)
        s.add(Sum(dim=1))

        p = Parallel()
        p.add(s)
        p.add(Identity())

        self.mod_query = Sequential()
        self.mod_query.add(p)
        self.mod_query.add(MatVecProd(True))
        self.mod_query.add(Softmax())

    def init_output_module(self):
        self.emb_out = LookupTable(self.voc_sz, self.out_dim)
        s = Sequential()
        s.add(self.emb_out)
        s.add(Sum(dim=1))

        p = Parallel()
        p.add(s)
        p.add(Identity())

        self.mod_out = Sequential()
        self.mod_out.add(p)
        self.mod_out.add(MatVecProd(False))


class MemoryL(Memory):
    """
    MemoryL:
        Query module  = Parallel((LookupTable + ElemMult + Sum(1)) + Identity) + MatVecProd with transpose + Softmax
        Output module = Parallel((LookupTable + ElemMult + Sum(1)) + Identity) + MatVecProd
    """
    def __init__(self, train_config):
        super(MemoryL, self).__init__(train_config)
        self.data = np.zeros((train_config["max_words"], self.sz, train_config["bsz"]), np.float32)

    def init_query_module(self):
        self.emb_query = LookupTable(self.voc_sz, self.in_dim)
        s = Sequential()
        s.add(self.emb_query)
        s.add(ElemMult(self.config["weight"]))
        s.add(Sum(dim=1))

        p = Parallel()
        p.add(s)
        p.add(Identity())

        self.mod_query = Sequential()
        self.mod_query.add(p)
        self.mod_query.add(MatVecProd(True))
        self.mod_query.add(Softmax())

    def init_output_module(self):
        self.emb_out = LookupTable(self.voc_sz, self.out_dim)
        s = Sequential()
        s.add(self.emb_out)
        s.add(ElemMult(self.config["weight"]))
        s.add(Sum(dim=1))

        p = Parallel()
        p.add(s)
        p.add(Identity())

        self.mod_out = Sequential()
        self.mod_out.add(p)
        self.mod_out.add(MatVecProd(False))

