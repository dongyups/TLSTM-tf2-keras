# reference: On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation
# reference: Explaining Recurrent Neural Network Predictions in Sentiment Analysis
# https://github.com/ArrasL/LRP_for_LSTM


import numpy as np
from numpy import newaxis as na

def npsigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def npgelu(x): # gelu_approximate 임
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def lrp_linear(hin, w, b, hout, Rout, bias_nb_units, eps, bias_factor=0.0, debug=False):
    """
    LRP for a linear layer with input dim D and output dim M.
    Args:
    - hin:            forward pass input, of shape (D,)
    - w:              connection weights, of shape (D, M)
    - b:              biases, of shape (M,)
    - hout:           forward pass output, of shape (M,) (unequal to np.dot(w.T,hin)+b if more than one incoming layer!)
    - Rout:           relevance at layer output, of shape (M,)
    - bias_nb_units:  total number of connected lower-layer units (onto which the bias/stabilizer contribution is redistributed for sanity check)
    - eps:            stabilizer (small positive number)
    - bias_factor:    set to 1.0 to check global relevance conservation, otherwise use 0.0 to ignore bias/stabilizer redistribution (recommended)
    Returns:
    - Rin:            relevance at layer input, of shape (D,)
    """
    sign_out = np.where(hout[na,:]>=0, 1., -1.) # shape (1, M)
    
    numer    = (w * hin[:,na]) + ( bias_factor * (b[na,:]*1. + eps*sign_out*1.) / bias_nb_units ) # shape (D, M)
    # Note: here we multiply the bias_factor with both the bias b and the stabilizer eps since in fact
    # using the term (b[na,:]*1. + eps*sign_out*1.) / bias_nb_units in the numerator is only useful for sanity check
    # (in the initial paper version we were using (bias_factor*b[na,:]*1. + eps*sign_out*1.) / bias_nb_units instead)
    
    denom    = hout[na,:] + (eps*sign_out*1.)   # shape (1, M)
    
    message  = (numer/denom) * Rout[na,:]       # shape (D, M)
    
    Rin      = message.sum(axis=1)              # shape (D,)
    
    if debug:
        print("local diff: ", Rout.sum() - Rin.sum())
    # Note: 
    # - local  layer   relevance conservation if bias_factor==1.0 and bias_nb_units==D (i.e. when only one incoming layer)
    # - global network relevance conservation if bias_factor==1.0 and bias_nb_units set accordingly to the total number of lower-layer connections 
    # -> can be used for sanity check
    
    return Rin


class LRPforTLSTM(object):
    def __init__(self, model, input_dim=59, rnn_dim=128, dense_dim=64, output_dim=1):
        self.model = model
        self.input_dim = input_dim
        self.rnn_dim = rnn_dim
        self.dense_dim = dense_dim
        self.output_dim = output_dim

        # model weights, 아키텍처: tlstm, dense, dense
        # lstm 스택 순서 --> igfo, 원래는 ifgo 순서인것으로 보임
        ln = [model.variables[idx].name for idx in range(len(model.variables))]
        # Time-aware
        self.Wh_Time = model.variables[ln.index("rnn/tlstm_cell/kernel_time:0")].numpy().T
        self.Wb_Time = model.variables[ln.index("rnn/tlstm_cell/bias_time:0")].numpy().T
        # LSTM
        self.Wx_Left = np.concatenate([
            model.variables[ln.index("rnn/tlstm_cell/kernel_i:0")].numpy().T,
            model.variables[ln.index("rnn/tlstm_cell/kernel_g:0")].numpy().T,
            model.variables[ln.index("rnn/tlstm_cell/kernel_f:0")].numpy().T,
            model.variables[ln.index("rnn/tlstm_cell/kernel_o:0")].numpy().T,
        ], axis=0)
        self.Wh_Left = np.concatenate([
            model.variables[ln.index("rnn/tlstm_cell/recurrent_kernel_i:0")].numpy().T,
            model.variables[ln.index("rnn/tlstm_cell/recurrent_kernel_g:0")].numpy().T,
            model.variables[ln.index("rnn/tlstm_cell/recurrent_kernel_f:0")].numpy().T,
            model.variables[ln.index("rnn/tlstm_cell/recurrent_kernel_o:0")].numpy().T,
        ], axis=0)
        self.Wb_Left = np.concatenate([
            model.variables[ln.index("rnn/tlstm_cell/bias_i:0")].numpy().T,
            model.variables[ln.index("rnn/tlstm_cell/bias_g:0")].numpy().T,
            model.variables[ln.index("rnn/tlstm_cell/bias_f:0")].numpy().T,
            model.variables[ln.index("rnn/tlstm_cell/bias_o:0")].numpy().T,
        ], axis=0)
        # Dense 0,1
        self.Wx_Dense0 = model.variables[ln.index("dense/kernel:0")].numpy().T
        self.Wb_Dense0 = model.variables[ln.index("dense/bias:0")].numpy().T
        self.Wx_Dense1 = model.variables[ln.index("dense_1/kernel:0")].numpy().T
        self.Wb_Dense1 = model.variables[ln.index("dense_1/bias:0")].numpy().T


    def set_input(self, w, time_elapse, seq_pos):
        """
        기존 2D input을 w로 설정, seq_pos 이후 패딩 부분을 잘라낼거임
        """
        self.seq_pos = int(seq_pos + 1)
        self.time_elapse = time_elapse[:self.seq_pos]
        self.x = w[:self.seq_pos, :]

        d = self.rnn_dim  # hidden layer dimension
        self.h_Left = np.zeros((self.seq_pos+1, d))
        self.c_Left = np.zeros((self.seq_pos+1, d))


    def forward(self):
        """
        Left to Right TLSTM
        """
        T = self.seq_pos
        d = self.rnn_dim
        # gate indices (assuming the gate ordering in the LSTM weights is i,g,f,o):     
        idx = np.hstack((np.arange(0,d), np.arange(2*d,4*d))).astype(int) # indices of gates i,f,o together
        idx_i, idx_g, idx_f, idx_o = np.arange(0,d), np.arange(1*d,2*d), np.arange(2*d,3*d), np.arange(3*d,4*d) # indices of gates i,g,f,o separately
          
        # initialize
        self.gates_pre_Left  = np.zeros((T, 4*d))  # gates pre-activation
        self.gates_Left      = np.zeros((T, 4*d))  # gates activation

        for t in range(T): 

            elapse = 1.0 / np.log(self.time_elapse[t] + np.exp(1.0))
            elapse = np.dot(np.ones([1, d]), elapse)
            C_ST = np.tanh(np.dot(self.Wh_Time, self.c_Left[t-1]) + self.Wb_Time)
            C_ST_dis = np.multiply(elapse, C_ST)
            self.c_Left[t-1] = self.c_Left[t-1] - C_ST + C_ST_dis

            self.gates_pre_Left[t]    = np.dot(self.Wx_Left, self.x[t]) + np.dot(self.Wh_Left, self.h_Left[t-1]) + self.Wb_Left
            self.gates_Left[t,idx]    = npsigmoid(self.gates_pre_Left[t,idx])
            self.gates_Left[t,idx_g]  = np.tanh(self.gates_pre_Left[t,idx_g]) 

            self.c_Left[t]            = self.gates_Left[t,idx_f] * self.c_Left[t-1] + self.gates_Left[t,idx_i] * self.gates_Left[t,idx_g]
            self.h_Left[t]            = self.gates_Left[t,idx_o] * np.tanh(self.c_Left[t])
            
        self.pre_s  = npgelu(np.dot(self.Wx_Dense0, self.h_Left[T-1]) + self.Wb_Dense0)
        self.s      = npsigmoid(np.dot(self.Wx_Dense1, self.pre_s) + self.Wb_Dense1)

        return self.s # prediction scores


    def lrp(self, LRP_class, eps=0.001, bias_factor=1.0):
        """
        Layer-wise Relevance Propagation (LRP) backward pass.
        Compute the hidden layer relevances by performing LRP for the target class LRP_class
        (according to the papers:
            - https://doi.org/10.1371/journal.pone.0130140
            - https://doi.org/10.18653/v1/W17-5221 )
        """
        # forward pass
        self.forward()
        
        T      = self.seq_pos
        d      = self.rnn_dim
        e      = self.input_dim
        CC     = self.dense_dim
        C      = self.output_dim

        # LSTM에 weights 각 위치
        idx    = np.hstack((np.arange(0,d), np.arange(2*d,4*d))).astype(int) # indices of gates i,f,o together
        idx_i, idx_g, idx_f, idx_o = np.arange(0,d), np.arange(1*d,2*d), np.arange(2*d,3*d), np.arange(3*d,4*d) # indices of gates i,g,f,o separately
        
        # initialize
        Rx        = np.zeros(self.x.shape)
        Rh_Left   = np.zeros((T+1, d))
        Rc_Left   = np.zeros((T+1, d))
        Rg_Left   = np.zeros((T,   d)) # gate g only

        if C != 1:
            # multi-label ex) 10레이블에 3인경우 --> array([0., 0., 0., 1., 1., 1., 1., 1., 1., 1.])
            Rout_mask = np.zeros((C))
            Rout_mask[LRP_class:] = 1.0
            self.final = self.s*Rout_mask
        else:
            # binary 는 마스킹할 부분이 필요없음
            self.final = self.s

        # format reminder: lrp_linear(hin, w, b, hout, Rout, bias_nb_units, eps, bias_factor)
        s_pre_s = lrp_linear(self.pre_s, self.Wx_Dense1.T, self.Wb_Dense1, self.s, self.final, CC+C, eps, bias_factor, debug=False)
        Rh_Left[T-1]  = lrp_linear(self.h_Left[T-1], self.Wx_Dense0.T, self.Wb_Dense0, self.pre_s, s_pre_s, d+CC, eps, bias_factor, debug=False)
        
        for t in reversed(range(T)):
            Rc_Left[t]   += Rh_Left[t]
            Rc_Left[t-1]  = lrp_linear(self.gates_Left[t,idx_f]*self.c_Left[t-1],         np.identity(d), np.zeros((d)), self.c_Left[t], Rc_Left[t], d+d, eps, bias_factor, debug=False)
            Rg_Left[t]    = lrp_linear(self.gates_Left[t,idx_i]*self.gates_Left[t,idx_g], np.identity(d), np.zeros((d)), self.c_Left[t], Rc_Left[t], d+d, eps, bias_factor, debug=False)
            Rx[t]         = lrp_linear(self.x[t],        self.Wx_Left[idx_g].T, self.Wb_Left[idx_g], self.gates_pre_Left[t,idx_g], Rg_Left[t], d+e, eps, bias_factor, debug=False)
            Rh_Left[t-1]  = lrp_linear(self.h_Left[t-1], self.Wh_Left[idx_g].T, self.Wb_Left[idx_g], self.gates_pre_Left[t,idx_g], Rg_Left[t], d+e, eps, bias_factor, debug=False)
            # time-aware 부분
            elapse = 1.0 / np.log(self.time_elapse[t] + np.exp(1.0))
            elapse = np.dot(np.ones([1, d]), elapse)
            Rc_Left[t-1]  = lrp_linear(elapse, self.Wh_Time, self.Wb_Time, self.c_Left[t-1], Rc_Left[t-1], d, eps, bias_factor, debug=False)

        return Rx #, Rh_Left[-1].sum()+Rc_Left[-1].sum()
