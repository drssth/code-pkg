import math
import numpy as np
from typing import Tuple, Callable

def check_params(Q, r, J):
    if Q < 1:
        raise ValueError("Q must be greater or equal 1!")
    if r <= 1:
        raise ValueError("The redundancy must be strictly greater than 1")
    if J < 1:
        raise ValueError("J must be a positive integer!")



def analysis_filter_bank(x, n0, n1):
    x = np.array(x)
    n = x.shape[0]

    p = int((n-n1) / 2)
    t = int((n0 + n1 - n) / 2 - 1)
    s = int((n - n0) / 2)

    # transition band function
    v = np.arange(start=1, stop=t+1) / (t+1) * np.pi
    transit_band = (1 + np.cos(v)) * np.sqrt(2 - np.cos(v)) / 2.0

    # low-pass subband
    lp_subband = np.zeros(n0, dtype=x.dtype)
    # 0 --> 0
    lp_subband[0] = x[0]
    # 1:p+1 --> 1:p+1
    lp_subband[1:p+1] = x[1:p+1]
    # p+1:p+t+1 --> p+1:p+t+1
    lp_subband[p+1:p+t+1] = x[p+1:p+t+1] * transit_band
    # n0/2
    lp_subband[int(n0/2)] = 0
    # np-p-t:n0-p --> n-p-t:n-p
    lp_subband[n0-p-t:n0-p] = x[n-p-t:n-p] * transit_band
    # n0-p:n0 --> n-p:n
    lp_subband[n0-p:n0] = x[n-p:n]

    # high-pass subband
    hp_subband = np.zeros(n1, dtype=x.dtype)
    # 0 --> 0
    hp_subband[0] = 0
    # 1:t+1 --> p+1:p+t+1
    hp_subband[1:t+1] = x[p+1:p+t+1] * np.flip(transit_band)
    # t+1:t+s+1 --> p+t+1:p+t+s+1
    hp_subband[t+1:t+s+1] = x[p+t+1:p+t+s+1]
    # n1/2
    if n % 2 == 0:
        hp_subband[int(n1/2)] = x[int(n/2)]
    # n1-t-s:n1-t --> n-p-t-s:n-p-t
    hp_subband[n1-t-s:n1-t] = x[n-p-t-s:n-p-t]
    # n1-t:n1 --> n-p-t:n-p
    hp_subband[n1-t:n1] = x[n-p-t:n-p] * np.flip(transit_band)

    return lp_subband, hp_subband



def synthesis_filter_bank(V0, V1, N):
    N0 = V0.shape[0]
    N1 = V1.shape[0]

    s = int((N-N0) / 2)
    p = int((N-N1) / 2)
    t = int((N0+N1-N)/2 - 1)

    v = np.arange(start=1, stop=t+1) / (t+1) * np.pi
    transit_band = (1 + np.cos(v)) * np.sqrt(2 - np.cos(v)) / 2.0

    # low-pass subband
    lp_subband = np.zeros(N, dtype=complex)
    # 0 --> 0
    lp_subband[0] = V0[0]
    # 1:p+1 --> 1:p+1
    lp_subband[1:p+1] = V0[1:p+1]
    # p+1:p+t+1 --> p+1:p+t+1
    lp_subband[1+p:p+t+1] = V0[1+p:p+t+1] * transit_band
    # p+t+1:p+t+s+1 --> 0
    lp_subband[p+t+1:p+t+s+1] = 0
    # N/2
    if N % 2 == 0:
        lp_subband[int(N/2)] = 0
    # N-p-t-s:N-p-t --> 0
    lp_subband[N-p-t-s:N-p-t] = 0
    # N-p-t:N-p --> N0-p-t:N0-p
    lp_subband[N-p-t:N-p] = V0[N0-p-t:N0-p] * transit_band
    # N-p:N --> N0-p:N0
    lp_subband[N-p:N] = V0[N0-p:N0]

    # high-pass subband
    hp_subband = np.zeros(N, dtype=complex)
    # 0 --> 0
    hp_subband[0] = 0
    #  1:p+1 --> 0
    hp_subband[1:p+1] = 0
    # p+1:p+t+1 --> 1:t+1
    hp_subband[p+1:p+t+1] = V1[1:t+1] * np.flip(transit_band)
    # p+t+1:p+t+s+1 --> t+1:t+s+1
    hp_subband[p+t+1:p+t+s+1] = V1[t+1:t+s+1]
    # N/2
    if N % 2 == 0:
        hp_subband[int(N/2)] = V1[int(N1/2)]
    # N-p-t-s:N-p-t -->N1-t-s:N1-t
    hp_subband[N-p-t-s:N-p-t] = V1[N1-t-s:N1-t]
    # N-p-t:N-p --> N1-t:N1
    hp_subband[N-p-t:N-p] = V1[N1-t:N1] * np.flip(transit_band)
    # N-p:N --> 0
    hp_subband[N-p:N] = 0

    return lp_subband + hp_subband


def low_pass_scaling(x, n0):
    # - output Y will be length N0
    # - length(X) should be even
    N = x.shape[0]
    Y = np.zeros(n0, dtype=x.dtype)

    if N % 2 != 0:
        raise ValueError("Input signal x needs to be of even length!")

    if n0 == 0:
        return 0

    if n0 <= N:
        k = np.array(list(range(0, int(n0/2))))
        Y[k] = x[k]
        Y[int(n0/2)] = x[int(N/2)]

        k = np.array(list(range(1, int(n0/2))))
        Y[n0-k] = x[N-k]

    elif n0 >= N:
        k = np.array(list(range(0, int(N/2))))
        Y[k] = x[k]

        k = np.array(list(range(int(N/2), int(n0/2))))
        Y[k] = 0

        Y[int(n0/2)] = x[int(N/2)]

        Y[n0-k] = 0

        k = np.array(list(range(1, int(N/2))))
        Y[n0-k] = x[N-k]
    return Y


def uDFT(x):
    # unitary DFT
    N = x.shape[0]
    xx = np.fft.fft(x)
    xx = xx / math.sqrt(N)
    return xx


def uDFTinv(x):
    # inverse unitary DFT
    N = x.shape[0]
    xx = np.fft.ifft(x)
    xx = math.sqrt(N) * xx
    #xx = np.absolute(xx)
    xx = np.real(xx)
    return xx


def next_power_of_2(k):
    r = math.log(k, 2)
    c = math.ceil(r)
    return 2 ** c


def tqwt(x, Q, r, J):
    check_params(Q, r, J)

    if x.shape[0] % 2 or len(x.shape) != 1:
        raise ValueError("Input signal x needs to be one dimensional and of even length!")
    x = np.asarray(x)

    beta = 2.0 / float(Q + 1)
    alpha = 1.0 - beta / float(r)
    N = x.shape[0]

    Jmax = int(np.floor(np.log(beta * N / 8.0) / np.log(1.0 / alpha)))

    if J > Jmax:
        if Jmax > 0:
            raise ValueError("Too many subbands, reduce subbands to " + str(Jmax))
        else:
            raise ValueError("increase signal length")

    # unitary DFT
    X = uDFT(x)
    # init list of wavelet coefficients
    wm = []
    W = None

    for j in range(1, J + 1):
        n0 = 2 * round(alpha ** j * N / 2.0)
        n1 = 2 * round(beta * alpha ** (j - 1) * N / 2.0)
        X, W = analysis_filter_bank(X, n0, n1)
        wm.append(uDFTinv(W))

    # inverse unitary DFT
    wm.append(uDFTinv(X))
    return wm



def tqwt_radix2(x, Q, r, J):
    check_params(Q, r, J)

    beta = 2.0 / float(Q + 1)
    alpha = 1.0 - beta / float(r)
    L = x.shape[0]
    N = next_power_of_2(L)

    Jmax = int(np.floor(np.log(beta * N / 8.0) / np.log(1.0 / alpha)))

    if J > Jmax:
        if Jmax > 0:
            raise ValueError("Too many subbands, reduce subbands to " + str(Jmax))
        else:
            raise ValueError("increase signal length")

    # unitary DFT
    X = uDFT(x)
    # init list of wavelet coefficients
    wm = []
    lps = []
    W = None

    for j in range(1, J + 1):
        n0 = 2 * round(alpha ** j * N / 2.0)
        n1 = 2 * round(beta * alpha ** (j - 1) * N / 2.0)
        X, W = analysis_filter_bank(X, n0, n1)
        W = low_pass_scaling(W, next_power_of_2(n1))
        WW = uDFTinv(W)
        lps.append(W)
        wm.append(WW)

    # inverse unitary DFT
    X = low_pass_scaling(X, next_power_of_2(n0))
    lps.append(X)
    WW = uDFTinv(X)
    wm.append(WW)
    return wm, lps



def itqwt(w, Q, r, N):
    check_params(Q,r,1)

    beta = 2.0 / float(Q + 1)
    alpha = 1.0 - beta / float(r)
    J = len(w) - 1

    Y = uDFT(w[J])

    for j in range(J, 0, -1):
        W = uDFT(w[j])
        M = 2 * round((alpha ** (j-1)) * N/2)
        Y = synthesis_filter_bank(Y, W, M)

    y = uDFTinv(Y)
    return y



def itqwt_radix2(w, Q, r, L):
    check_params(Q,r,1)

    beta = 2.0 / float(Q + 1)
    alpha = 1.0 - beta / float(r)
    J = len(w) - 1

    N = next_power_of_2(L)
    Y = uDFT(w[J])

    M = 2 * round((alpha ** J) * N/2)
    Y = low_pass_scaling(Y, M)

    for j in range(J, 0, -1):
        W = uDFT(w[j])
        N1 = 2 * round(beta * (alpha ** (j-1)) * N/2)
        W = low_pass_scaling(W, N1)
        M = 2 * round((alpha ** (j-1)) * N/2)
        Y = synthesis_filter_bank(Y, W, M)

    y = uDFTinv(Y)
    y = y[0:L+1]
    return y



def compute_wavelets(n: int, q: float, redundancy: float, stages: int) -> np.ndarray:

    n_zeros = np.zeros(n)
    wavelet_shaped_zeros = tqwt_radix2(n_zeros, q, redundancy, stages)
    wavelets = np.array([None for _ in range(stages + 1)], dtype=np.object)

    for j_i in range(stages + 1):
        w = deepcopy(wavelet_shaped_zeros)
        m = int(round(w[j_i].shape[0]/2))
        w[j_i][m] = 1.0
        wavelet = itqwt_radix2(w, q, redundancy, n)
        wavelets[j_i] = wavelet

    return wavelets


def compute_wavelet_norms(n: int, q: float, redundancy: float, stages: int,
                          norm_function: Callable = np.linalg.norm) -> np.ndarray:
    return np.array([norm_function(w_j) for w_j in compute_wavelets(n, q, redundancy, stages)])



class DualQDecomposition:
    """
    Resonance signal decomposition using two Q-factors for signal with noise.
    This is obtained by minimizing the cost function:
    || x - x1 - x2 ||_2^2 + lambda_1 ||w1||_1 + lambda_2 ||w2||_2
    References
    ----------
    .. [1] Selesnick, I. W. (2011). Resonance-based signal decomposition: A new sparsity-enabled signal analysis method.
           Signal Processing, 91(12), 2793-2809.
    .. [2] Selesnick, I. W. (2011). TQWT toolbox guide. Electrical and Computer Engineering, Polytechnic Institute of
           New York University. Available online at: http://eeweb.poly.edu/iselesni/TQWT/index.html
    Parameters
    ----------
    q1: float
    redundancy_1: float
    stages_1: int
    q2: float
    redundancy_2: float
    stages_2: int
    lambda_1: float
    lambda_2: float
    mu: float
    num_iterations: int
    compute_cost_function: bool
    """

    def __init__(self, q1: float, redundancy_1: float, stages_1: int, q2: float, redundancy_2: float, stages_2: int,
                 lambda_1: float, lambda_2: float, mu: float, num_iterations: int,
                 compute_cost_function: bool = False):
        # parameters of the first transform, define (i)tqwt lambdas
        self._q1 = q1
        self._redundancy_1 = redundancy_1
        self._stages_1 = stages_1
        self.tqwt1 = lambda x: tqwt_radix2(x, self._q1, self._redundancy_1, self._stages_1)
        self.itqwt1 = lambda w, n: itqwt_radix2(w, self._q1, self._redundancy_1, n)
        # parameters of the second transform, define (i)tqwt lambdas
        self._q2 = q2
        self._redundancy_2 = redundancy_2
        self._stages_2 = stages_2
        self.tqwt2 = lambda x: tqwt_radix2(x, self._q2, self._redundancy_2, self._stages_2)
        self.itqwt2 = lambda w, n: itqwt_radix2(w, self._q2, self._redundancy_2, n)
        # SALSA parameters
        self._lambda_1 = lambda_1
        self._lambda_2 = lambda_2
        self._mu = mu
        self._num_iterations = num_iterations

        self._history = None
        self._compute_cost_function = compute_cost_function

    def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform a Dual-Q decomposition on the one-dimensional time-domain signal `x`.
        Parameters
        ----------
        x: np.ndarray, one-dimensional with even length
            Input signal
        Returns
        -------
        x1: np.ndarray with x.shape
            Signal component corresponding to the first transform
        x2: np.ndarray with x.shape
            Signal component corresponding to the second transform
        """
        assert len(x.shape) == 1

        n = x.shape[0]
        w1, w2 = self.tqwt1(x), self.tqwt2(x)
        d1, d2 = (np.array([np.zeros(s.shape, s.dtype) for s in w], dtype=np.object) for w in [w1, w2])
        u1, u2 = (np.array([np.zeros(s.shape, s.dtype) for s in w], dtype=np.object) for w in [w1, w2])
        t1 = self._lambda_1 * compute_wavelet_norms(n, self._q1, self._redundancy_1, self._stages_1) / (2 * self._mu)
        t2 = self._lambda_2 * compute_wavelet_norms(n, self._q2, self._redundancy_2, self._stages_2) / (2 * self._mu)

        for iter_idx in range(self._num_iterations):

            for j in range(self._stages_1 + 1):
                u1[j] = self.soft_threshold(w1[j] + d1[j], t1[j]) - d1[j]
            for j in range(self._stages_2 + 1):
                u2[j] = self.soft_threshold(w2[j] + d2[j], t2[j]) - d2[j]

            c = (x - self.itqwt1(u1, n) - self.itqwt2(u2, n)) / (self._mu + 2)
            d1, d2 = self.tqwt1(c), self.tqwt2(c)

            for j in range(self._stages_1 + 1):
                w1[j] = d1[j] + u1[j]
            for j in range(self._stages_2 + 1):
                w2[j] = d2[j] + u2[j]

            if self._compute_cost_function:
                self.update_history(iter_idx, w1, w2, t1, t2, x)

        return self.itqwt1(w1, n), self.itqwt2(w2, n)

    @staticmethod
    def soft_threshold(x: np.ndarray, thresh: float) -> np.ndarray:
        y = np.abs(x)-thresh
        y[np.where(y < 0)] = 0
        return y / (y + thresh) * x

    def update_history(self, iter_idx, w1, w2, t1, t2, x):
        if iter_idx == 0:  # re-initialize the history for every run
            self._history = SortedDict()

        residual = x - self.itqwt1(w1, x.shape[0]) - self.itqwt2(w2, x.shape[0])
        cost_function = np.sum(np.abs(residual) ** 2)
        for j in range(self._stages_1 + 1):
            cost_function += t1[j] * np.sum(np.abs(w1[j]))
        for j in range(self._stages_2 + 1):
            cost_function += t2[j] * np.sum(np.abs(w2[j]))
        self._history[iter_idx] = cost_function

    @property
    def history(self):
        return self._history


if __name__ == '__main__':
    s = ''
    with open('speech2.txt', 'r') as f:
        s = f.read()

    x = [float(i) for i in s.split()]
    x = np.array(x)
    N = x.shape[0]
    print('x', x[0:10])

    Q, r, L, L1 = 3, 3, 23, 10

    w, lps = tqwt_radix2(x, Q, r, L)
    l = len(w)
    for i in range(l):
        np.savetxt("w-{}.csv".format(i), w[i], delimiter=",")

    for i in range(l):
        np.savetxt("lps-{}.csv".format(i), lps[i], delimiter=",")

    y = itqwt_radix2(w, Q, r, N)

    mae = abs(y - x)
    reconstruction_error = np.mean(mae)
    print(mae)
    print('reconstruction_error', reconstruction_error)
