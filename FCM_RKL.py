import numpy as np
from scipy.spatial.distance import cdist


def cmeans(data, c, h, error, maxiter, metric='euclidean', init=None, seed=None):
    """
    Fuzzy c-means clustering algorithm [1].

    Parameters
    ----------
    data : 2d array, size (S, N)
        Data to be clustered.  N is the number of data sets; S is the number
        of features within each sample vector.
    c : int
        Desired number of clusters or classes.
    m : float
        Array exponentiation applied to the membership function u_old at each
        iteration, where U_new = u_old ** m.
    error : float
        Stopping criterion; stop early if the norm of (u[p] - u[p-1]) < error.
    maxiter : int
        Maximum number of iterations allowed.
    metric: string
        By default is set to euclidean. Passes any option accepted by
        ``scipy.spatial.distance.cdist``.
    init : 2d array, size (S, N)
        Initial fuzzy c-partitioned matrix. If none provided, algorithm is
        randomly initialized.
    seed : int
        If provided, sets random seed of init. No effect if init is
        provided. Mainly for debug/testing purposes.

    Returns
    -------
    cntr : 2d array, size (S, c)
        Cluster centers.  Data for each center along each feature provided
        for every cluster (of the `c` requested clusters).
    u : 2d array, (S, N)
        Final fuzzy c-partitioned matrix.
    u0 : 2d array, (S, N)
        Initial guess at fuzzy c-partitioned matrix (either provided init or
        random guess used if init was not provided).
    d : 2d array, (S, N)
        Final Euclidian distance matrix.
    jm : 1d array, length P
        Objective function history.
    p : int
        Number of iterations run.
    fpc : float
        Final fuzzy partition coefficient.


    Notes
    -----
    The algorithm implemented is from Ross et al. [1]_.

    Fuzzy C-Means has a known problem with high dimensionality datasets, where
    the majority of cluster centers are pulled into the overall center of
    gravity. If you are clustering data with very high dimensionality and
    encounter this issue, another clustering method may be required. For more
    information and the theory behind this, see Winkler et al. [2]_.

    References
    ----------
    .. [1] Ross, Timothy J. Fuzzy Logic With Engineering Applications, 3rd ed.
           Wiley. 2010. ISBN 978-0-470-74376-8 pp 352-353, eq 10.28 - 10.35.

    .. [2] Winkler, R., Klawonn, F., & Kruse, R. Fuzzy c-means in high
           dimensional spaces. 2012. Contemporary Theory and Pragmatic
           Approaches in Fuzzy Computing Utilization, 1.
    """
    # Setup u0
    # 初始化聚类划分矩阵
    if init is None:
        if seed is not None:
            np.random.seed(seed=seed)
        n = data.shape[1]
        u0 = np.random.rand(c, n)
        u0 = normalize_columns(u0)
        init = u0.copy()
    u0 = init
    u = np.fmax(u0, np.finfo(np.float64).eps)#精度的比较，小于这个精度会自动取这个值
    #计算派i
    s = np.sum(u, axis=1, keepdims=True)/u.shape[1]

    # Initialize loop parameters
    jm = np.zeros(0)
    p = 0


    # Main cmeans loop
    while p < maxiter - 1:
        u2 = u.copy()
        s0 = s.copy()
        [cntr, u, Jjm , d, s] = _cmeans0(data, u2, c, h, s0, metric)
        jm = np.hstack((jm, Jjm))
        p += 1

        # Stopping rule
        if np.linalg.norm(u - u2) < error:
            break

    # Final calculations
    error = np.linalg.norm(u - u2)
    fpc = _fp_coeff(u)

    return cntr, u, u0, d, jm, p, fpc

def _cmeans0(data, u_old, c, h, s, metric):
    """
    Single step in generic fuzzy c-means clustering algorithm.

    Modified from Ross, Fuzzy Logic w/Engineering Applications (2010),
    pages 352-353, equations 10.28 - 10.35.

    Parameters inherited from cmeans()
    """
    # Normalizing, then eliminating any potential zero values.标准化，然后消除任何潜在的零值。
    u_old = normalize_columns(u_old)# 标准化，然后消除任何潜在的零值，，用于最开始的时候，归一化
    u_old = np.fmax(u_old, np.finfo(np.float64).eps)#精度的比较，小于这个精度会自动取这个值

     # 计算分布先验Pi [[s1],[s2],...[s..]]
    s = np.sum(u_old, axis=1, keepdims=True) / u_old.shape[1]  ##[c1 c2 ....cn]每个聚类中心的先验分布
    s = np.fmax(s, np.finfo(np.float64).eps)#精度的比较，小于这个精度会自动取这个值

    um = u_old

    # Calculate cluster centers
    data = data.T
    # 点乘，得到聚类中心
    cntr = um.dot(data) / np.atleast_2d(um.sum(axis=1)).T #待处理公式聚类中心,不用改动

    d = _distance(data, cntr, metric)  #待处理欧式距离公式，目前估计也不用改动
    d = np.fmax(d, np.finfo(np.float64).eps)##精度的比较，小于这个精度会自动取这个值

    jm = (um * d ** 2).sum()

    # u = normalize_power_columns(d, - 2. / (m - 1)) #待处理划分矩阵公式
    u = _uij(d, s, h)
    # u = np.exp()#指数运算

    return cntr, u, jm, d, s

'''
将模糊m换成正则化项系数
1.先计算派i
2.在更加派i求隶属度
3.聚类中心


'''

def _uij(d, s, h):
    '''

    :param d: 聚类距离矩阵
    :param n: 正则化系数
    :return:
    '''
    s1 = s.repeat(d.shape[1], axis=1)
    tmp = s1*np.exp(d/(-h))
    tmp = np.fmax(tmp, np.finfo(np.float64).eps)##精度的比较，小于这个精度会自动取这个值
    # s2 = s.repeat(d.shape[1], axis=1)
    tmp1 = np.sum(tmp, axis=0, keepdims=True)##
    # 需要改的地方。。。。。
    temp1 = tmp1.repeat(d.shape[0], axis=0)
    u = tmp/temp1
    u = normalize_columns(u)
    return u

def _fp_coeff(u):
    """
    Fuzzy partition coefficient `fpc` relative to fuzzy c-partitioned
    matrix `u`. Measures 'fuzziness' in partitioned clustering.

    Parameters
    ----------
    u : 2d array (C, N)
        Fuzzy c-partitioned matrix; N = number of data points and C = number
        of clusters.

    Returns
    -------
    fpc : float
        Fuzzy partition coefficient.

    """
    n = u.shape[1]
    return np.trace(u.dot(u.T)) / float(n)

def _distance(data, centers, metric='euclidean'):
    """
    Euclidean distance from each point to each cluster center.

    Parameters
    ----------
    data : 2d array (N x Q)
        Data to be analyzed. There are N data points.
    centers : 2d array (C x Q)
        Cluster centers. There are C clusters, with Q features.
    metric: string
        By default is set to euclidean. Passes any option accepted by
        ``scipy.spatial.distance.cdist``.
    Returns
    -------
    dist : 2d array (C x N)
        Euclidean distance from each point, to each cluster center.

    See Also
    --------
    scipy.spatial.distance.cdist
    """
    return cdist(data, centers, metric=metric).T


"""
_normalize_columns.py : Normalize columns.
"""

# import numpy as np

def normalize_columns(columns):
    """
    Normalize columns of matrix.

    Parameters
    ----------
    columns : 2d array (M x N)
        Matrix with columns

    Returns
    -------
    normalized_columns : 2d array (M x N)
        columns/np.sum(columns, axis=0, keepdims=1)
    """

    # broadcast sum over columns
    normalized_columns = columns / np.sum(columns, axis=0, keepdims=1)

    return normalized_columns

def normalize_power_columns(x, exponent):
    """
    Calculate normalize_columns(x**exponent)
    in a numerically safe manner.

    Parameters
    ----------
    x : 2d array (M x N)
        Matrix with columns
    n : float
        Exponent

    Returns
    -------
    result : 2d array (M x N)
        normalize_columns(x**n) but safe

    """

    assert np.all(x >= 0.0)

    x = x.astype(np.float64)

    # values in range [0, 1]
    x = x / np.max(x, axis=0, keepdims=True)

    # values in range [eps, 1]
    x = np.fmax(x, np.finfo(x.dtype).eps)

    if exponent < 0:
        # values in range [1, 1/eps]
        x /= np.min(x, axis=0, keepdims=True)

        # values in range [1, (1/eps)**exponent] where exponent < 0
        # this line might trigger an underflow warning
        # if (1/eps)**exponent becomes zero, but that's ok
        x = x ** exponent
    else:
        # values in range [eps**exponent, 1] where exponent >= 0
        x = x ** exponent

    result = normalize_columns(x)

    return result



