import os
import numpy as np
import pickle
import torch
import random
from datetime import datetime
# from sksfa import SFA

import numpy as np
 
class SFA:  # slow feature analysis class
    def __init__(self):
        self._Z = []
        self._B = []
        self._eigenVector = []
 
    def getB(self, data):
        self._B = np.matrix(data.T.dot(data)) / (data.shape[0] - 1)
        ## torch.dot 不支持2D tensor，换成torch.mm
        # self._B = np.matrix(data.T.mm(data)) / (data.shape[0] - 1)
 
    def getZ(self, data):
        derivativeData = self.makeDiff(data)
        self._Z = np.matrix(derivativeData.T.dot(derivativeData)) / (derivativeData.shape[0] - 1)
        ## torch.dot 不支持2D tensor，换成torch.mm
        # self._Z = np.matrix(derivativeData.T.mm(derivativeData)) / (derivativeData.shape[0] - 1)
 
    def makeDiff(self, data):
        diffData = np.mat(np.zeros((data.shape[0], data.shape[1])))
        for i in range(data.shape[1] - 1):
            diffData[:, i] = data[:, i] - data[:, i + 1]
        diffData[:, -1] = data[:, -1] - data[:, 0]
        return np.mat(diffData)
 
    def fit_transform(self, data, threshold=1e-7, conponents=-1):
        if conponents == -1:
            conponents = data.shape[0]
        self.getB(data)
        U, s, V = np.linalg.svd(self._B)
 
        count = len(s)
        for i in range(len(s)):
            if s[i] ** (0.5) < threshold:
                count = i
                break
        s = s[0:count]
        s = s ** 0.5
        S = (np.mat(np.diag(s))).I
        U = U[:, 0:count]
        whiten = S * U.T
        Z = (whiten * data.T).T

 
        self.getZ(Z)
        PT, O, P = np.linalg.svd(self._Z)
 
        self._eigenVector = P * whiten
        self._eigenVector = self._eigenVector[-1 * conponents:, :]
 
        return data.dot(self._eigenVector.T)
        # return data.mm(self._eigenVector.T)
 
    def transfer(self, data):
        return data.dot(self._eigenVector.T)
        # return data.mm(self._eigenVector.T)

def lifting_old(arr, sub_length, dim=5):
    ## 一维数据堆叠升维，通过滑动采样然后进行堆叠获得多维数据,
    ## 旧版，实现功能，但是仅能用于普通变量，不能用于tensor变量
    # 输入参数：
        # arr: 输入的一维数据
        # sub_length: 滑动采样的子样本长度
        # dim: lifting升维的输出目标维度
    # 输出参数：
        # res: lifting升维后的多维数据，维度为dim
    res = []
    arr_length = len(arr)
    step = int((arr_length-sub_length) / (dim-1)) # 滑动步长
    for i in range(dim):
        res.append(arr[0+(i*step):sub_length+(i*step)])
    res = np.squeeze(res) # 去除维度项为1的维度
    res = np.transpose(res,(1,0)) # 调换0,1维度，调换之后为：（时间维度，堆叠增加的特征维度）

    return res

def lifting(arr, sub_length, dim=5):
    ## 一维数据堆叠升维，通过滑动采样然后进行堆叠获得多维数据
    # 输入参数：
        # arr: 输入的一维数据
        # sub_length: 滑动采样的子样本长度
        # dim: lifting升维的输出目标维度
    # 输出参数：
        # res: lifting升维后的多维数据，维度为dim
    res = []
    # res = np.array(res)
    # res = torch.from_numpy(res).to(torch.float)
    arr_length = len(arr)
    step = int((arr_length-sub_length) / (dim-1)) # 滑动步长
    for i in range(dim):
        res.append(arr[0+(i*step):sub_length+(i*step)])
        # res = torch.cat((res, arr[0+(i*step):sub_length+(i*step)]), dim=0)
    # res = np.squeeze(res) # 去除维度项为1的维度
    # res = torch.cat(res,dim=0)
    res = torch.stack(res,dim=0)
    # res = np.transpose(res,(1,0)) # 调换0,1维度，调换之后为：（时间维度，堆叠增加的特征维度）
    res = res.transpose(1,0)

    return res

def sfa(data, n_comp=1):
    ## 慢特征分析的实现
    # 输入参数：
        # data：需要进行慢特征分析的数据
        # n_comp：慢特征分析输出项数，即前n_comp个慢特征
    # 输出参数：
        # extracted_features:提取的慢特征

    sfa_model = SFA()
    data = data.cpu().numpy()
    extracted_features = sfa_model.fit_transform(data,conponents=n_comp)
    extracted_features = torch.Tensor(extracted_features)
    return extracted_features

def lifting_sfa(abc,bcd,sub_length=300,dim=10,n_comp=1):
    abc_new = []
    bcd_new = []
    for i in range(abc.shape[0]):
        abc_sfa_temp = abc[i,:,0]
        bcd_sfa_temp = bcd[i,:,0]
        abc_sfa_temp = lifting(abc_sfa_temp, sub_length=sub_length, dim=dim)
        bcd_sfa_temp = lifting(bcd_sfa_temp, sub_length=sub_length, dim=dim)
        abc_sfa_temp = sfa(abc_sfa_temp, n_comp=n_comp)   
        bcd_sfa_temp = sfa(bcd_sfa_temp, n_comp=n_comp)   
        abc_new.append(abc_sfa_temp)
        bcd_new.append(bcd_sfa_temp)
    abc_new = torch.stack(abc_new, dim=0)
    bcd_new = torch.stack(bcd_new, dim=0)
    return abc_new, bcd_new

def pkl_save(name, var):
    with open(name, 'wb') as f:
        pickle.dump(var, f)

def pkl_load(name):
    with open(name, 'rb') as f:
        return pickle.load(f)
    
def torch_pad_nan(arr, left=0, right=0, dim=0):
    if left > 0:
        padshape = list(arr.shape)
        padshape[dim] = left
        arr = torch.cat((torch.full(padshape, np.nan), arr), dim=dim)
    if right > 0:
        padshape = list(arr.shape)
        padshape[dim] = right
        arr = torch.cat((arr, torch.full(padshape, np.nan)), dim=dim)
    return arr
    
def pad_nan_to_target(array, target_length, axis=0, both_side=False):
    assert array.dtype in [np.float16, np.float32, np.float64]
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array
    npad = [(0, 0)] * array.ndim
    if both_side:
        npad[axis] = (pad_size // 2, pad_size - pad_size//2)
    else:
        npad[axis] = (0, pad_size)
    return np.pad(array, pad_width=npad, mode='constant', constant_values=np.nan)

def split_with_nan(x, sections, axis=0):
    assert x.dtype in [np.float16, np.float32, np.float64]
    arrs = np.array_split(x, sections, axis=axis)
    target_length = arrs[0].shape[axis]
    for i in range(len(arrs)):
        arrs[i] = pad_nan_to_target(arrs[i], target_length, axis=axis)
    return arrs

def take_per_row(A, indx, num_elem):
    all_indx = indx[:,None] + np.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:,None], all_indx]

def centerize_vary_length_series(x):
    prefix_zeros = np.argmax(~np.isnan(x).all(axis=-1), axis=1)
    suffix_zeros = np.argmax(~np.isnan(x[:, ::-1]).all(axis=-1), axis=1)
    offset = (prefix_zeros + suffix_zeros) // 2 - prefix_zeros
    rows, column_indices = np.ogrid[:x.shape[0], :x.shape[1]]
    offset[offset < 0] += x.shape[1]
    column_indices = column_indices - offset[:, np.newaxis]
    return x[rows, column_indices]

def data_dropout(arr, p):
    B, T = arr.shape[0], arr.shape[1]
    mask = np.full(B*T, False, dtype=np.bool)
    ele_sel = np.random.choice(
        B*T,
        size=int(B*T*p),
        replace=False
    )
    mask[ele_sel] = True
    res = arr.copy()
    res[mask.reshape(B, T)] = np.nan
    return res

def name_with_datetime(prefix='default'):
    now = datetime.now()
    return prefix + '_' + now.strftime("%Y%m%d_%H%M%S")

def init_dl_program(
    device_name,
    seed=None,
    use_cudnn=True,
    deterministic=False,
    benchmark=False,
    use_tf32=False,
    max_threads=None
):
    import torch
    if max_threads is not None:
        torch.set_num_threads(max_threads)  # intraop
        if torch.get_num_interop_threads() != max_threads:
            torch.set_num_interop_threads(max_threads)  # interop
        try:
            import mkl
        except:
            pass
        else:
            mkl.set_num_threads(max_threads)
        
    if seed is not None:
        random.seed(seed)
        seed += 1
        np.random.seed(seed)
        seed += 1
        torch.manual_seed(seed)
        
    if isinstance(device_name, (str, int)):
        device_name = [device_name]
    
    devices = []
    for t in reversed(device_name):
        t_device = torch.device(t) # 使用指定的GPU运行程序
        devices.append(t_device)
        if t_device.type == 'cuda':
            assert torch.cuda.is_available()
            torch.cuda.set_device(t_device)
            if seed is not None:
                seed += 1
                torch.cuda.manual_seed(seed)
    devices.reverse()
    torch.backends.cudnn.enabled = use_cudnn
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = use_tf32
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        
    return devices if len(devices) > 1 else devices[0]

