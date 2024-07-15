'''
  这版初始代码是处理权值为小数，激活值为整数[-128, 127)
  这版是自己修改了权值和激活值范围判断条件、删掉了异或门的错误化简操作
'''
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)


def sp_sc_multi(ia_data, w_data):
    flag_ia = 0
    flag_w = 0
    if w_data > 31:  # [-1, 1)
        w_data = 31
    elif w_data < -32:
        w_data = -32
    elif 1 > w_data >= -1:
        flag_w = 1
        w_data = round(w_data * 32)
    else:
        w_data = w_data

    if ia_data > 31:
        ia_data = 31
    elif ia_data < -32:  # [-128, 127)
        ia_data = -32
    # elif 1 > ia_data >= -1:
    #     ia_data = round(ia_data * 128)
    #     flag_ia = 1
    else:
        ia_data = ia_data

    w_h = 0
    if w_data < 0:
        b = w_data + 64  # format函数只支持正整数的格式转换，所以先+256表示成无符号数
        w_b = format(int(b), '06b')
        for i in range(0, 3):
            if int(w_b[i]) == 1:
                w_h += 2 ** (2 - i)
        w_h = w_h - 8  # 转回十进制数
    else:
        w_b = format(int(w_data), '06b')
        for i in range(1, 3):
            if int(w_b[i]) == 1:
                w_h += 2 ** (2 - i)  # 大于零时符号位是0，所以也可以直接累加进来不影响最终结果的
    # print("8bits权值数据二进制表征：", w_b)

    ia_h = 0
    if ia_data < 0:
        a = ia_data + 64
        ia_b = format(int(a), '06b')
        for i in range(0, 4):
            if int(ia_b[i]) == 1:
                ia_h += 2 ** (3 - i)
        ia_h = ia_h - 8
    else:
        ia_b = format(int(ia_data), '06b')
        for i in range(1, 3):
            if int(ia_b[i]) == 1:
                ia_h += 2 ** (2 - i)

    bw_and_re0 = np.zeros([4, 1])
    bw_and_re1 = np.zeros([4, 1])

    bw_and_re0[0] = (int(w_b[0]) ^ int(w_b[1])) & int(ia_b[2])
    bw_and_re0[1] = (int(w_b[0]) ^ int(w_b[2])) & int(ia_b[2])
    bw_and_re0[2] = (int(w_b[0]) ^ int(w_b[1])) & int(ia_b[1])
    bw_and_re0[3] = 0

    bw_and_re1[0] = (int(ia_b[0]) ^ int(ia_b[1])) & int(w_b[2])
    bw_and_re1[1] = (int(ia_b[0]) ^ int(ia_b[2])) & int(w_b[2])
    bw_and_re1[2] = (int(ia_b[0]) ^ int(ia_b[1])) & int(w_b[1])
    bw_and_re1[3] = 0

    result0 = 0
    result1 = 0

    for i in range(4):
        result0 = result0 + bw_and_re0[i]
        result1 = result1 + bw_and_re1[i]

    result0 = 2 * result0

    result1 = 2 * result1

    if int(w_b[0]) & int(ia_b[3]) == 1:
        result0 = result0 + 1
    else:
        result0 = result0

    if int(ia_b[0]) & int(w_b[3]) == 1:
        result1 = result1 + 1
    else:
        result1 = result1

    if w_data < 0:
        result0 = -result0

    if ia_data < 0:
        result1 = -result1

    # if flag_ia == 1 & flag_w == 1:
    #     sto_re = (2 * w_h * ia_h + result0 + result1)/128  # 精度补偿，所以2只乘在最高分段
    # elif flag_ia == 0 & flag_w == 0:
    #     sto_re = (2 * w_h * ia_h + result0 + result1)*128
    # else:
    sto_re = (2 * w_h * ia_h + (result0 + result1)/2)
    # sto_re = 2 * w_h * ia_h + result0 + result1
    # print("8bits概率乘法的结果：", sto_re)
    return sto_re[0]


if __name__ == '__main__':
    w_f = np.random.normal(loc=0.0, scale=0.3, size=4096)  # 生成随机正态分布数。
    ia_f = np.random.normal(loc=0.0, scale=10.66, size=4096)
    w = np.round(w_f * 32)/32
    ia = np.round(ia_f)

    error = []
    error_size = []                 # ME
    error_abs_size = []             # MAE
    error_relative = []             # MRE
    for i in range(64*64):
        exact_re = np.round(w[i] * ia[i])
        sto_re = np.round(sp_sc_multi(ia[i], w[i]))
        # exact_re = w[i] * ia[i]
        # sto_re = sp_sc_multi(w[i], ia[i])
        if (exact_re == sto_re):
            judge = 0
        else:
            judge = 1
        error.append(judge)
        error_size.append(exact_re - sto_re)                    # Mean Error
        error_abs_size.append(abs(exact_re - sto_re))           # Mean Absolute Error
        if exact_re == 0:
            error_relative.append((exact_re - sto_re))  # Mean Relative Error(MRE)
        else:
            error_relative.append((exact_re - sto_re) / exact_re)   # Mean Relative Error(MRE)
    # print("8bits精确计算结果：", c)
    counter = error.count(0)
    print('right rate: {:.2f}%'.format(counter / (64 * 64) * 100))
    ERMAX = max(error_size)                 # Maximun Error
    ERMIN = min(error_size)                 # Min Error
    ERMAX_counter = error_size.count(ERMAX)
    ERMIN_counter = error_size.count(ERMIN)
    print('最大误差：', ERMAX, '最大误差出现次数(MEO)：', ERMAX_counter)
    print('最小误差：', ERMIN, '最小误差出现次数(MEO)：', ERMIN_counter)

    EO_counter = error.count(1)             # EO
    print('误差出现次数(EO)：', EO_counter, '总运算次数', 64*64)

    MAX_Abs_err = max(error_abs_size)       # Maximum Absolute Error(MAAE)
    print('最大绝对误差(MAAE)', MAX_Abs_err)

    Ex = np.mean(error_size)
    Ex_abs = np.mean(error_abs_size)
    Ex_relative = np.mean(error_relative)
    print('平均误差(MEE):', Ex)
    print('平均绝对误差(MEAE):', Ex_abs)
    print('平均相对误差(MRE):', Ex_relative)
    variance = np.var(error_size)
    print('方差:', variance)
    standard = np.std(error_size)
    print('标准差:', standard)
    # MSE = np.mean(error_abs_size)*np.mean(error_abs_size) / 4096
    MSE = sum([num ** 2 for num in error_size]) / (256 * 256)
    print('平均平方误差(MSE):', MSE)
