import numpy as np
import matplotlib.pyplot as plt
import math
from R4ABM2_8bits import *

np.random.seed(0)


def dec2bin_8bits(data):  # 注意，因为整版代码都是从verilog翻译的，硬件中的索引与软件索引是相反的
    b = np.zeros(8, dtype=bool)  # 所以这个函数生成的二进制格式是反序的,e.g. 8 = [00010000]
    if data >= 0:
        b[7] = 0
    else:
        b[7] = 1
        data = data + 256
    if 1 > data > -1:
        data_b = np.round(data * 128)
    else:
        data_b = data

    # b = format(int(b), '08b')
    b[0] = data_b % 2
    b[1] = (data_b // 2) % 2
    b[2] = (data_b // 4) % 2
    b[3] = (data_b // 8) % 2
    b[4] = (data_b // 16) % 2
    b[5] = (data_b // 32) % 2
    b[6] = (data_b // 64) % 2
    # print('dec2bin:', b)
    return b


def SC_4bits_signed_m(ai_sample, w_sample, n):
    bitwidth_str = '0' + str(n) + 'b'

    if w_sample > 127:  # [-1, 1)
        w_sample = 127
    elif w_sample < -128:
        w_sample = -128
    elif 1 > w_sample >= -1:
        w_sample = round(w_sample * 128)
    else:
        w_sample = w_sample

    if ai_sample > 127:
        ai_sample = 127
    elif ai_sample < -128:  # [-128, 127)
        ai_sample = -128
    else:
        ai_sample = ai_sample

    if w_sample < 0:
        b = w_sample + 256  # format函数只支持正整数的格式转换，所以先+256表示成无符号数
        w = format(int(b), '08b')
    else:
        w = format(int(w_sample), '08b')
    # 大于零时符号位是0，所以也可以直接累加进来不影响最终结果的

    if ai_sample < 0:
        a = ai_sample + 256
        a = format(int(a), '08b')
    else:
        a = format(int(ai_sample), '08b')

    # -------------------------------- ���ʳ˷� -------------------------------------
    result_1 = -int(w[0]) * ai_sample * pow(2, n - 1)
    result_2 = 0
    for i in range(n):
        for j in range(n - 1):
            if i + j > n - 1:
                if i == n - 1:
                    result_2 += -(int(a[n - 1 - i]) & int(w[n - 1 - j])) * pow(2, i + j)
                else:
                    result_2 += (int(a[n - 1 - i]) & int(w[n - 1 - j])) * pow(2, i + j)
            elif i + j == n - 1:
                if i == n - 1:
                    result_2 += -(int(a[n - 1 - i]) & int(w[n - 1 - j])) * pow(2, n - 1)
                else:
                    result_2 += (int(a[n - 1 - i]) & int(w[n - 1 - j])) * pow(2, n)
            else:
                result_2 = result_2
    result = result_1 + result_2
    return result


def SC_4bits_signed_n(ai_sample, w_sample, n):  # ai_sample 和 w_sample必须是整数
    bitwidth_str = '0' + str(n) + 'b'

    w_sample = round(w_sample * 128)
    # ai_sample = round(ai_sample * 128)    # (-1, 1)

    ai_sample = round(ai_sample)  # (-128, 127)
    # print(w_sample, ai_sample)

    if ai_sample < 0:
        a = format(ai_sample + pow(2, n), bitwidth_str)
    else:
        a = format(ai_sample, bitwidth_str)
    if w_sample < 0:
        w = format(w_sample + pow(2, n), bitwidth_str)
    else:
        w = format(w_sample, bitwidth_str)
    # -------------------------------- 概率乘法 -------------------------------------
    result_1 = -int(w[0]) * ai_sample * pow(2, n - 1)
    result_2 = 0
    for i in range(n):
        for j in range(n - 1):
            if i + j > n - 1:
                if i == n - 1:
                    result_2 += -(int(a[n - 1 - i]) & int(w[n - 1 - j])) * pow(2, i + j)
                else:
                    result_2 += (int(a[n - 1 - i]) & int(w[n - 1 - j])) * pow(2, i + j)
            elif i + j == n - 1:
                if i == n - 1:
                    result_2 += -(int(a[n - 1 - i]) & int(w[n - 1 - j])) * pow(2, n - 1)
                else:
                    result_2 += (int(a[n - 1 - i]) & int(w[n - 1 - j])) * pow(2, n)
            else:
                result_2 = result_2
    result = result_1 + result_2
    # return result / pow(2, 2*n-2)      #(-1, 1)
    return result / pow(2, n - 1)  # (-128, 127)


def calculate_snr(signal, noise):
    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = np.mean(np.abs(noise) ** 2)

    snr = 10 * np.log10(signal_power / noise_power)
    return snr


#     test = 0


if __name__ == '__main__':
    # ======= 高斯分布数据 (-1,1) (-128, 127) begin =========
    w_f = np.random.normal(loc=0.0, scale=0.3, size=65536)  # 生成随机正态分布数。
    ia_f = np.random.normal(loc=0.0, scale=41.3, size=65536)
    w_f = w_f / max(abs(w_f))
    ia_f = ia_f * 127 / max(abs(ia_f))
    print(max(abs(w_f)), max(abs(ia_f)))
    w = np.round(w_f * 128) / 128
    ia = np.round(ia_f)
    w_6bit = np.round(w_f * 32) / 32
    # ia_6bit = np.zeros(65536, dtype=int)
    ia_6bit = np.round(ia_f)
    print(max(abs(w_f)), max(abs(ia_f)))

    str_ia = 0
    signal_6bit = []
    error_6bit = []
    error_6bit_relative = []
    error_6bit_abs_relative = []
    error_6bit_SNR = []

    signal_8bit = []
    error_8bit = []
    error_8bit_relative = []
    error_8bit_abs_relative = []
    error_8bit_SNR = []

    signal_spsc = []
    error = []
    error_size = []  # ME
    error_abs_size = []  # MAE
    error_relative = []  # MRE
    error_spsc_SNR = []
    for i in range(256 * 256):
        # ======= 高斯分布数据 (-1,1) (-128, 127) begin =========
        if ia[i] > 32:
            ia_6bit[i] = 32
        elif ia[i] < -32:
            ia_6bit[i] = -32
        else:
            if ia[i] < 0:
                str_ia = format(round(ia[i]) + pow(2, 6), '06b')
            else:
                str_ia = format(round(ia[i]), '06b')
            ia_6bit[i] = int(str_ia[0]) * (-32) + int(str_ia[1]) * (16) + int(str_ia[2]) * 8 + int(str_ia[3]) * 4 + int(
                str_ia[4]) * 2 + int(str_ia[5]) * 1
        exact_re = w_f[i] * ia_f[i]  # 8-bit量化乘法
        # print('w_f:', w_f[i], 'ia_f:', ia_f[i])
        fixed_8bit = w[i] * ia[i]  # 8-bit量化乘法
        # print('w:', w[i], 'ia:', ia[i])
        fixed_6bit = w_6bit[i] * ia_6bit[i]
        # print('w_6bit:', w_6bit[i], 'ia_6bit:', ia_6bit[i])
        sto_re = SC_4bits_signed_n(ia[i], w[i], 8)
        # ======= 高斯分布数据 (-1,1) (-128, 127) end =========
        if (exact_re == sto_re):
            judge = 0
        else:
            judge = 1
        signal_spsc.append(sto_re)
        error.append(judge)
        error_size.append(exact_re - sto_re)  # Mean Error
        error_abs_size.append(abs(exact_re - sto_re))  # Mean Absolute Error

        signal_6bit.append(fixed_6bit)
        error_6bit.append(fixed_6bit - exact_re)
        error_6bit_abs_relative.append(abs(fixed_6bit - exact_re))  # Mean Absolute Error

        signal_8bit.append(fixed_8bit)
        error_8bit.append(fixed_8bit - exact_re)
        error_8bit_abs_relative.append(abs(fixed_8bit - exact_re))  # Mean Absolute Error

        if exact_re == 0:
            error_relative.append((exact_re - sto_re))  # Mean Relative Error(MRE)
        else:
            error_relative.append((exact_re - sto_re) / exact_re)
        # 6-bit计算
        if exact_re == 0:
            error_6bit_relative.append((fixed_6bit - exact_re))  # Mean Relative Error(MRE)

        else:
            error_6bit_relative.append((fixed_6bit - exact_re) / exact_re)
        # 8-bit 定点计算
        if exact_re == 0:
            error_8bit_relative.append((fixed_8bit - exact_re))  # Mean Relative Error(MRE)

        else:
            error_8bit_relative.append((fixed_8bit - exact_re) / exact_re)

    # SNR 计算
    SNR_spsc = calculate_snr(signal_spsc, error_size)
    SNR_6bit = calculate_snr(signal_6bit, error_6bit)
    SNR_8bit = calculate_snr(signal_8bit, error_8bit)

    Ex_6bit = np.mean(error_6bit)
    # Ex_6bit_relative = np.mean(error_6bit_relative)
    # Ex_6bit_abs_relative = np.mean(error_6bit_abs_relative)
    Ex_6bit_relative = sum(error_6bit_relative)/65536
    Ex_6bit_abs_relative = sum(error_6bit_abs_relative)/65536
    variance_6bit = np.var(error_6bit)
    standard_6bit = np.std(error_6bit)
    ERMAX_6bit = max(error_6bit)  # Maximun Error
    # print(error_6bit.index(max(error_6bit)))
    ERMIN_6bit = min(error_6bit)  # Min Error
    # print(error_6bit.index(min(error_6bit)))
    MAAE_6bit = max(abs(ERMAX_6bit), abs(ERMIN_6bit))
    ERMAX_6bit_counter = error_6bit.count(max(error_6bit))
    ERMIN_6bit_counter = error_6bit.count(min(error_6bit))
    MSE_6bit = sum([num ** 2 for num in error_6bit]) / (256 * 256)

    print('最大误差：', ERMAX_6bit, '最大误差出现次数(MEO)：', ERMAX_6bit_counter)
    print('最小误差：', ERMIN_6bit, '最小误差出现次数(MEO)：', ERMIN_6bit_counter)
    print('最大绝对误差(MAAE)：', MAAE_6bit)
    print('6bit平均误差(MEE):', Ex_6bit)
    print('6bit平均绝对误差(MEAE):', Ex_6bit_abs_relative)
    print('6bit平均相对误差(MRE):', Ex_6bit_relative)
    print('6bit均方误差(MSE):', MSE_6bit)
    print('6bit方差:', variance_6bit)
    print('6bit标准差:', standard_6bit)
    print('6bit SNR', SNR_6bit)
    print('\n')
test = 0
