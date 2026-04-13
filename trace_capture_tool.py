# 这个文件要实现的是，与智能卡通信，采波
# 与智能卡通信输入是：算法名称，明文长度，密钥长度，密文长度，端口号；输出无，支撑采波。
# 采波输入是：采样率，条数、点数、触发及量程、通道B量程，直流；输出是波形文件。


import numpy as np

import threading
from matplotlib import pyplot as plt

import ctypes
import tqdm
from picosdk.ps3000a import ps3000a as ps
from picosdk.functions import adc2mV, assert_pico_ok
from smartcard_tool import send_apdu_command

def trace_capture(npy_file, point_num=10000, trace_num=1000):
    """
    采波函数，支持断点续采
    :param start_idx:
    :param len:
    :return:
    """

    status = {}

    """标识示波器设备的句柄号"""
    chandle = ctypes.c_int16()

    """打开示波器，返回句柄号"""
    status["openunit"] = ps.ps3000aOpenUnit(ctypes.byref(chandle), None)

    """一些检测是否成功打开的判断"""
    try:
        assert_pico_ok(status["openunit"])
    except:

        # powerstate becomes the status number of openunit
        powerstate = status["openunit"]

        # If powerstate is the same as 282 then it will run this if statement
        if powerstate == 282:
            # Changes the power input to "PICO_POWER_SUPPLY_NOT_CONNECTED"
            status["ChangePowerSource"] = ps.ps3000aChangePowerSource(chandle, 282)
        # If the powerstate is the same as 286 then it will run this if statement
        elif powerstate == 286:
            # Changes the power input to "PICO_USB3_0_DEVICE_NON_USB3_0_PORT"
            status["ChangePowerSource"] = ps.ps3000aChangePowerSource(chandle, 286)
        else:
            raise

        assert_pico_ok(status["ChangePowerSource"])


    """开启通道A"""
    chARange = 8    # 6是1V [10mV, 20mV, 50mV, 100mV, 200mV, 500mV, 1V, 2V, 5V, 10V, 20V, 50V] 从0开始数
    handle = chandle
    channel = PS3000A_CHANNEL_A = 0
    enabled = 1
    coupling_type = PS3000A_DC = 1  # ["DC", "AC"]
    analogue_offset = 0
    status["setChA"] = ps.ps3000aSetChannel(handle, channel, enabled, coupling_type, chARange, analogue_offset)
    assert_pico_ok(status["setChA"])

    """开启通道B"""
    chBRange = 5    # 5是500mV [10mV, 20mV, 50mV, 100mV, 200mV, 500mV, 1V, 2V, 5V, 10V, 20V, 50V] 从0开始数
    handle = chandle
    channel = PS3000A_CHANNEL_B = 1
    enabled = 1
    coupling_type = PS3000A_DC = 1  # ["DC", "AC"]
    analogue_offset = 0
    status["setChB"] = ps.ps3000aSetChannel(handle, channel, enabled, coupling_type, chBRange, analogue_offset)
    assert_pico_ok(status["setChB"])

    """设置触发参数"""
    handle = chandle
    enable = 1
    source = ps3000A_channel_A = 0  # channels_4 = ["A", "B", "C", "D"]
    threshold = 3251 # 计算公式为(目标电压 / 量程) * 32512 #the ADC count at which the trigger will fire.
    direction = 2  # direction = ["ABOVE", "BELOW", "RISING", "FALLING", "RISING_OR_FALLING"]
    delay = 402500 #这里的计算公式就是，采样率*想要延时的时间  # if delay = 100, the scope would wait 100 sample periods before sampling. At a timebase of 500 MS/s, or 2 ns per sample, the total delay would then be 100 x 2 ns = 200 ns.
    autoTrigger_ms = 3000 # the number of milliseconds the device will wait if no trigger occurs. If this is set to zero, the scope device will wait indefinitely for a trigger.
    status["trigger"] = ps.ps3000aSetSimpleTrigger(handle, enable, source, threshold, direction, delay, autoTrigger_ms)
    assert_pico_ok(status["trigger"])

    """配置触发前采波点数，触发后采波点数，采样率"""
    preTriggerSamples = 0
    postTriggerSamples = point_num
    maxsamples = preTriggerSamples + postTriggerSamples
    # Creates converted types maxsamples
    cmaxSamples = ctypes.c_int32(maxsamples)

    timebase = 27  # 12.5M也就是80ns的采样间隔 #"1000MS/s": (1.e-9, 0), "500MS/s": (2.e-9, 1), "250MS/s": (4.e-9, 2), "125MS/s": (8.e-9, 3), "62.5MS/s": (16.e-9, 4),"25MS/s": (40.e-9, 7), "12.5MS/s": (80.e-9, 12), "5MS/s": (200.e-9, 27), "2.5MS/s": (400.e-9, 52), "1MS/s": (1000.e-9, 127)

    """这个函数根据设置的参数查询采样率和最大采样的点数"""
    handle = chandle
    no_sample = maxsamples
    timeIntervalns = ctypes.c_float()
    returnedMaxSamples = ctypes.c_int16()
    TimeIntervalNanoseconds = ctypes.byref(timeIntervalns)
    MaxSamples = ctypes.byref(returnedMaxSamples)
    Segement_index = 0
    status["GetTimebase"] = ps.ps3000aGetTimebase2(handle, timebase, no_sample, TimeIntervalNanoseconds, 1, MaxSamples, Segement_index)
    assert_pico_ok(status["GetTimebase"])
    # print(f"采样间隔: {timeIntervalns.value:.2f} ns")
    # print(f"最多采样点数: {returnedMaxSamples.value}")


    # 用于保存所有波形的矩阵：point_num 行，trace_num 列
    all_waveforms = np.zeros((point_num, trace_num), dtype=np.int16)

    for trace_idx in tqdm.trange(trace_num, desc="trace capturing"):
        rej_num = 5
        for rej_idx in range(rej_num):
            # Step 1: 运行采样
            status["runblock"] = ps.ps3000aRunBlock(chandle, preTriggerSamples, postTriggerSamples, timebase, 1, None, 0, None, None)
            assert_pico_ok(status["runblock"])

            # Step 2: 发 APDU
            apdu_input = [
                0x00, 0x11, 0x00, 0x00, 0x20,
                0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88,
                0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00,
                0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88,
                0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00
            ]
            try:
                response = send_apdu_command(apdu_input)
            except Exception as e:
                print(f"APDU 指令失败: {e}")
                continue

            # Step 3: 设置缓冲区（A, B）
            """创建buffer用于保存返回数据"""
            bufferAMax = np.empty(maxsamples, dtype=np.dtype('int16'))
            bufferBMax = np.empty(maxsamples, dtype=np.dtype('int16'))

            """把A通道的数据返回到bufferA里面"""
            handle = chandle
            source = ps3000A_channel_A = 0
            buffer_length = maxsamples
            segment_index = 0
            ratio_mode = ps3000A_Ratio_Mode_None = 0
            status["SetDataBuffers"] = ps.ps3000aSetDataBuffer(handle, source, bufferAMax.ctypes.data, buffer_length, segment_index, ratio_mode)
            assert_pico_ok(status["SetDataBuffers"])

            """把B通道的数据返回到bufferB里面"""
            handle = chandle
            source = ps3000A_channel_B = 1
            buffer_length = maxsamples
            segment_index = 0
            ratio_mode = ps3000A_Ratio_Mode_None = 0
            status["SetDataBuffers"] = ps.ps3000aSetDataBuffer(handle, source, bufferBMax.ctypes.data, buffer_length, segment_index, ratio_mode)
            assert_pico_ok(status["SetDataBuffers"])

            # Step 4: 等待采样完成取回波形数据

            # Creates a overlow location for data
            overflow = ctypes.c_int16()
            # Creates converted types maxsamples
            cmaxSamples = ctypes.c_int32(maxsamples)

            """等待采波结束"""
            ready = ctypes.c_int16(0)
            check = ctypes.c_int16(0)
            while ready.value == check.value:
                status["isReady"] = ps.ps3000aIsReady(chandle, ctypes.byref(ready))

            """从示波器取回采波数据"""
            status["GetValuesBulk"] = ps.ps3000aGetValues(chandle, 0, ctypes.byref(cmaxSamples), 1, 0, 0, ctypes.byref(overflow))
            assert_pico_ok(status["GetValuesBulk"])

            # 保存当前波形

            # Converts ADC to mV
            # 创建 c_int16 对象来接收返回值
            maxADC = ctypes.c_int16()
            # 传入句柄和地址，获取最大 ADC 值
            status["maxADC"] = ps.ps3000aMaximumValue(chandle, ctypes.byref(maxADC))
            assert_pico_ok(status["maxADC"])

            # print(f"最大 ADC 值: {maxADC.value}")
            adc2mVChAMax = adc2mV(bufferAMax, chARange, maxADC)

            adc2mVChBMax = adc2mV(bufferBMax, chBRange, maxADC)

            all_waveforms[:, trace_idx] = adc2mVChBMax[:cmaxSamples.value]
            # plt.plot(bufferAMax)
            # plt.plot(bufferBMax, linewidth=0.9, alpha=0.9)
            # plt.show()

            break  # 成功就退出再采波

    # Creates the time data
    time = np.linspace(0, (cmaxSamples.value - 1) * timeIntervalns.value, cmaxSamples.value)
    # Plots the data from channel A onto a graph
    plt.plot(time, adc2mVChAMax)
    print('test')
    plt.xlabel('Time (ns)')
    plt.ylabel('Voltage (mV)')
    plt.show()

    # Stops the scope
    status["stop"] = ps.ps3000aStop(chandle)
    assert_pico_ok(status["stop"])
    # Closes the unit
    status["close"] = ps.ps3000aCloseUnit(chandle)
    assert_pico_ok(status["close"])

    # 保存最终大矩阵
    np.save(npy_file, all_waveforms)
    print(f"保存成功: {npy_file}, 形状: {all_waveforms.shape}")
    # 获取第一条波形（注意：每一列是一条波）
    waveform = all_waveforms[:, 0]

    # Plots the data from channel B onto a graph
    plt.plot(time, adc2mVChBMax)
    print('test')
    plt.xlabel('Time (ns)')
    plt.ylabel('Voltage (mV)')
    plt.show()


import ctypes
from picosdk.ps3000a import ps3000a as ps
from picosdk.functions import assert_pico_ok

def pico_config_info(timebase=27, chARange=8, chBRange=5, trigger_source=0, threshold=3251, delay=412500):
    status = {}
    chandle = ctypes.c_int16()

    # 打开设备
    status["openunit"] = ps.ps3000aOpenUnit(ctypes.byref(chandle), None)
    try:
        assert_pico_ok(status["openunit"])
    except:
        powerstate = status["openunit"]
        if powerstate in [282, 286]:
            status["ChangePowerSource"] = ps.ps3000aChangePowerSource(chandle, powerstate)
            assert_pico_ok(status["ChangePowerSource"])
        else:
            raise

    # 设置通道 A
    status["setChA"] = ps.ps3000aSetChannel(chandle, 0, 1, 1, chARange, 0)
    # 设置通道 B
    status["setChB"] = ps.ps3000aSetChannel(chandle, 1, 1, 1, chBRange, 0)

    assert_pico_ok(status["setChA"])
    assert_pico_ok(status["setChB"])

    # 获取采样间隔
    timeIntervalns = ctypes.c_float()
    returnedMaxSamples = ctypes.c_int32()
    status["GetTimebase"] = ps.ps3000aGetTimebase2(
        chandle, timebase, 0, ctypes.byref(timeIntervalns), 1,
        ctypes.byref(returnedMaxSamples), 0
    )
    assert_pico_ok(status["GetTimebase"])

    # 获取最大 ADC 值
    maxADC = ctypes.c_int16()
    status["maxADC"] = ps.ps3000aMaximumValue(chandle, ctypes.byref(maxADC))
    assert_pico_ok(status["maxADC"])

    # 打印结果
    print("\n📋 PicoScope 当前配置状态：")
    print("——————————————————————————————————————")
    print(f"通道 A 量程等级 (索引): {chARange}")
    print(f"通道 B 量程等级 (索引): {chBRange}")
    print(f"触发通道: {'A' if trigger_source == 0 else 'B'}")
    print(f"触发方向: RISING (代码 2)")
    print(f"ADC 最大值 (maxADC): {maxADC.value}")
    print(f"设定的 ADC 阈值: {threshold}")
    print(f"估算触发电压: {round((threshold / maxADC.value) * 1000, 2)} mV")
    print(f"采样时间间隔: {timeIntervalns.value:.2f} ns")
    print(f"采样率: {round(1e9 / timeIntervalns.value, 2)} S/s")
    print(f"触发延迟: {delay} 点 × {timeIntervalns.value:.2f} ns = {delay * timeIntervalns.value / 1e6:.2f} ms")
    print("——————————————————————————————————————")

    # 关闭设备
    ps.ps3000aCloseUnit(chandle)




# pico参数
# pico_config_info()

trace_capture("AES_7816.npy", point_num=240000, trace_num=1)
