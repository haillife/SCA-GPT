import numpy as np

def compute_timing_difference(
    npy_file: str,
    num_traces: int,
    num_points: int,
    sampling_rate: float,
    clock_freq: float,
    voltage_threshold: float = None
) -> float:
    """
    计算固定与随机输入下的平均穿越时间差 |T1 - T2|，基于电压阈值点。

    :param npy_file: 包含波形的 .npy 文件路径，格式应为 shape=(num_points, num_traces)
    :param num_traces: 采样波形的总数量，需为偶数（固定+随机）
    :param num_points: 每条波形的点数
    :param sampling_rate: 采样率，单位为样本数/秒（Hz）
    :param clock_freq: 时钟频率，单位为Hz
    :param voltage_threshold: 电压阈值（可选），若未提供，将使用第一条固定波形的 (max + min)/2
    :return: 平均穿越时间差的绝对值（单位：秒）
    """
    data = np.load(npy_file)

    assert data.shape == (num_points, num_traces), f"Expected shape=({num_points}, {num_traces}), but got {data.shape}"
    assert num_traces % 2 == 0, "num_traces must be even (fixed + random)"

    half = num_traces // 2
    fixed_traces = data[:, :half]
    random_traces = data[:, half:]

    # 如果未提供阈值，默认取第一条固定波形的 (max + min)/2
    if voltage_threshold is None:
        first_waveform = fixed_traces[:, 0]
        voltage_threshold = (np.max(first_waveform) + np.min(first_waveform)) / 2
        print(f"[INFO] voltage_threshold auto-calculated as: {voltage_threshold:.6f}")

    def find_first_crossing_indices(traces, threshold):
        """返回每条波形第一个穿越阈值的索引。未穿越的返回 num_points。"""
        crossings = np.argmax(traces >= threshold, axis=0)
        # 修正全 False 时 argmax 返回 0 的问题
        mask = np.all(traces < threshold, axis=0)
        crossings[mask] = num_points
        return crossings

    fixed_indices = find_first_crossing_indices(fixed_traces, voltage_threshold)
    random_indices = find_first_crossing_indices(random_traces, voltage_threshold)

    time_fixed = np.mean(fixed_indices) / sampling_rate
    time_random = np.mean(random_indices) / sampling_rate

    delta_time = abs(time_fixed - time_random)

    # --- 最终判定逻辑 ---
    clock_period = 1 / clock_freq
    is_safe = delta_time < clock_period
    judgment = "✅ Secure" if is_safe else "❌ Insecure"

    # --- 打印最终报告 ---
    print("🕒 Timing Analysis Result:")
    print(f"→ Absolute time difference |T1 - T2|: {delta_time:.9f} seconds")
    print(f"→ Clock period: {clock_period:.9f} seconds")
    print(f"→ Final judgment: {judgment}")

    return delta_time

