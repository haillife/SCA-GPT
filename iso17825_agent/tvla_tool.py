
import numpy as np

def compute_tvla_general(npy_file, num_traces, num_points, save_t_file=None):
    data = np.load(npy_file)

    assert data.shape == (num_points, num_traces), f"文件维度应为({num_points}, {num_traces})，但实际上是{data.shape}"

    half = num_traces // 2
    group1 = data[:, :half]
    group2 = data[:, half:]

    mean1 = np.mean(group1, axis=1)
    mean2 = np.mean(group2, axis=1)
    var1 = np.var(group1, axis=1, ddof=1)
    var2 = np.var(group2, axis=1, ddof=1)

    denominator = np.sqrt(var1 / half + var2 / half) + 1e-30
    t_values = (mean1 - mean2) / denominator
    t_values = np.abs((mean1 - mean2) / denominator)

    if save_t_file is None:
        save_t_file = npy_file.replace(".npy", "_tvla_result.npy")
    np.save(save_t_file, t_values)


    return t_values
