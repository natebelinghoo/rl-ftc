# 数据可视化
import matplotlib.pyplot as plt

def plot_compare(xs1, xs2, refs, path):
    plt.figure(figsize=(10,5))
    plt.plot(xs1, label="正常控制")
    plt.plot(xs2, label="执行器故障")
    plt.plot(refs, '--', label="参考轨迹")
    plt.legend()
    plt.grid()
    plt.savefig(path, dpi=300)
    plt.close()
