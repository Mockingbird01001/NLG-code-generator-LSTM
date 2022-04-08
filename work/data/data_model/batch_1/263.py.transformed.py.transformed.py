
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
def run():
    data =
    total_width = 0.8
    data_x = data_x.split()
    data_y = [i_str[4:] for i_str in data_y.split('\n')[1:-1]]
    data = [i_str.split() for i_str in data.split('\n')[1:-1]]
    ary = np.array(data, dtype=np.int)
    ary = ary.reshape((len(data_y), len(data_x), 2))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    grep_id = np.array((0, 1))
    title = 'ISAC and IAC simple task'
    data_x = np.array(data_x)[grep_id]
    ary = ary[:, grep_id]
    colors = np.array(colors)[grep_id]
    labels = data_x
    n_label = len(labels)
    n_bars = len(data_y)
    bar_width = total_width / n_bars
    bars = []
    bars_width = (n_label + 1) * bar_width
    fig, ax = plt.subplots()
    for i, name in enumerate(data_y):
        means = ary[i, :, 0]
        errors = ary[i, :, 1]
        x_offset = i * bars_width
        for j in range(n_label):
            loc = j * bar_width + x_offset
            bar = ax.bar(loc, means[j], yerr=errors[j],
                         width=bar_width, color=colors[j % len(colors)])
            if i == 0:
                bars.append(bar)
    ax.legend(bars, labels, loc='upper center')
    x_loc = np.arange(n_bars) * bars_width + (bars_width / 2 - bar_width)
    x_tricks = data_y
    plt.xticks(x_loc, x_tricks, rotation=20)
    plt.title(title)
    plt.grid()
    plt.gcf().subplots_adjust(bottom=0.2)
    save_path = f'comparison_target_reward_{title}.pdf'
    plt.savefig(save_path, dpi=200)
    print(save_path)
def plot__multi_error_bars(ary_avg, ary_std=None, labels0=None, labels1=None, title='multi_error_bars'):
    if ary_std is None:
        ary_std = np.empty_like(ary_avg)
        ary_std[:, :] = None
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    lab0_len = len(labels0)
    lab1_len = len(labels1)
    bar_width = 1 / lab0_len
    bars = []
    bars_width = (lab1_len + 1) * bar_width
    fig, ax = plt.subplots()
    for i in range(lab0_len):
        avg = ary_avg[i, :]
        std = ary_std[i, :]
        x_offset = i * bars_width
        for j in range(lab1_len):
            x1_loc = j * bar_width + x_offset
            bar = ax.bar(x1_loc, avg[j], yerr=std[j],
                         width=bar_width, color=colors[j % len(colors)])
            if i == 0:
                bars.append(bar)
    ax.legend(bars, labels1, loc='upper right')
    '''if the name of x-axis is too long, adjust the rotation and bottom'''
    x0_loc = np.arange(lab0_len) * bars_width - bar_width + bars_width / 2
    plt.xticks(x0_loc, labels0, rotation=15)
    plt.gcf().subplots_adjust(bottom=0.1)
    plt.title(title)
    plt.grid()
    plt.show()
def plot__error_std(ys, xs=None, k=8):
    if xs is None:
        xs = np.arange(ys.shape[0])
    ys_pad = np.pad(ys, pad_width=(k, 0), mode='edge')
    ys_avg = list()
    ys_std = list()
    for i in range(len(ys)):
        ys_part = ys_pad[i:i + k]
        ys_avg.append(ys_part.mean())
        ys_std.append(ys_part.std())
    plt.plot(xs, ys, color='royalblue')
    plt.plot(xs, ys_avg, color='lightcoral')
    ys_avg = np.array(ys_avg)
    ys_std = np.array(ys_std)
    plt.fill_between(xs, ys_avg - ys_std, ys_avg + ys_std, facecolor='lightcoral', alpha=0.3)
    plt.show()
def plot__error_plot_round(ys, xs=None, k=8):
    if xs is None:
        xs = np.arange(ys.shape[0])
    ys_pad = np.pad(ys, pad_width=(k // 2, k // 2), mode='edge')
    ys_avg = list()
    ys_std1 = list()
    ys_std2 = list()
    for i in range(len(ys)):
        ys_part = ys_pad[i:i + k]
        avg = ys_part.mean()
        ys_avg.append(avg)
        ys_std1.append((ys_part[ys_part > avg] - avg).mean())
        ys_std2.append((ys_part[ys_part <= avg] - avg).mean())
    plt.plot(xs, ys, color='royalblue')
    plt.plot(xs, ys_avg, color='lightcoral')
    ys_avg = np.array(ys_avg)
    ys_std1 = np.array(ys_std1)
    ys_std2 = np.array(ys_std2)
    plt.fill_between(xs, ys_avg + ys_std1, ys_avg + ys_std2, facecolor='lightcoral', alpha=0.3)
    plt.show()
def run_demo():
    xs = np.linspace(0, 2, 64)
    ys = np.sin(xs)
    ys[rd.randint(64, size=8)] = 0
    plot__error_plot_round(ys, xs, k=8)
if __name__ == '__main__':
    run_demo()
