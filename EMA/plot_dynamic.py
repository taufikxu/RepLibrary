import torch
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image


def loss_function(x):
    center1 = torch.Tensor([0.3, 0.3]).float()
    loss1 = -5 * torch.exp(-torch.sum((center1 - x) ** 2, 1) * 10)

    center2 = torch.Tensor([-0.3, -0.3]).float()
    loss2 = -3 * torch.exp(-torch.sum((center2 - x) ** 2, 1) * 10)

    center3 = torch.Tensor([0.6, -0.5]).float()
    loss3 = -4 * torch.exp(-torch.sum((center3 - x) ** 2, 1) * 10)
    return loss1 + loss2 + loss3


def plot_image(initx, emax, XYValue=None):
    if initx is None:
        tx = 0
        ty = 0
        ex = 0
        ey = 0
    else:
        tx, ty = initx.data.numpy()[0, :]
        ex, ey = emax.data.numpy()[0, :]
    if XYValue is None:
        xrange = yrange = 1
        step = 200
        x = np.linspace(-xrange, xrange, step)
        y = np.linspace(-yrange, yrange, step)
        X, Y = np.meshgrid(x, y)
        inp = np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], 1)
        value = loss_function(inp).numpy().reshape(step, step)
    else:
        X, Y, value = XYValue

    fig, ax = plt.subplots()
    z_min, z_max = np.min(value), np.max(value)
    c = ax.pcolormesh(X, Y, value, vmin=z_min, vmax=z_max, shading="auto")
    ax.axis([X.min(), X.max(), Y.min(), Y.max()])
    fig.colorbar(c, ax=ax)
    plt.plot(tx, ty, "r.")
    plt.plot(ex, ey, "r*")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    im = Image.open(buf)
    plt.close()
    return im


xrange = yrange = 1
step = 200
x = np.linspace(-xrange, xrange, step)
y = np.linspace(-yrange, yrange, step)
X, Y = np.meshgrid(x, y)
inp = np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], 1)
value = loss_function(inp).numpy().reshape(step, step)
XYValue = [X, Y, value]

img_list = []
momentum_coe = 0.9
init_x = torch.autograd.Variable(torch.rand((1, 2)).float())
momentum = torch.rand((1, 2)).float() * 4
ema_x = torch.autograd.Variable(torch.rand((1, 2)).float())
init_x.data = torch.tensor([[0.1, -0.3]])
# ema_x.data = init_x.data

for i in range(30):
    init_x.requires_grad_(True)
    ema_x.requires_grad_(False)
    loss = loss_function(init_x)  # + torch.sum((init_x - ema_x) ** 2) * 10
    grad = torch.autograd.grad(loss, init_x)[0]
    momentum.data = momentum_coe * momentum.data + (1.0 - momentum_coe) * grad.data
    init_x.data = init_x.data - 0.01 * momentum
    ema_x.data = 0.9 * ema_x.data + 0.1 * init_x.data
    print(init_x, momentum)

    img_list.append(plot_image(init_x, ema_x, XYValue))

print(len(img_list))
img_list[0].save("dynamic.gif", save_all=True, append_images=img_list)
