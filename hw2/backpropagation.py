# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torch
import numpy as np
import math


def f(x):
    return 1/(1+torch.exp(-x))


def main():
    # Use a breakpoint in the code line below to debug your script.
    # Press Ctrl+F8 to toggle the breakpoint.
    wx = [[1.0, 3.0]]
    wx = torch.tensor(wx, requires_grad=True)
    wxt = torch.transpose(wx, 0, 1)
    # print(wxt.requires_grad)
    wxt = wxt.double()

    wyt = torch.tensor([[3.0, 1.0]], requires_grad=True)
    wyt = wyt.double()

    wht = torch.tensor([[1.0, 2.0], [3.0, 1.0]], requires_grad=True)
    wht = wht.double()

    bh = torch.tensor([[1.0], [2.0]], requires_grad=True)
    bh = bh.double()

    by = torch.tensor([1.0], requires_grad=True)
    by = by.double()

    h0 = torch.tensor([[2.0], [1.0]])
    h0 = h0.double()

    x1 = torch.tensor([-1])
    x1 = x1.double()

    x2 = torch.tensor([0])
    x2 = x2.double()

    # q1 = wxt*x1 + torch.matmul(wht, h0) + bh
    q1 = wxt*x1 + torch.matmul(wht, h0) + bh
    # print(q1)
    h1 = f(q1)
    print("h1 = ")
    print(h1)
    y_hat1 = f(torch.matmul(wyt, h1)+by)
    print("y_hat1 = ")
    print(y_hat1)
    q2 = wxt*x2 + torch.matmul(wht, h1) + bh
    print("q2 = ")
    print(q2)
    h2 = f(q2)
    print("h2 = ")
    print(h2)
    y_hat2 = f(torch.matmul(wyt, h2)+by)
    print("y_hat2 = ")
    print(y_hat2)

    y1 = [[0]]
    y1 = torch.tensor(y1)
    y1 = y1.double()

    y2 = [[1]]
    y2 = torch.tensor(y2)
    y2 = y2.double()

    loss = (y_hat1 - y1)**2 + (y_hat2 - y2)**2
    print("loss = ")
    print(loss)

    wyt.retain_grad()
    wht.retain_grad()
    wxt.retain_grad()
    by.retain_grad()
    bh.retain_grad()
    loss.backward()
    print("Gradient of wyt: ")
    print(wyt.grad)

    print("Gradient of wht: ")
    print(wht.grad)

    print("Gradient of wxt: ")
    print(wxt.grad)

    print("Gradient of by: ")
    print(by.grad)

    print("Gradient of bh: ")
    print(bh.grad)

    lr = 0.01
    by -= lr*by.grad
    print("Updated by")
    print(by)

    wht -= lr*wht.grad
    print("updated wht")
    print(wht)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
