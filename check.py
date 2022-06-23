import threading
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import torch

from classes import PATH, r
import data

r.load_state_dict(torch.load(PATH))
r.eval()

SCORE = 0


def run():

    global SCORE

    for i, tup in enumerate(hum_gen):
        img, target = tup
        input = next(bot_gen)[0]
        # DISP.set_data(img)
        # data.FIG.canvas.manager.set_window_title(target)
        # data.FIG.canvas.draw()
        ans = evaluate(input, target)
        print("[%s / 60000]" % i)
        if ans == target:
            SCORE += 1
    print("Success Rate:", str(round((SCORE / 60000) * 100, 3)))


def on_click(event):
    input = next(bot_gen)[0]
    img, target = next(hum_gen)
    DISP.set_data(img)
    data.FIG.canvas.manager.set_window_title(target)
    data.FIG.canvas.draw()
    evaluate(input, target)


def evaluate(ten: torch.Tensor, target: str):
    out = r.forward(ten).detach()
    print("Target:", target)
    ret = 0
    highest = 0
    for array in out:
        for i, item in enumerate(array):
            item = float(item)
            if item != 0:
                if item > highest:
                    highest = item
                    ret = i
                print("\nGuess: %s" % i, "\nCeratinty: %s" % item)
    print("\n~~~~~~~~\n")
    return str(ret)


if __name__ == "__main__":
    ans = input("Display by click?[Y/n] ") != "n"
    bot_gen = data.get_next()
    hum_gen = data.show_next()
    print("~~~~~~~~\n")
    img, label = next(hum_gen)
    inpt = next(bot_gen)[0]
    DISP = plt.imshow(
        img,
        cmap="gist_gray",
    )
    evaluate(inpt, label)
    data.FIG.canvas.manager.set_window_title(label)
    if ans:
        cid = data.FIG.canvas.mpl_connect("button_press_event", on_click)
    else:
        threading.Thread(target=run).start()
    plt.show()
