import threading

import matplotlib.pyplot as plt
import numpy as np
import torch

import data
from classes import PATH, r

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
        print("[%s / %s]" % (i, NUM_OF_IMAGES))
        if ans == target:
            SCORE += 1
    print("Success Rate:", str(round((SCORE / NUM_OF_IMAGES) * 100, 3)))


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
    ans1 = input("Train?[y/N] ") == "y"
    NUM_OF_IMAGES = data.load(ans1)
    if ans1:
        for epoch in range(3):
            running_loss = 0.0
            for i, tup in enumerate(data.get_next()):
                inpt, target = tup
                # zero the parameter gradients

                r.optimizer.zero_grad()

                # forward + backward + optimize
                output = r.forward(inpt)
                # print("Out:\n", output, "\nTarget:\n", target, "\n~~~\n")
                # if i == 3:
                #     break
                # print("Target:\n", target, "\nOutput:\n", output)
                loss: torch.Tensor = r.criterion(output, target)
                loss.backward()
                r.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                    running_loss = 0.0

        torch.save(r.state_dict(), PATH)
        print("Saved data at: %s" % PATH)
    else:
        r.load_state_dict(torch.load(PATH))
        r.eval()
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
