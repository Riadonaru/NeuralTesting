import threading

import matplotlib.pyplot as plt
import torch

import data
from classes import PATH, r

LOAD_PATH = "./networks/96.7.pth"
SCORE = 0
DISPLAY_GUI = True  # Considerabley slower when set to True!
TEST_WITH_TRAINING_IMAGES = False
# match str(type(r)):
#     case "<class 'classes.Recognizer'>":
#         print("R")
#     case "<class 'classes.OneOutRecognizer'>":
#         print("OOR")


def run():

    global SCORE

    for i, tup in enumerate(hum_gen):
        img, target = tup
        inpt = next(bot_gen)[0]
        if DISPLAY_GUI:
            DISP.set_data(img)
            FIG.canvas.manager.set_window_title(target)
            FIG.canvas.draw()
        print("[%s / %s]" % (i + 2, NUM_OF_IMAGES))
        ans = evaluate(inpt, target)
        if ans == target:
            SCORE += 1
        elif DISPLAY_GUI:
            e.wait()
            e.clear()
    print("Success Rate:", str(round((SCORE / NUM_OF_IMAGES) * 100, 3)))


def on_click(event):
    input = next(bot_gen)[0]
    img, target = next(hum_gen)
    DISP.set_data(img)
    FIG.canvas.manager.set_window_title(target)
    FIG.canvas.draw()
    evaluate(input, target)


def resume(event):
    e.set()


def evaluate(ten: torch.Tensor, target: str):
    out = r.forward(ten).detach()
    print("Target:", target)
    ret = 0
    highest = 0
    for array in out:
        for i, item in enumerate(array):
            item = float(item)
            if item > highest:
                highest = item
                ret = i
    print("\nGuess: %s" % ret, "\nCeratinty: %s" % highest)
    print("\n~~~~~~~~\n")
    return str(ret)


if __name__ == "__main__":
    ans1 = input("Train?[y/N] ") == "y"
    FIG = plt.gcf()
    if TEST_WITH_TRAINING_IMAGES:
        NUM_OF_IMAGES = data.load(TEST_WITH_TRAINING_IMAGES)
    else:
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
        r.load_state_dict(torch.load(LOAD_PATH))
        r.eval()
        ans = input("Pause on Error?[Y/n] ") != "n"
        bot_gen = data.get_next()
        hum_gen = data.show_next()
        print("~~~~~~~~\n")
        img, target = next(hum_gen)
        inpt = next(bot_gen)[0]
        if DISPLAY_GUI:
            DISP = plt.imshow(
                img,
                cmap="gist_gray",
            )
            FIG.canvas.manager.set_window_title(target)
        print("[1 / %s]" % (NUM_OF_IMAGES))
        pred = evaluate(inpt, target)
        if pred == target:
            SCORE += 1
        if not ans:
            cid = FIG.canvas.mpl_connect("button_press_event", on_click)
        else:
            e = threading.Event()
            if DISPLAY_GUI:
                cid = FIG.canvas.mpl_connect("button_press_event", resume)
            threading.Thread(target=run).start()
        if DISPLAY_GUI:
            plt.show()
