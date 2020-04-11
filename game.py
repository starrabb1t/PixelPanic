import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from scipy import signal
import vlc
import sys

block = np.array([[1, 1],
                  [1, 1]], dtype='bool')

glider = np.array([[1, 1, 0],
                   [1, 0, 1],
                   [1, 0, 0]], dtype='bool')

shotgun = np.array(
    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
     [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    dtype='bool')


class Game:
    def __init__(self):
        self.patterns = [block, glider, shotgun]
        self.pattern_names = ["block", "glider", "shotgun"]

        self.selected_pattern = 0
        self.selected_side = True
        self.pause = False

        self.size = (180, 320)
        self.battle_size = (173, 213)

        self.red_mask = np.zeros(self.battle_size, dtype='bool')
        self.blue_mask = np.zeros(self.battle_size, dtype='bool')


g = Game()
Instance = vlc.Instance('--input-repeat=999999')

edge = np.ones(g.battle_size, dtype='bool')
edge[0, :] = False
edge[g.battle_size[0] - 1, :] = False
edge[:, 0] = False
edge[:, g.battle_size[1] - 1] = False

sum_filter_3x3 = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]], dtype="uint8")


def menu():
    player = Instance.media_player_new()
    media = Instance.media_new("sound/menu.mp3")
    media.get_mrl()
    player.set_media(media)
    player.play()

    canvas = np.zeros((g.size[0], g.size[1], 3), dtype='uint8')
    cv2.imshow('canvas', canvas)

    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)

    font = ImageFont.truetype("font/prospect.ttf", 42)
    draw.text((50, 60), "Clapeyron", (255, 255, 255), font=font)

    font = ImageFont.truetype("font/arcade.ttf", 8)
    draw.text((195, 100), "g a m e s", (255, 255, 255), font=font)

    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    image = image.astype('bool').astype('uint8')

    cv2.waitKey(1500)

    for i in range(255):
        cv2.imshow('canvas', i * image)
        key_flag = cv2.waitKey(18) & 0xFF

        if key_flag == 32:
            break

    cv2.imshow('canvas', 255 * image)

    n = 0
    for i in range(200):
        n += 1

        sum_map = signal.convolve2d(image, sum_filter_3x3, boundary='symm', mode='same')
        a = ((sum_map >= 2)).astype('uint8')  # mask to delete cells with < 2 neighbours
        b = ((sum_map <= 3)).astype('uint8')  # mask to delete cells with > 3 neighbours
        c = ((sum_map == 3)).astype('uint8')  # mask to create cells where == 3 neighbours

        image = (image * a * b + c).astype('bool').astype('uint8')

        cv2.imshow('canvas', (255 - n) * image)
        key_flag = cv2.waitKey(10) & 0xFF

        if key_flag == 32:
            break

    n = 0

    while True:
        n += 0.1

        sum_map = signal.convolve2d(image, sum_filter_3x3, boundary='symm', mode='same')
        a = ((sum_map >= 2)).astype('uint8')  # mask to delete cells with < 2 neighbours
        b = ((sum_map <= 3)).astype('uint8')  # mask to delete cells with > 3 neighbours
        c = ((sum_map == 3)).astype('uint8')  # mask to create cells where == 3 neighbours

        image = (image * a * b + c).astype('bool').astype('uint8')

        canvas = np.zeros((g.size[0], g.size[1], 3), dtype='uint8')
        canvas[:, :, 0] = (255 - 199) * image
        canvas[:, :, 1] = (255 - 199) * image
        canvas[:, :, 2] = (255 - 199) * image
        canvas = Image.fromarray(canvas)
        draw = ImageDraw.Draw(canvas)

        font = ImageFont.truetype("font/arcade.ttf", 6)
        draw.text((95, 150), "press space to play..",
                  (int(120 * (np.cos(n) + 1)), int(120 * (np.cos(n) + 1)), int(120 * (np.cos(n) + 1))), font=font)

        font = ImageFont.truetype("font/karmatic.ttf", 32)
        draw.text((40, 40), "PIXEL", (212, 229, 22), font=font)
        draw.text((90, 80), "PANIC !!!", (208, 50, 255), font=font)

        cv2.imshow('canvas', np.array(canvas))
        key_flag = cv2.waitKey(5) & 0xFF

        if key_flag == 32:
            player.stop()
            break
        elif key_flag == 27:
            sys.exit(0)


def loop():
    g.selected_pattern = 0
    g.red_mask = np.zeros(g.battle_size, dtype='bool')
    g.blue_mask = np.zeros(g.battle_size, dtype='bool')
    g.score = 0
    g.cells = 50

    """
    for i in range(g.battle_size[1] // 4):
        for j in range(5):
            spawn(1, 1, (10 * j, 10 * i))
    """

    player = Instance.media_player_new()
    media = Instance.media_new("sound/arcade.mp3")
    media.get_mrl()
    player.set_media(media)
    player.play()

    cv2.setMouseCallback('canvas', click_handler)

    while (True):
        key_flag = cv2.waitKey(5) & 0xFF

        if key_flag == ord('q'):
            player.stop()
            break
        elif key_flag == 32:
            pause()
        elif key_flag == ord('s'):
            g.selected_side = not g.selected_side
        elif key_flag == ord('c'):
            g.red_mask = np.zeros(g.battle_size, dtype='bool')
            g.blue_mask = np.zeros(g.battle_size, dtype='bool')
        elif key_flag == ord('r'):
            g.patterns[g.selected_pattern] = np.rot90(g.patterns[g.selected_pattern])
        elif key_flag == ord('f'):
            g.patterns[g.selected_pattern] = np.flip(g.patterns[g.selected_pattern], axis=0)
        elif key_flag == 27:
            sys.exit(0)
        elif key_flag == ord('1'):
            g.selected_pattern = 0
        elif key_flag == ord('2'):
            g.selected_pattern = 1
        elif key_flag == ord('3'):
            g.selected_pattern = 2

        render()
        update()


def spawn(pattern, player, coords):
    pat = g.patterns[pattern]
    pat_size = pat.shape

    if g.battle_size[0] - coords[0] <= pat_size[0] // 2 or \
            g.battle_size[1] - coords[1] <= pat_size[1] // 2 or \
            coords[0] <= pat_size[0] // 2 or \
            coords[1] <= pat_size[1] // 2:

        pass

    elif player == 0:
        bias = (pat_size[0] % 2, pat_size[1] % 2)

        print(pat_size)
        print(g.blue_mask.shape)
        print(g.blue_mask[
              coords[0] - pat_size[0] // 2:coords[0] + pat_size[0] // 2 + bias[0],
              coords[1] - pat_size[1] // 2:coords[1] + pat_size[1] // 2 + bias[1]].shape)

        g.blue_mask[
        coords[0] - pat_size[0] // 2:coords[0] + pat_size[0] // 2 + bias[0],
        coords[1] - pat_size[1] // 2:coords[1] + pat_size[1] // 2 + bias[1]] = pat

    elif player == 1:
        bias = (pat_size[0] % 2, pat_size[1] % 2)

        print(coords)
        print(pat_size)
        print(g.red_mask.shape)
        print(g.red_mask[
              coords[0] - pat_size[0] // 2:coords[0] + pat_size[0] // 2 + bias[0],
              coords[1] - pat_size[1] // 2:coords[1] + pat_size[1] // 2 + bias[1]].shape)

        g.red_mask[
        coords[0] - pat_size[0] // 2:coords[0] + pat_size[0] // 2 + bias[0],
        coords[1] - pat_size[1] // 2:coords[1] + pat_size[1] // 2 + bias[1]] = pat


def update():
    sum_map = signal.convolve2d(g.red_mask.astype('uint8'), sum_filter_3x3, boundary='symm', mode='same')
    a = ((sum_map >= 2)).astype('uint8')  # mask to delete cells with < 2 neighbours
    b = ((sum_map <= 3)).astype('uint8')  # mask to delete cells with > 3 neighbours
    c = ((sum_map == 3)).astype('uint8')  # mask to create cells where == 3 neighbours

    g.red_mask = (edge * (g.red_mask.astype('uint8') * a * b + c).astype('bool'))

    sum_map = signal.convolve2d(g.blue_mask.astype('uint8'), sum_filter_3x3, boundary='symm', mode='same')
    a = ((sum_map >= 2)).astype('uint8')  # mask to delete cells with < 2 neighbours
    b = ((sum_map <= 3)).astype('uint8')  # mask to delete cells with > 3 neighbours
    c = ((sum_map == 3)).astype('uint8')  # mask to create cells where == 3 neighbours

    g.blue_mask = (edge * (g.blue_mask.astype('uint8') * a * b + c).astype('bool'))

    inv_intersection = np.invert(g.red_mask * g.blue_mask)
    g.blue_mask = inv_intersection * g.blue_mask
    g.red_mask = inv_intersection * g.red_mask


def render():
    canvas = 255 * np.zeros((g.size[0], g.size[1], 3), dtype='uint8')

    canvas = cv2.rectangle(canvas, (103, 3), (g.size[1] - 4, g.size[0] - 4), (200, 200, 200), 1)

    # print(g.size[1] - 4 - 103, g.size[0] - 4 - 3)

    canvas = cv2.rectangle(canvas, (3, 120), (99, g.size[0] - 4), (200, 200, 200), 1)

    canvas[3:g.size[0] - 4, 103:g.size[1] - 4, 0][g.blue_mask] = 212
    canvas[3:g.size[0] - 4, 103:g.size[1] - 4, 1][g.blue_mask] = 229
    canvas[3:g.size[0] - 4, 103:g.size[1] - 4, 2][g.blue_mask] = 22

    canvas[3:g.size[0] - 4, 103:g.size[1] - 4, 0][g.red_mask] = 208
    canvas[3:g.size[0] - 4, 103:g.size[1] - 4, 1][g.red_mask] = 50
    canvas[3:g.size[0] - 4, 103:g.size[1] - 4, 2][g.red_mask] = 255

    pat_center = (140, 50)

    if g.selected_side:
        pat = g.patterns[g.selected_pattern]
        pat_size = pat.shape

        bias = (pat_size[0] % 2, pat_size[1] % 2)

        canvas[
        pat_center[0] - pat_size[0] // 2:pat_center[0] + pat_size[0] // 2 + bias[0],
        pat_center[1] - pat_size[1] // 2:pat_center[1] + pat_size[1] // 2 + bias[1],
        0][pat] = 212

        canvas[
        pat_center[0] - pat_size[0] // 2:pat_center[0] + pat_size[0] // 2 + bias[0],
        pat_center[1] - pat_size[1] // 2:pat_center[1] + pat_size[1] // 2 + bias[1],
        1][pat] = 229

        canvas[
        pat_center[0] - pat_size[0] // 2:pat_center[0] + pat_size[0] // 2 + bias[0],
        pat_center[1] - pat_size[1] // 2:pat_center[1] + pat_size[1] // 2 + bias[1],
        2][pat] = 22

    else:
        pat = g.patterns[g.selected_pattern]
        pat_size = pat.shape

        bias = (pat_size[0] % 2, pat_size[1] % 2)

        canvas[
        pat_center[0] - pat_size[0] // 2:pat_center[0] + pat_size[0] // 2 + bias[0],
        pat_center[1] - pat_size[1] // 2:pat_center[1] + pat_size[1] // 2 + bias[1],
        0][pat] = 208

        canvas[
        pat_center[0] - pat_size[0] // 2:pat_center[0] + pat_size[0] // 2 + bias[0],
        pat_center[1] - pat_size[1] // 2:pat_center[1] + pat_size[1] // 2 + bias[1],
        1][pat] = 50

        canvas[
        pat_center[0] - pat_size[0] // 2:pat_center[0] + pat_size[0] // 2 + bias[0],
        pat_center[1] - pat_size[1] // 2:pat_center[1] + pat_size[1] // 2 + bias[1],
        2][pat] = 255

    canvas = Image.fromarray(canvas)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("font/arcade.ttf", 7)
    draw.text((10, 5), "control:", (255, 255, 255), font=font)
    draw.text((10, 105), "pattern: {}".format(g.selected_pattern + 1), (255, 255, 255), font=font)

    if g.pause:
        draw.text((195, 85), "pause", (200, 200, 200), font=font)

    font = ImageFont.truetype("font/arcade.ttf", 6)
    draw.text((10, 20), "q-menu", (200, 200, 200), font=font)
    draw.text((10, 30), "esc-exit", (200, 200, 200), font=font)
    draw.text((10, 40), "space-pause", (200, 200, 200), font=font)
    draw.text((10, 50), "s-switch side", (200, 200, 200), font=font)
    draw.text((10, 60), "f-flip", (200, 200, 200), font=font)
    draw.text((10, 70), "r-rotate", (200, 200, 200), font=font)
    draw.text((10, 80), "1..{}-patterns".format(len(g.patterns)), (200, 200, 200), font=font)
    draw.text((10, 90), "c-clean", (200, 200, 200), font=font)
    draw.text((10, 160), "{}".format(g.pattern_names[g.selected_pattern]), (255, 255, 255), font=font)

    canvas = np.array(canvas)

    cv2.imshow('canvas', canvas)


def click_handler(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if g.selected_side:
            spawn(g.selected_pattern, 0, (y - 3, x - 103))
        else:
            spawn(g.selected_pattern, 1, (y - 3, x - 103))


def pause():
    g.pause = True
    while True:
        key_flag = cv2.waitKey(1) & 0xFF

        if key_flag == 32:
            g.pause=False
            break
        elif key_flag == 27:
            sys.exit(0)
        elif key_flag == ord('s'):
            g.selected_side = not g.selected_side
        elif key_flag == ord('c'):
            g.red_mask = np.zeros(g.battle_size, dtype='bool')
            g.blue_mask = np.zeros(g.battle_size, dtype='bool')
        elif key_flag == ord('r'):
            g.patterns[g.selected_pattern] = np.rot90(g.patterns[g.selected_pattern])
        elif key_flag == ord('f'):
            g.patterns[g.selected_pattern] = np.flip(g.patterns[g.selected_pattern], axis=0)
        elif key_flag == 27:
            sys.exit(0)
        elif key_flag == ord('1'):
            g.selected_pattern = 0
        elif key_flag == ord('2'):
            g.selected_pattern = 1
        elif key_flag == ord('3'):
            g.selected_pattern = 2

        render()


if __name__ == '__main__':
    print("Hello, World!")

    cv2.namedWindow('canvas', cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty("canvas", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.resizeWindow('canvas', 1920, 1080)

    menu()

    while (True):
        loop()
        menu()
