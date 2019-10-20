#! /usr/bin/env python

import sys

import numpy as np
import pytesseract
from pytesseract import Output
import cv2


def detect_dots(img, min_dist=5, param1=50, param2=30, min_rad=50, max_rad=100):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img, 5)
    circles = cv2.HoughCircles(
        img, cv2.HOUGH_GRADIENT,
        1, min_dist, param1=param1, param2=param2,
        minRadius=min_rad, maxRadius=max_rad,
    )
    return [
        (c[0], c[1], c[2])
        for c in np.uint16(np.around(circles[0]))
    ]


def detect_nums(img, min_conf=None):
    r = []

    data = pytesseract.image_to_data(
        img,
        config=
            '--psm 11'
            '--oem 1'
            '--user-words cfg/words'
            '-c tessedit_char_whitelist=0123456789'
        ,
        output_type=Output.DICT,
    )

    for i, x in enumerate(data['conf']):
        if int(x) < min_conf: continue

        if not data['text'][i].isnumeric(): continue

        r.append(( 
            int(data['text'][i]),
            (
                data['left'][i],
                data['top'][i],
                data['width'][i],
                data['height'][i],
            ),
        ))

    return r


def dist_sq(circ, box):
    x, y, _ = circ
    bx, by, bw, bh = box

    return (x - (bx + bw/2))**2 + (y - (by + bh/2))**2


def sort_dots(nums, dots):
    r = []
    nums.sort(key=(lambda x: x[0]))
    dots = dots[:]

    for num, box in nums:
        dists = list(map(lambda c: dist_sq(c, box), dots))
        ind_min = np.argmin(dists)
        d = dots.pop(ind_min)
        r.append(d) 

    return r


def fill_dots(img, dots, clr=(255, 255, 255), width=-1):
    for x, y, r in dots:
        cv2.circle(img,
            (x, y), r + 2,
            clr, width,
        )


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: %s SOURCE DEST' % sys.argv[0])
        exit(0)

    src = sys.argv[1]
    dest = sys.argv[2]

    orig_img = cv2.imread(src)
    img = orig_img.copy()

    dots = detect_dots(img, min_dist=20, param1=50, param2=25,
                       min_rad=30, max_rad=100)
    fill_dots(img, dots)
    nums = detect_nums(img, min_conf=70)

    dots_sorted = sort_dots(nums, dots)
    for i in range(-1, len(dots_sorted)-1):
        cx, cy, _ = dots_sorted[i]
        nx, ny, _ = dots_sorted[i+1]
        cv2.line(orig_img, (cx, cy), (nx, ny), (0, 0, 0), 2)

    for _, box in nums:
        cv2.rectangle(
            orig_img, box,
            (0, 255, 0), 2,
        )

    for x, y, r in dots:
        cv2.circle(orig_img, (x, y), r+1, (0, 255, 0), 2)

    cv2.imwrite(dest, orig_img)
