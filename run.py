#! /usr/bin/env python

from PIL import Image
import pytesseract
from pytesseract import Output
import cv2

img = cv2.imread('data/2.png')
ih, iw, _ = img.shape

data = pytesseract.image_to_data(
    img,
    output_type=Output.DICT,
)

print(data)
n = len(data['level'])

for i in range(n):
    x, y, w, h = (
        data['left'][i],
        data['top'][i],
        data['width'][i],
        data['height'][i],
    )
    cv2.rectangle(
        img,
        (x, y, w, h),
        (0, 255, 0),
        2,
    )

cv2.imwrite('ass.png', img)
