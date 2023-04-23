import cv2
import numpy as np


def compute_contours(img, blur, log=False):
    orig_img = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, blur, 0)

    if log:
        cv2.imshow("blur", img)

    # detect very bright parts of the image (reflections)
    mask = cv2.threshold(img, 210, 255, cv2.THRESH_BINARY_INV)[1]
    img = cv2.bitwise_and(img, img, mask=mask)

    ret, img = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY_INV)

    # fill holes
    kernel = np.ones((50,50),np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    if log:
        cv2.imshow("masked", img)

    # detect edges and contours
    edges = cv2.Canny(img, 50, 200)
    cnts, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(orig_img, cnts, -1, (0, 255, 0), 3)

    if log:
        cv2.imshow('Contours', orig_img)

    return cnts


def get_contours_dimension(cnt):
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.000001 * peri, True)
    x, y, w, h = cv2.boundingRect(approx)
    return x, y, w, h


EURO_COIN_DIAMETER_MM = 22.25  # diameter of reference object
orig_img = cv2.imread("img/loeffel_groß.jpeg")

referenceObjIdx = -1

# coin dimensions are more precise with a more blured img
cnts = compute_contours(orig_img.copy(), (7,7), log=True)

# map ref object to contour
for idx, cnt in enumerate(cnts):
    x, y, w, h = get_contours_dimension(cnt)
    if 0.9 <= w/h <= 1.1: # rund
        referenceObjIdx = idx
        ppmm = EURO_COIN_DIAMETER_MM / w  # pixel per millimeter
        break

if referenceObjIdx == -1:
    raise ValueError('No reference Point found.')

# details of the image (jags of the fork) require less blur
cnts = compute_contours(orig_img, (3,3), log=True)

for cnt in cnts:
    x, y, w, h = get_contours_dimension(cnt)
    width = round(w * ppmm, 2)
    height = round(h * ppmm, 2)
    # filter out noise
    if height > 12.5 and width > 12.5:
        bob = orig_img[y:y + h, x:x + w]
        cv2.rectangle(orig_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(orig_img, f'''width: {width}mm''',
                    (x, y-50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(orig_img, f'''height: {height}mm''',
                    (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.imshow("results", orig_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
