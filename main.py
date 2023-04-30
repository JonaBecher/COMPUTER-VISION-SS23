import cv2
import numpy as np
import largestinteriorrectangle as lir
import math


def compute_contours(img, log=False):
    cv2.imshow('contours123', img)

    # detect edges and contours
    edges = cv2.Canny(img, 50, 200)
    cnts, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if log:
        cv2.drawContours(orig_img, cnts, -1, (0, 255, 0), 3)
        cv2.imshow('contours', orig_img)

    return cnts, img


def mask_img(img):
    # detect very bright parts of the image (reflections)
    mask = cv2.threshold(img, 215, 255, cv2.THRESH_BINARY_INV)[1]
    img = cv2.bitwise_and(img, img, mask=mask)

    ret, img = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY_INV)

    # fill holes
    kernel = np.ones((50,50),np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img


def get_contours_dimension(cnt):
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.0000000000000001 * peri, True)
    x, y, w, h = cv2.boundingRect(approx)

    return x, y, w, h


def get_contours_dimension_ellipse(cnt):
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.000000000000000000001 * peri, True)
    rect = cv2.fitEllipse(approx)
    (x, y), (MA, ma), angle = rect

    return x, y, MA, ma, angle, rect


EURO_COIN_DIAMETER_MM = 22.25  # diameter of reference object
orig_img = cv2.imread("img/alles6.jpeg")
img_gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)

img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
img_masked = mask_img(img_blur)

# coin dimensions are more precise with a more blured img
cnts, img = compute_contours(img_masked, log=False)

referenceObjIdx = -1
# map ref object to contour
for idx, cnt in enumerate(cnts):
    x, y, w, h = get_contours_dimension(cnt)
    if 0.9 <= w/h <= 1.1 and w > 20 and h > 20: # rund
        referenceObjIdx = idx
        ppmm = EURO_COIN_DIAMETER_MM / w  # pixel per millimeter
        print(w)
        print(h)
        break

if referenceObjIdx == -1:
    raise ValueError('No reference Point found.')

# details of the image (jags of the fork) require less blur
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
img_masked = mask_img(img_blur)
cnts, img = compute_contours(img_masked, log=False)
cv2.imshow("img_masked", img)

for idx, cnt in enumerate(cnts):
    if idx != -1:
        x, y, w, h = get_contours_dimension(cnt)
        if h > 25 and w > 25:
            mask = np.zeros(img.shape, np.uint8)
            mask[y:y+h, x:x+w] = img[y:y+h, x:x+w]

            #x, y, MA, ma, angle, rect = get_contours_dimension_ellipse(cnt)

            max_distance = 0
            angle = 0
            for i in range (int(180)):
                (img_h, img_w) = orig_img.shape[:2]
                (cX, cY) = (img_w // 2, img_h // 2)

                M = cv2.getRotationMatrix2D((cX, cY), i, 1.0)
                img_rotated = cv2.warpAffine(mask, M, (img_w, img_h))

                cnts, _ = compute_contours(img_rotated)
                cv2.drawContours(img_rotated, cnts, -1, (0, 255, 0), 2)

                if len(cnts) > 0:
                    largest_cnt = None
                    area = 0
                    for cnt in cnts:
                        x, y, w, h = get_contours_dimension(cnt)
                        if w * h > area:
                            largest_cnt = cnt

                    # get the maximally inscribed rectangle
                    polygon = np.array(list(largest_cnt), np.int32)
                    rectangle = lir.lir(largest_cnt.reshape(1, len(cnt), 2))
                    distance = math.sqrt((lir.pt1(rectangle)[0]-lir.pt2(rectangle)[0])**2+
                          (lir.pt1(rectangle)[1]-lir.pt2(rectangle)[1])**2)
                    if distance > max_distance:
                        max_distance = distance
                        angle = i

            (img_h, img_w) = orig_img.shape[:2]
            (cX, cY) = (img_w // 2, img_h // 2)

            M = cv2.getRotationMatrix2D((cX, cY), angle-360, 1.0)
            img_rotated = cv2.warpAffine(mask, M, (img_w, img_h))

            cnts, _ = compute_contours(img_rotated)
            cv2.drawContours(img_rotated, cnts, -1, (0, 255, 0), 2)
            cv2.imshow("Ergebnis"+str(idx), img_rotated)

            if len(cnts)>0:
                largest_cnt = None
                area = 0
                for cnt in cnts:
                    x, y, w, h = get_contours_dimension(cnt)
                    if w*h > area:
                       largest_cnt = cnt

                x, y, w, h = get_contours_dimension(largest_cnt)

                (mask_h, mask_w) = img_rotated.shape[:2]
                (maskX, maskY) = (img_w // 2, img_h // 2)

                M = cv2.getRotationMatrix2D((cX, cY), 360-angle, 1.0)
                mask = cv2.warpAffine(img_rotated, M, (img_w, img_h))


                coordinates = np.zeros(img_rotated.shape, np.uint8)
                coordinates[y-1, x-1] = 255
                coordinates[y-1, x - 1+ w] = 255
                coordinates[y-1 + h, x-1 + w] = 255
                coordinates[y-1+ h, x-1] = 255

                coordinates_rotated = cv2.warpAffine(coordinates, M, (img_w, img_h))
                itemindex = np.where(coordinates_rotated > 0)

                pts = []
                for i in range(len(itemindex[0])):
                    pts.append([itemindex[1][i], itemindex[0][i]])

                coordinates = []
                for i in range(len(pts)):
                    if len(coordinates) == 0:
                        coordinates.append(pts[i])
                    add = True
                    for y in range(len(coordinates)):
                        if abs(coordinates[y][1] - pts[i][1]) < 20 and abs(coordinates[y][0] - pts[i][0]) < 20:
                            add = False
                    if add:
                        coordinates.append(pts[i])

                text_coordinate = coordinates[np.argsort(np.array(coordinates)[:, 1])[0]]
                height = round(h * ppmm, 2)
                width = round(w * ppmm, 2)
                text_size_height, _ = cv2.getTextSize(f'''height: {height}mm''', cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                text_size_width, _ = cv2.getTextSize(f'''width: {width}mm''', cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)

                # Calculate the x and y coordinates for the top-left corner of the text
                text_height_x = int(text_coordinate[0] - text_size_height[0] / 2)
                text_width_y = int(text_coordinate[1] - text_size_height[1] / 2)

                text_height_y = int(text_coordinate[0] - text_size_width[0] / 2)
                text_width_y = int(text_coordinate[1] - text_size_width[1] / 2)

                # Draw the text on the image
                orig_img = cv2.putText(orig_img, f'''height: {height}mm''', (text_height_x, text_width_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                orig_img = cv2.putText(orig_img, f'''width: {width}mm''', (text_height_y, text_width_y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                cv2.polylines(orig_img, [np.array([coordinates[0], coordinates[1], coordinates[3], coordinates[2]])], True, (0, 0, 255))
                cv2.imshow("rectangle", orig_img)

cv2.imshow("Ergebnis", orig_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
