import cv2
import numpy as np
import largestinteriorrectangle as lir
import math


def compute_contours(img, log=False):
    # Kanten und Konturen erkennen
    edges = cv2.Canny(img, 50, 200)
    cnts, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if log:
        cv2.drawContours(orig_img, cnts, -1, (0, 255, 0), 3)
        cv2.imshow('contours', orig_img)

    return cnts, img


def mask_img(img):
    # Sehr helle Bereiche des Bildes erkennen (Reflexionen)
    mask = cv2.threshold(img, 215, 255, cv2.THRESH_BINARY_INV)[1]
    img = cv2.bitwise_and(img, img, mask=mask)

    # Bildschwellenwert anwenden
    ret, img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("Treshhold", img)

    # Kernel für die Morphologieoperation erstellen -> Löcher
    kernel = np.ones((25,25),np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img


def get_contours_dimension(cnt):
    return cv2.boundingRect(cnt)


EURO_COIN_DIAMETER_MM = 23.25  # diameter of reference object
orig_img = cv2.imread("img/gegenstaende.jpeg")
img_gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", img_gray)
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
cv2.imshow("Blur", img_blur)
img_masked = mask_img(img_blur)
cv2.imshow("Masked", img_masked)

cnts, img = compute_contours(img_masked, log=True)

cv2.imshow("img_masked", img)

referenceObjIdx = -1
# map ref object to contour
for idx, cnt in enumerate(cnts):
    x, y, w, h = get_contours_dimension(cnt)
    if 0.9 <= w/h <= 1.1 and w > 20 and h > 20: # rund
        referenceObjIdx = idx
        ppmm = EURO_COIN_DIAMETER_MM / h  # pixel per millimeter
        break

if referenceObjIdx == -1:
    raise ValueError('No reference Point found.')

for idx, cnt in enumerate(cnts):
    if idx != referenceObjIdx:
        # Abfrage von Koordinaten und Dimensionen der Kontur
        x, y, w, h = get_contours_dimension(cnt)
        if h > 25 and w > 25:
            # Erzeugung einer Bildmaske mit Nullen
            mask = np.zeros(img.shape, np.uint8)
            # Kopieren des Konturbereichs ins Maskenbild
            mask[y:y+h, x:x+w] = img[y:y+h, x:x+w]

            max_distance = 0
            angle = 0
            x1 = None
            x2 = None

            # Schleife über 180 Grad
            for i in range (int(180)):
                # Abrufen der Höhe und Breite des Originalbildes
                (img_h, img_w) = orig_img.shape[:2]
                # Berechnung des Schwerpunkts
                (cX, cY) = (img_w // 2, img_h // 2)
                # Erzeugung einer Rotationsmatrix
                M = cv2.getRotationMatrix2D((cX, cY), i, 1.0)
                # Anwendung der Rotation
                img_rotated = cv2.warpAffine(mask, M, (img_w, img_h))
                # Berechnung der Konturen im rotierten Bild
                cnts, _ = compute_contours(img_rotated)
                # Zeichnen der Konturen im rotierten Bild
                cv2.drawContours(img_rotated, cnts, -1, (0, 255, 0), 2)

                if len(cnts) > 0:
                    largest_cnt = None
                    area = 0
                    for cnt in cnts:
                        x, y, w, h = get_contours_dimension(cnt)
                        if w * h > area:
                            largest_cnt = cnt  # Auswahl der Kontur mit der größten Fläche

                    # Maximales Rechteck ermitteln
                    polygon = np.array(list(largest_cnt), np.int32)
                    rectangle = lir.lir(largest_cnt.reshape(1, len(cnt), 2))
                    distance = math.sqrt((lir.pt1(rectangle)[0]-lir.pt2(rectangle)[0])**2+
                          (lir.pt1(rectangle)[1]-lir.pt2(rectangle)[1])**2)
                    # Wenn bisher größtes Rechteck
                    if distance > max_distance:
                        max_distance = distance
                        angle = i

            (img_h, img_w) = orig_img.shape[:2]
            (cX, cY) = (img_w // 2, img_h // 2)

            # Erzeugung einer Rotationsmatrix mit dem gefundenen, maximalen Winkel
            M = cv2.getRotationMatrix2D((cX, cY), angle-360, 1.0)
            img_rotated = cv2.warpAffine(mask, M, (img_w, img_h))

            cv2.rectangle(img, x1, x2, (0, 255, 0), 2)

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

                # Erzeugung eines Koordinatenbildes mit Nullen
                coordinates = np.zeros(img_rotated.shape, np.uint8)
                # Setzen der Eckpunkte des Rechtecks im Koordinatenbild
                coordinates[y-1, x-1] = 255
                coordinates[y-1, x - 1+ w] = 255
                coordinates[y-1 + h, x-1 + w] = 255
                coordinates[y-1+ h, x-1] = 255

                coordinates_rotated = cv2.warpAffine(coordinates, M, (img_w, img_h))
                itemindex = np.where(coordinates_rotated > 0)

                pts = []
                for i in range(len(itemindex[0])):
                    # Erzeugung einer Liste von Koordinatenpunkten
                    pts.append([itemindex[1][i], itemindex[0][i]])

                coordinates = []
                for i in range(len(pts)):
                    if len(coordinates) == 0:
                        # Füge den ersten Koordinatenpunkt hinzu
                        coordinates.append(pts[i])
                    add = True
                    for y in range(len(coordinates)):
                        # Überprüfe, ob der Koordinatenpunkt nahe genug an einem vorhandenen Punkt liegt
                        if abs(coordinates[y][1] - pts[i][1]) < 20 and abs(coordinates[y][0] - pts[i][0]) < 20:
                            add = False
                    if add:
                        # Füge den Koordinatenpunkt hinzu, wenn er nicht nahe genug an einem vorhandenen Punkt liegt
                        coordinates.append(pts[i])

                text_coordinate = coordinates[np.argsort(np.array(coordinates)[:, 1])[0]]
                # Berechne die Höhe in Millimetern
                height = round(h * ppmm, 2)
                # Berechne die Breite in Millimetern
                width = round(w * ppmm, 2)
                text_size_height, _ = cv2.getTextSize(f'''height: {height}mm''', cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                text_size_width, _ = cv2.getTextSize(f'''width: {width}mm''', cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)

                # Berechne die x- und y-Koordinaten für die obere linke Ecke des Textes zur Höhe
                text_height_x = int(text_coordinate[0] - text_size_height[0] / 2)
                text_width_y = int(text_coordinate[1] - text_size_height[1] / 2)

                text_height_y = int(text_coordinate[0] - text_size_width[0] / 2)
                text_width_y = int(text_coordinate[1] - text_size_width[1] / 2)

                # Zeichne den Text auf das Bild
                orig_img = cv2.putText(orig_img, f'''height: {height}mm''', (text_height_x, text_width_y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)
                orig_img = cv2.putText(orig_img, f'''width: {width}mm''', (text_height_y, text_width_y-40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)

                cv2.polylines(orig_img, [np.array([coordinates[0], coordinates[1], coordinates[3], coordinates[2]])], True, (0, 0, 255))
                cv2.imshow("rectangle", orig_img)

print("finished")

cv2.imshow("Ergebnis", orig_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
