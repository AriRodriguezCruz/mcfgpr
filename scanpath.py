import cv2

coords = [{"fixation_number": "1", "x": 59.0, "y": 129.0}, {"fixation_number": "2", "x": 97.0, "y": 193.0}, {"fixation_number": "3", "x": 182.0, "y": 159.0}, {"fixation_number": "4", "x": 296.0, "y": 192.0}, {"fixation_number": "5", "x": 359.0, "y": 164.0}, {"fixation_number": "6", "x": 448.0, "y": 176.0}, {"fixation_number": "7", "x": 538.0, "y": 175.0}, {"fixation_number": "8", "x": 624.0, "y": 174.0}, {"fixation_number": "9", "x": 704.0, "y": 180.0}, {"fixation_number": "10", "x": 542.0, "y": 224.0}, {"fixation_number": "11", "x": 360.0, "y": 252.0}, {"fixation_number": "12", "x": 175.0, "y": 281.0}, {"fixation_number": "13", "x": 272.0, "y": 287.0}, {"fixation_number": "14", "x": 463.0, "y": 292.0}, {"fixation_number": "15", "x": 559.0, "y": 303.0}, {"fixation_number": "16", "x": 193.0, "y": 383.0}, {"fixation_number": "17", "x": 284.0, "y": 401.0}, {"fixation_number": "18", "x": 385.0, "y": 399.0}, {"fixation_number": "19", "x": 296.0, "y": 468.0}, {"fixation_number": "20", "x": 177.0, "y": 540.0}, {"fixation_number": "21", "x": 319.0, "y": 546.0}, {"fixation_number": "22", "x": 186.0, "y": 586.0}, {"fixation_number": "23", "x": 339.0, "y": 584.0}, {"fixation_number": "24", "x": 192.0, "y": 630.0}, {"fixation_number": "25", "x": 282.0, "y": 627.0}, {"fixation_number": "26", "x": 199.0, "y": 669.0}]

img = cv2.imread('lenguajes.jpg')
for index_fixation, row in enumerate(coords):
    print(int(row['x']), int(row['y']))
    # cv2.circle(img, (int(row['x']), int(row['y'])), 15, (0, 0, 255), -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (int(row['x']), int(row['y']))
    fontScale = 1
    fontColor = (0, 0, 255)
    lineType = 2

    cv2.putText(img, str(index_fixation),
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)
    # img[int(row['y']), int(row['x'])] = [0, 0, 255]


cv2.imwrite("scanpath.png", img)  # save the last-displayed image to file, for our report
cv2.imshow('scanpath.png', img)
cv2.waitKey(0)
cv2.destroyAllWindows()