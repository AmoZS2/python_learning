import cv2 as cv

cap = cv.VideoCapture(1)

while True:
    key = cv.waitKey(1)
    if key != -1:
        break

    ret, frame = cap.read()
    cv.imshow('image', frame)

cap.release()
cv.destroyAllWindows()