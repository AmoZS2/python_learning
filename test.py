import cv2 as cv

img = cv.imread('cat.jpg')
img = cv.resize(img, (400, 300))
cv.putText(img, 'Stay Home', (50, 50), cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (0, 255, 255), 4)
cv.rectangle(img, (60, 120), (140, 140), (20, 20, 20), -1)
cv.circle(img, (90, 240), 30, (0, 255, 255), -1)
# imshowは最後に記述
cv.imshow('image', img)

cv.waitKey(0)
cv.destroyAllWindows()