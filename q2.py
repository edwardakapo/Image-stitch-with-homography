#Author Oluwademilade Edward Akapo
#student No 101095403

import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('Image stitch\\uttower_right.jpg',0)          # queryImage
img2 = cv2.imread('Image stitch\\large2_uttower_left.jpg',0) # trainImage

detector = cv2.AKAZE_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = detector.detectAndCompute(img1, None)
kp2, des2 = detector.detectAndCompute(img2, None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
print(matches)

src_pts = np.float32([ kp1[matches[m].queryIdx].pt for m in range(0, 20) ]).reshape(-1,1,2)
dst_pts = np.float32([ kp2[matches[m].trainIdx].pt for m in range(0, 20) ]).reshape(-1,1,2)
print(src_pts)
print(dst_pts)

# routine findhomography()
h , status = cv2.findHomography(src_pts,dst_pts,cv2.RANSAC)
# warp prerspective routine witrh computed homograpy to warp right image into image of the same size as left
warped_img = cv2.warpPerspective(img1 ,h , (img2.shape[1], img2.shape[0]))
# do paste the warped right image inot the large left image
#loop over img2 then add the bigger pixel value
merged_img = np.copy(img2)
for x in range(img2.shape[0]):
    for y in range(img2.shape[1]):
        if img2[x][y] >= warped_img[x][y]:
            merged_img[x][y] = img2[x][y]
        else:
            merged_img[x][y] = warped_img[x][y]

# write paragraph on why merged has anomalies 

#display matching point in both input images
img3 = cv2.drawMatches(img1 ,kp1 , img2,kp2 , matches[:20], None , flags = 2)

cv2.imshow('Key points', img3)
cv2.imshow('Warped', warped_img)
cv2.imshow('Merged', merged_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

#save images
cv2.imwrite('Image stitch\\mywarped.jpg', warped_img)
cv2.imwrite('Image stitch\\mymerged.jpg', merged_img)
cv2.imwrite('Image stitch\\matchingpoints.jpg', img3)