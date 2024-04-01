import numpy as np
import cv2

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15

# Encontramos la matriz fundamental
def stereoRectification(im1Gray, im2Gray):

    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    matches.sort(key=lambda x: x.distance, reverse=False)

    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC) 

    pts1 = points1[mask.ravel()==1]
    pts2 = points2[mask.ravel()==1]

    h1, w1 = im1Gray.shape
    h2, w2 = im2Gray.shape

    _, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(pts1), np.float32(pts2), F, imgSize=(w1,h1))

    recImg1 = cv2.warpPerspective(im1Gray, H1, (w1,h1))
    recImg2 = cv2.warpPerspective(im2Gray, H2, (w2,h2))

    return recImg1, recImg2