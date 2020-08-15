import cv2
import numpy as np

def stitch(images):
		
    # Detect keypoints and features for the images
    (image2, image1) = images
    (keypoints_image1, features_image1) = detect_features(image1)
    (keypoints_image2, features_image2) = detect_features(image2)
    
    # Matching the features between the two images
    matched_keypoints, homography_matrix, status = match_keypoints(keypoints_image1, keypoints_image2, features_image1, features_image2)
    
    result_image = cv2.warpPerspective(image1, homography_matrix, (image1.shape[1] + image2.shape[1], image1.shape[0]))
    result_image[0:image2.shape[0], 0:image2.shape[1]] = image2

    # Return the stitched image
    return result_image


def detect_features(image):
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    orb = cv2.ORB_create(nfeatures=2500)
    keypoints, features = orb.detectAndCompute(gray_image, None)

    # Converting the Keypoints to numpy array
    keypoints = np.float32([keypoint.pt for keypoint in keypoints])

    return (keypoints, features)


def match_keypoints(keypoints_image1, keypoints_image2, features_image1, features_image2):
    
    number_of_closest_key_points = 2

    # Find the matches using brute force
    descriptor_matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE)
    raw_matches = descriptor_matcher.knnMatch(features_image1, features_image2, number_of_closest_key_points)
    matches = best_keypoints_matches(raw_matches)
        
    # Location of good matches
    points_image1 = np.float32([keypoints_image1[i] for (i, _) in matches])
    points_image2 = np.float32([keypoints_image2[i] for (_, i) in matches])
    
    # Compute the homography between the two sets of points using RANSAC
    homography_matrix, status = cv2.findHomography(points_image1, points_image2, cv2.RANSAC)
    
    # Return the matches with the homograpy matrix and status of each matched point (if it's correct or not)
    return (matches, homography_matrix, status)


def best_keypoints_matches(raw_matches):
    
    lowes_ratio = 0.75
    matches = []

    # loop over the raw matches
    for match in raw_matches:
        
        point1_distance = match[0].distance
        point2_distance = match[1].distance
        
        # Using Lowe's ratio test to minimize false positives and creating the list of best keypoints
        if point1_distance < point2_distance * lowes_ratio:
            keypoints1 = match[0].queryIdx
            keypoints2 = match[0].trainIdx
            matches.append((keypoints1, keypoints2))
    
    return matches


def remove_black_borders(image):
    height, width = image.shape[:2]
    cropI=0
    val=image[height-1][width-1]

    for i in range(width):
        if(not(image[height-1][width-i-1][0] == val[0])):
            return image[:,0:width-i-1]



def resize_image(image, scale_percent=50):

    # Calculate the 50 percent of original dimensions
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)

    size = (width, height)

    result_image = cv2.resize(image, size)

    return result_image


imageA = resize_image(cv2.imread('images/image1.jpg'))
imageB = resize_image(cv2.imread('images/image2.jpg'))
imageC = resize_image(cv2.imread('images/image3.jpg'))
imageD = resize_image(cv2.imread('images/image4.jpg'))
imageE = resize_image(cv2.imread('images/image5.jpg'))

result1 = stitch([imageA, imageB])
result1 = remove_black_borders(result1)

result2 = stitch([result1, imageC])
result2 = remove_black_borders(result2)

result3 = stitch([result2, imageD])
result3 = remove_black_borders(result3)

result4 = stitch([result3, imageE])
result4 = remove_black_borders(result4)

cv2.imwrite('stitchedImage.png', result4)

cv2.imshow("Result", result4)
cv2.waitKey(0)