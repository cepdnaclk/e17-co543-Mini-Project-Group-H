from re import template
import cv2
import numpy as np
from skimage.exposure import match_histograms
from skimage.metrics import structural_similarity as ssim

def check_aspect_ratio(dim=None, contour=None, tolerance=0.25, eps=1e-9):
    if contour is not None:
        _, _, w, h = cv2.boundingRect(contour)
    if dim is not None:
        w, h = dim
    aspect_ratio = w / (h + eps)
    # Allow 25% (is it too big?) tolerance.
    # 1.5 aspect ratio thiyena number plates input wenne na kiyala assume karamu neh naththan kela wenna puluwan🥲
    return abs(aspect_ratio - 5) / 5 < tolerance # or abs(aspect_ratio - 1.5) / 1.5 < tolerance

def power_transform(img, gamma=1.0):
    return np.array(255. * (img / 255.) ** gamma, dtype=np.uint8)

def contrast_stretch(img, r1, s1, r2, s2):
    def mapPixels(x):
        if x <= r1: return (s1 / r1) * x
        elif r1 < x <= r2: return (x - r1) * (s2 - s1) / (r2 - r1) + s1
        return (x - r2) * (255 - s2) / (255 - r2) + s2

    return np.array([[mapPixels(x) for x in row] for row in img], dtype=np.uint8).reshape(img.shape)

def fix_contour(contour):
    top_left, top_right, bottom_left, bottom_right = contour
    if top_left[1] > top_right[1]:
        top_left, top_right = top_right, top_left
    if bottom_left[1] > bottom_right[1]:
        bottom_left, bottom_right = bottom_right, bottom_left
    min_height = min(top_right[1] - top_left[1], bottom_right[1] - bottom_left[1])
    max_height = max(top_right[1] - top_left[1], bottom_right[1] - bottom_left[1])
    min_width = min(bottom_right[0] - top_right[0], bottom_left[0] - top_left[0])
    max_width = max(bottom_right[0] - top_right[0], bottom_left[0] - top_left[0])
    if check_aspect_ratio(dim=(min_width, min_height)):
        if bottom_right[0] - top_right[0] < bottom_left[0] - top_left[0]:
            top_left = (top_right[0], top_right[1] - min_height)
            bottom_left = (bottom_right[0], bottom_right[1] - min_height)
        else:
            top_right = (top_left[0], top_left[1] + min_height)
            bottom_right = (bottom_left[0], bottom_left[1] + min_height)
    elif check_aspect_ratio(dim=(max_width, max_height)):
        if bottom_right[0] - top_right[0] < bottom_left[0] - top_left[0]:
            top_right = (top_left[0], top_left[1] + max_height)
            bottom_right = (bottom_left[0], bottom_left[1] + max_height)
        else:
            top_left = (top_right[0], top_right[1] - max_height)
            bottom_left = (bottom_right[0], bottom_right[1] - max_height)
    elif check_aspect_ratio(dim=(max_width, min_height)):
        if bottom_right[0] - top_right[0] < bottom_left[0] - top_left[0]:
            top_right = (top_left[0], top_left[1] + min_height)
            bottom_right = (bottom_left[0], bottom_left[1] + min_height)
        else:
            top_left = (top_right[0], top_right[1] - min_height)
            bottom_left = (bottom_right[0], bottom_right[1] - min_height)
    elif check_aspect_ratio(dim=(min_width, max_height)):
        if bottom_right[0] - top_right[0] < bottom_left[0] - top_left[0]:
            top_left = (top_right[0], top_right[1] - max_height)
            bottom_left = (bottom_right[0], bottom_right[1] - max_height)
        else:
            top_right = (top_left[0], top_left[1] + max_height)
            bottom_right = (bottom_left[0], bottom_left[1] + max_height)
    return np.array([top_right, top_left, bottom_left, bottom_right])

def find_rect_contour(contour):
    if len(contour) == 0:
        return None
   # TODO
    return contour

def normalize(arr, eps=1e-9):
    range = arr.max() - arr.min()
    amin = arr.min()
    return (arr - amin) * 255 / (range + eps)

def compare_images(img1, img2):
    img1 = normalize(img1)
    img2 = normalize(img2)
    img1 = cv2.resize(img1, img2.shape[::-1])
    diff = img1 - img2 
    zero_norm = np.linalg.norm(diff.ravel(), 0)
    return zero_norm

def brightness(img):
    if len(img.shape) == 3:
        return np.average(np.linalg.norm(img, axis=2)) / np.sqrt(3)
    else:
        return np.average(img)

def perspective_transform(I, pts1, pts2, dsize):
    '''
    Perspective transform.

    pts1: List of points in the original image.
    pts2: List of points in the transformed image.

    Returns:
        Transformed image.
    '''
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(I, M, dsize)

def preprocess_licenseplate(image):
    ref_hist = cv2.imread('./metadata/hist_ref.png', 0)
    ref_hist_1 = cv2.imread('./metadata/hist_ref_1.png', 0)
    # Convert to grayscale if needed.
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Remove noise by matching histogram with reference.
    brightness_val = brightness(image)
    if brightness_val < 110:
        image = match_histograms(image, ref_hist)
    else:
        image = match_histograms(image, ref_hist_1)
    return image.astype(np.uint8)

def localize_licenseplate(image, k1, k2, k3, k4):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 15))
    image = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    # Thresholding the tophat result
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # Eroding the thresholded image to remove background noise
    image = cv2.erode(image, kernel, iterations=k1)
    # Closing the gaps (removing false negatives) in between white pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 3))
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=k2)
    # Remove noise (false positives) from foreground
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=k3)
    # Dilating to expand the foreground closer to the original (expected) size
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    image = cv2.dilate(image, kernel, iterations=k4)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    return image

def detect_licenseplate(image, visualize=True):
    # Find the contour that bounds the license plate.
    plate_contour = None
    best_score = float('inf')
    best_mask = None
    im_w, im_h = image.shape[0], image.shape[1]
    for k1 in range(1, 4):
        for k2 in range(2, 7):
            for k3 in range(3, 9):
                for k4 in range(2, 6):
                    temp_contour = None
                    mask = localize_licenseplate(image, k1, k2, k3, k4)
                    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
                    for contour in contours:
                        perimeter = cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, 0.06 * perimeter, True).reshape(-1, 2)
                        if len(approx) == 4:
                            if check_aspect_ratio(contour=approx):
                                temp_contour = fix_contour(approx)
                                break
                    if temp_contour is None:
                        edges = cv2.Canny(image, 100, 200)
                        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
                        for contour in contours:
                            perimeter = cv2.arcLength(contour, True)
                            approx = cv2.approxPolyDP(contour, 0.06 * perimeter, True).reshape(-1, 2)
                            if len(approx) == 4:
                                if check_aspect_ratio(contour=approx, tolerance=0.28):
                                    temp_contour = fix_contour(approx)
                                    break
                    if temp_contour is None: continue
                    x, y, w, h = cv2.boundingRect(temp_contour)
                    cropped_mask = mask[y:y + h, x:x + w]
                    if cropped_mask.shape[0] == 0: continue
                    ideal_mask = np.ones(cropped_mask.shape)
                    # Minimize the score returned by compare_images as well as the aspect ratio difference while maximizing the area of the license plate.
                    # Each three metrics are weighted by experimenting with different values.
                    white_pixels_ratio = np.sum(cropped_mask == 255) / (im_w * im_h)
                    structural_similarity = ssim(cropped_mask.astype(np.uint8), ideal_mask.astype(np.uint8)) if max(w, h) < 7 else 0
                    score = compare_images(cropped_mask, ideal_mask) / 3500 + 3 * abs(w / h - 5) - 150 * white_pixels_ratio - 25 * structural_similarity
                    if best_score > score and w * h / (im_w * im_h) > 0.02:
                        plate_contour = temp_contour
                        best_mask = mask
                        best_score = score
                    # if plate_contour is None: continue
                    # test = mask.copy()
                    # test = cv2.cvtColor(test, cv2.COLOR_GRAY2BGR)
                    # test = cv2.drawContours(test, [plate_contour], -1, (0, 255, 0), 2)
                    # cv2.imshow(f'k1={k1}, k2={k2}, k3={k3}, k4={k4}, score={score}', test)
                    # cv2.waitKey(0)
    if plate_contour is None:
        return None
    # test = best_mask.copy()
    # test = cv2.cvtColor(test, cv2.COLOR_GRAY2BGR)
    # test = cv2.drawContours(test, [plate_contour], -1, (0, 255, 0), 2)
    # cv2.imshow('test', test)
    # cv2.waitKey(0)
    # Crop the license plate.
    x, y, w, h = cv2.boundingRect(plate_contour)
    cropped = image[y:y + h, x:x + w]
    # cv2.imshow('cropped', cropped)
    # cv2.waitKey(0)
    # Blackhat morphological transformation to enhance the text in the white background.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    cropped = cv2.morphologyEx(cropped, cv2.MORPH_BLACKHAT, kernel)
    # Thresholding to extract the text.
    cropped = cv2.threshold(cropped, 50, 255, cv2.THRESH_BINARY)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cropped = cv2.morphologyEx(cropped, cv2.MORPH_OPEN, kernel)
    cropped_area = w * h
    # cv2.imshow('test', cropped)
    # cv2.waitKey(0)
    # Segment the characters in the license plate.
    (num_labels, _, stats, centroids) = cv2.connectedComponentsWithStats(cropped, 4, cv2.CV_32S)
    avg_area = (stats[1:, cv2.CC_STAT_WIDTH] * stats[1:, cv2.CC_STAT_HEIGHT]).mean()
    avg_height = stats[1:, cv2.CC_STAT_HEIGHT].mean()
    avg_height_num, avg_width_num = 0, 0
    num_candidates = []
    output = cropped.copy()
    output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    prev_y = 0
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if abs(y - prev_y) > cropped.shape[0] / 3.5 or 9 * cropped.shape[1] / 10 < x or x < cropped.shape[1] / 6: continue
        if w * h > cropped_area / avg_area * 5 and avg_height * 1.45 < h and 0.5 < w / h < 1:
            num_candidates.append((cropped[y:y + h, x:x + w], centroids[i][0]))
            avg_height_num += h
            avg_width_num += w
            output = cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
        prev_y = y
    if len(num_candidates) > 0:
        avg_height_num /= len(num_candidates)
        avg_width_num /= len(num_candidates)
    if visualize:
        cv2.imshow('Segmented characters', output)
        cv2.waitKey(0)
    num_candidates = sorted(num_candidates, key=lambda x: x[1])
    num_candidates = [x[0] for x in num_candidates]
    province_candidates = []
    output = cropped.copy()
    output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if x > cropped.shape[1] / 4 or y < cropped.shape[0] / 3: continue
        if cropped_area / avg_area * 1.5 < w * h and avg_height_num * 0.8 > h > avg_height_num * 0.3 and avg_width_num * 0.3 < w and 0.4 < w / h < 1:
            province_candidates.append((cropped[y:y + h, x:x + w], centroids[i][0]))
            output = cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
    if visualize:
        cv2.imshow('Segmented characters', output)
        cv2.waitKey(0)
    province_candidates = sorted(province_candidates, key=lambda x: x[1])
    province_candidates = [x[0] for x in province_candidates]
    return plate_contour, num_candidates, province_candidates

def recognize_characters(candidates, ground_truth, idx2label, province=False):
    res = []
    def eval_func(candidate, img):
        if province:
            score = cv2.minMaxLoc(cv2.matchTemplate(candidate, img, cv2.TM_SQDIFF))[0] / 6000 + compare_images(candidate, img)
            img = cv2.resize(img, (candidate.shape[1], candidate.shape[0]))
            white_pixels_ratio_diff = abs(np.sum(candidate == 255) / (candidate.shape[0] * candidate.shape[1]) - np.sum(img == 255) / (img.shape[0] * img.shape[1]))
            win_size = min(candidate.shape[0], candidate.shape[1])
            win_size = win_size if win_size % 2 == 1 else win_size - 1
            return score / 1000 - 2 * ssim(candidate, img, win_size=win_size) - 10 * white_pixels_ratio_diff
        score = compare_images(candidate, img)
        img = cv2.resize(img, (candidate.shape[1], candidate.shape[0]))
        white_pixels_ratio_diff = abs(np.sum(candidate == 255) / (candidate.shape[0] * candidate.shape[1]) - np.sum(img == 255) / (img.shape[0] * img.shape[1]))
        win_size = min(candidate.shape[0], candidate.shape[1])
        win_size = win_size if win_size % 2 == 1 else win_size - 1
        return score / 1000 - 2.5 * ssim(candidate, img, win_size=win_size) * white_pixels_ratio_diff
    
    is_num = False

    for i, candidate in enumerate(candidates):
        scores = []
        for img in ground_truth:
            scores.append(eval_func(candidate, img))
        pred_char = idx2label[np.argmin(scores)]
        if not province and not is_num and (not pred_char.isalpha() or i > 2):
            is_num = True
            ground_truth = ground_truth[:10]
            idx2label = idx2label[:10]
            scores = []
            for img in ground_truth:
                scores.append(eval_func(candidate, img))
            pred_char = idx2label[np.argmin(scores)]
        res.append(pred_char)
    return res

def fix_province(pred_province):
    province_codes = ['CP', 'EP', 'NC', 'NE', 'SW', 'SB', 'SP', 'UP', 'WP']
    if len(pred_province) != 2 or pred_province in province_codes: return pred_province
    if pred_province[1] == 'P':
        if pred_province[0] == 'N':
            return 'WP'
        if pred_province[0] == 'V':
            return 'UP'
    if pred_province[0] == 'N':
        if pred_province[1] == 'N':
            return 'NW'
    return pred_province