import cv2
import argparse
import re
from pickle import load
from utils import *

def main(input_path, visualize=True):
    original_image = cv2.imread(input_path)
    image = preprocess_licenseplate(original_image)
    # cv2.imshow('thresh_image', thresh_image)
    # cv2.waitKey(0)
    res = detect_licenseplate(image, visualize=visualize)
    if res is None:
        print('No license plate detected.')
        return None
    else:
        plate_contour, num_candidates, province_candidates = res
        with open('./metadata/ground_truth_license.pkl', 'rb') as f:
            ground_truth_license = load(f)
        with open('./metadata/ground_truth_province.pkl', 'rb') as f:
            ground_truth_province = load(f)
        with open('./metadata/idx2label_license.pkl', 'rb') as f:
            idx2label_license = load(f)
        with open('./metadata/idx2label_province.pkl', 'rb') as f:
            idx2label_province = load(f)
        license_no = recognize_characters(num_candidates, ground_truth_license, idx2label_license)
        id = 0
        for i, c in enumerate(license_no):
            if not c.isalpha():
                id = i
                break
        license_no.insert(id, '-')
        license_no = ''.join(license_no)
        pattern = re.compile('[A-Z][A-Z][A-Z]?\-[0-9][0-9][0-9][0-9]')
        province = recognize_characters(province_candidates, ground_truth_province, idx2label_province, True)
        province = ''.join([c for c in province if c != 'UNK'])
        x, y, w, h = cv2.boundingRect(plate_contour)
        if visualize:
            final = cv2.rectangle(original_image, (x, y), (x + w, y + h), (36, 255, 12), 2)
            cv2.putText(final, f'License Plate Number: {license_no}', (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (36,255,12), 2)
            cv2.putText(final, f'Province: {province}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (36,255,12), 2)
            cv2.imshow('License plate', final)
            cv2.waitKey(0)
        if pattern.match(license_no) and province in ['CP', 'EP', 'NC', 'NE', 'SW', 'SB', 'SP', 'UP', 'WP']:
            # province = fix_province(province)
            print(f'License plate number: {license_no}')
            print(f'License plate province: {province}')
            return (x, y, w, h), num_candidates, province_candidates
        else:
            print('No license plate detected.')
            return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_image', type=str, help='Path to the input image.')
    input_path = parser.parse_args().input_image
    main(input_path)