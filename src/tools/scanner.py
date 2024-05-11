
# Collection of functions that process the detections from the detector

import cv2
import numpy as np
import imagehash
from PIL import Image
import json
import datetime
import viewer

hash_size = 16 # bytes
hash_filename = 'hashes_dphash_16byte.json'
min_similarity = 14*6.8 # lower is more exact
check_flipped = False # if no match, rotate 180 and check again

with open(f'data/{hash_filename}', 'r', encoding='utf-8') as json_file:
    hash_dict = json.load(json_file)

def get_match_thread(args):
    detection, image, mirror = args
    if 'match' in detection:
        return detection['match']
    if 'track_id' not in detection:
        return None
    card_image = perspective_transform(image, detection['mask'], mirror)
    if card_image is None:
        return None
    image_hash = hashImage(card_image)
    match, _ = findMatch(image_hash)
    if match is None and check_flipped is True:
        match = findFlippedMatch(card_image)
    if match is not None:
        return match
    return None


def getMatchesThreaded(pool, image, detections, mirror=False):
    results = [pool.apply_async(get_match_thread, args=((detection, image, mirror),)) for detection in detections]
    # Update detections with the results
    for i, result in enumerate(results):
        match = result.get()  # This will block until the result is ready
        if match is not None:
            detections[i]['match'] = match



#hashes a cv2 image
def hashImage(img):
    # Convert the NumPy array to a PIL image
    img = Image.fromarray(img)

    img = img.convert('RGB')

    # Resize the image to the desired size
    #img = img.resize((image_size, image_size))

    # Compute the hash
    # ahash = str(imagehash.average_hash(img, hash_size))
    dhash = imagehash.dhash(img, hash_size)
    phash = imagehash.phash(img, hash_size)

    hash = f'{dhash}{phash}'

    return hash


# Hamming distance calculation function. Lower is better
def hamming_distance(hash1, hash2):
    assert len(hash1) == len(hash2), "Hash lengths are not equal"
    return sum(ch1 != ch2 for ch1, ch2 in zip(hash1, hash2))


def findMatch(hash_a):
    best_match = None
    min_sim = min_similarity

    for card_id, data in hash_dict.items():
        hash_b = data['hash']
        similarity = hamming_distance(hash_a, hash_b)
        if similarity < min_sim:
            min_sim = similarity
            best_match = card_id

    if best_match is None:
        return None, None

    matchString = f"{hash_dict[best_match]['name']} {hash_dict[best_match]['id']}"

    return matchString, min_sim


def findFlippedMatch(card_image):
    card_image = cv2.rotate(card_image, cv2.ROTATE_180)
    image_hash = hashImage(card_image)
    match, _ = findMatch(image_hash)
    return match



def writeTrackId(image, detections):
    for detection in detections:
        bbox = detection['bbox']
        track_id = detection.get('track_id')
        if track_id is None:
            continue
        x1, y1, x2, y2 = bbox
        # cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, str(track_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def writeCardLabels(image, detections):
    for detection in detections:
        if 'match' not in detection or detection['match'] is None:
            continue
        bbox = detection['bbox']
        # Calculate the center of the bounding box
        center_x = int((bbox[0] + bbox[2]) / 2)
        center_y = int((bbox[1] + bbox[3]) / 2)
        center = (center_x, center_y)
        writeLabel(image, center, detection['match'])


def writeLabel(image, loc, text):
    [center_x, center_y] = loc
    font=cv2.FONT_HERSHEY_SIMPLEX
    font_scale=.6
    text_color=(0, 0, 0)
    background_color=(255, 255, 255)
    thickness=2

    # Get the size of the text
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

    # Calculate the position to start drawing the text (centered)
    text_x = int(center_x - text_size[0] / 2)
    text_y = int(center_y + text_size[1] / 2)

    # Calculate the bounding box for the background rectangle
    background_left = text_x
    background_top = text_y - text_size[1]
    background_right = text_x + text_size[0]
    background_bottom = text_y + 2

    # Draw the filled rectangle as the background
    cv2.rectangle(image, (background_left, background_top), (background_right, background_bottom), background_color, -1)

    # Write the text on top of the background
    cv2.putText(image, text, (text_x, text_y), font, font_scale, text_color, thickness)


def hashCards(detections, include_flipped=False):
    for detection in detections:
        # Skip if match is already found and tracked
        if 'match' in detection:
            continue
        if 'card_image' in detection:
            image_hash = hashImage(detection['card_image'])
            #print(image_hash)
            detection['hash'] = image_hash
            if include_flipped is True:
                card_image = cv2.rotate(detection['card_image'], cv2.ROTATE_180)
                image_hash = hashImage(card_image)
                detection['hash_flipped'] = image_hash


def matchHashes(detections, include_flipped=False):
    for detection in detections:
        # Skip if match is already found and tracked
        if 'match' in detection:
            continue
        if 'hash' in detection:
            # print(detection['hash'])
            match1, sim1 = findMatch(detection['hash'])
            if include_flipped is True:
                match2, sim2 = findMatch(detection['hash_flipped'])
                if match1 is None and match2 is None:
                    continue
                elif match1 is None:
                    detection['match'] = match2
                elif match2 is None:
                    detection['match'] = match1
                else:
                    # Compare similarity scores
                    if sim2 > sim1:
                        detection['match'] = match1
                    else:
                        detection['match'] = match2
            if 'match' in detection:
                continue
            elif match1 is None and check_flipped is True:
                match1 = findFlippedMatch(detection['card_image'])
            if match1 is not None:
                detection['match'] = match1


def drawBoxes(image, detections):
    for detection in detections:
        drawBox(image, detection['bbox'])


def drawMasks(image, detections):
    for detection in detections:
        drawMask(image, detection['mask'])


def processMasksToCards(image, detections, mirror):
    for detection in detections:
        # Skip if match is already found and tracked
        if 'match' in detection:
            continue
        card_image = perspective_transform(image, detection['mask'], mirror)
        if card_image is not None:
            #print('not none')
            detection['card_image'] = card_image

def readFrame(camera, size):
    # Read a frame from the webcam
    ret, frame = camera.read()

    # Check if the frame is successfully read
    if not ret:
        print("Failed to read frame from the webcam")
        return None

    # Resize the frame to the desired size
    resized_frame = cv2.resize(frame, (size, size))
    return resized_frame


# Read and resize the image
def readImage(img, size):
    img = cv2.imread(img)
    img = cv2.resize(img, (size, size))
    return img


def show_contour(contour, original_image):
    # Create a blank canvas with the same size and aspect ratio as the original image
    canvas = np.zeros_like(original_image)

    # Draw the contour on the canvas
    cv2.drawContours(canvas, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

    # Show the canvas
    cv2.imshow('Contour', canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def perspective_transform(image, mask, mirror):
    # Convert the boolean mask to uint8
    mask_uint8 = mask.astype(np.uint8) * 255

    # Visualize the mask
    h, w = mask_uint8.shape
    mask_visualized = np.zeros((h, w, 3), dtype=np.uint8)
    # Set the background pixels to black
    mask_visualized[:, :] = [0, 0, 0]
    # Set the object mask pixels to red
    mask_visualized[mask_uint8 != 0] = [0, 0, 255]
    # viewer.VideoFrameBuilder().add_image(mask_visualized, 2, "Segmentation Mask")

    # Find contours in the mask
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_with_detections = image.copy()

    # Approximate the contours to get the four corners of the card
    for contour in contours:
        epsilon = 0.1 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            # Rearrange the points in the approx array if needed
            approx = reorder_points(approx)

            # Draw the contour on the image
            cv2.drawContours(image_with_detections, [approx], -1, (0, 0, 255), 2)

            # Draw the corners on the image
            circle_size = 8
            for point in approx:
                x, y = point.ravel()
                cv2.circle(image_with_detections, (x, y), circle_size, (0, 255, 0), -1)

            viewer.VideoFrameBuilder().add_image(image_with_detections, 2, "Edge & Corner Detection")

            # Perform perspective transform
            width, height = get_card_dimensions(approx)
            dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(approx.astype(np.float32), dst)

            warped = cv2.warpPerspective(image, M, (int(width), int(height)))

            # Rotate the warped image to ensure portrait orientation
            if width > height:
                warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

            # Resize the warped image to fill the canvas without maintaining aspect ratio
            warped_stretched = cv2.resize(warped, (320, 320))

            if mirror is True:
                warped_stretched = cv2.flip(warped_stretched, 1)

            return warped_stretched

    return None


def reorder_points(points):
    # Calculate centroids
    centroids = np.mean(points, axis=0)

    # Sort points based on distance from centroids
    points_sorted = sorted(points, key=lambda x: np.arctan2(x[0][1] - centroids[0][1], x[0][0] - centroids[0][0]))

    return np.array(points_sorted)


def get_card_dimensions(corners):
    # Calculate the width and height of the card
    width = np.linalg.norm(corners[0] - corners[1])
    height = np.linalg.norm(corners[1] - corners[2])
    return width, height


def drawBox(img, bbox):
    xmin, ymin, xmax, ymax = map(int, bbox)
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)


def drawMask(original_image, mask):
    color = (0, 0, 255)
    alpha = 0.4

    # Convert mask to binary mask (0s and 255s)
    binary_mask = np.array(mask, dtype=np.uint8)

    # Create a copy of the original image
    overlay = original_image.copy()

    # Apply the color to the mask region
    overlay[binary_mask != 0] = color

    # Blend the overlay with the original image using alpha blending
    result = cv2.addWeighted(overlay, alpha, original_image, 1 - alpha, 0)

    # Update the original image with the result
    original_image[:, :] = result


def save_screenshot(img):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Write the image to a file
    screenshot_filename = f'output/screenshots/screenshot_{current_time}.png'
    cv2.imwrite(screenshot_filename, img)
    print(f"Screenshot saved to '{screenshot_filename}'")


def showImageWait(img):
    print('Press s for screenshot or any key to continue')
    cv2.imshow('Image', img)
    key = cv2.waitKey(0) & 0xFF
    if key == ord('s'):  # Check if 'S' key is pressed
        save_screenshot(img)
    # cv2.destroyAllWindows() # Close if not 'S'
