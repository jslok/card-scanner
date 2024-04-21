
import cv2
from scanner.tools import scanner
from scanner.tools import detector as detect

model = '../rtm_det_card_trainer.py'
weights = "../work_dirs/rtm_det_card_trainer/epoch_9.pth"
imagePath = '../media/cards3.jpg'
size = 1080
scoreThreshold = .5
save_image = False
mirror = False
include_flipped = True

def main():
    print("Starting process - inference with image")

    image_original = scanner.readImage(imagePath, size)
    image_copy = image_original.copy()

    # Initialize the DetInferencer
    detector = detect.Detector(model, weights)

    detections = detector.detect_objects(image_original, scoreThreshold)

    scanner.processMasksToCards(image_original, detections, mirror)
    scanner.hashCards(detections, include_flipped)
    scanner.matchHashes(detections, include_flipped)

    scanner.drawBoxes(image_copy, detections)
    scanner.drawMasks(image_copy, detections)
    scanner.writeCardLabels(image_copy, detections)

    # for detection in detections:
    #     if 'card_image' in detection:
    #         scanner.showImageWait(detection['card_image'])

    if save_image is True:
        # Save the image
        output_path = 'output_image.jpg'
        cv2.imwrite(f'output/{output_path}', image_copy)
        print('Image saved successfully.')

    scanner.showImageWait(image_copy)


if __name__ == '__main__':
    main()
