
import cv2
from scanner.tools import scanner
from scanner.tools import detector as detect
from scanner.tools import tracker as track

model = '../rtm_det_card_trainer.py'
weights = "../work_dirs/rtm_det_card_trainer_multicard/epoch_20.pth"
size = 640
scoreThreshold = .5

# detection = {
#     bbox,mask,score,label,track_id,hash, card_image,match
# }

def main():
    print("Starting process - inference with phone cam")

    # Open the webcam
    # Replace 'http://your_phone_ip:port/video' with the actual IP address and port number
    # provided by the webcam app on your Android phone
    phone_camera_url = 'http://192.168.4.100:4747/video'

    # Open the camera feed
    phone_camera = cv2.VideoCapture(phone_camera_url)

    # Check if the camera feed opened successfully
    if not phone_camera.isOpened():
        print("Error: Unable to open phone camera.")

    # Get the current FPS of the camera
    fps = phone_camera.get(cv2.CAP_PROP_FPS)

    # Print the current config
    print("Phone cam started. fps:", fps)

    # Initialize the detector
    Detector = detect.Detector(model, weights)

    # Initialize tracker
    Tracker = track.Tracker()

    # Keep track of track_ids matched to cards
    tracked_matches = {}

    while True:
        ret, image_original = phone_camera.read()
        if not ret:
            print("Error: Unable to read frame from phone camera.")
            break

        # Resize & rotate
        image_original = cv2.resize(image_original, (size, size))
        image_original = cv2.rotate(image_original, cv2.ROTATE_90_CLOCKWISE)

        image_copy = image_original.copy()

        # Get detected objects
        detections = Detector.detect_objects(image_original, scoreThreshold)

        # Add object tracking IDs to detections
        Tracker.track_objects(detections)

        # Reinstate matches for already tracked objects
        for detection in detections:
            track_id = detection.get('track_id')
            if track_id in tracked_matches:
                detection['match'] = tracked_matches[track_id]

        # Make hashes and matches to detections
        scanner.processMasksToCards(image_original, detections, mirror=True)
        scanner.hashCards(detections)
        scanner.matchHashes(detections)

        # Store tracked matches
        for detection in detections:
            if 'track_id' in detection and 'match' in detection:
                track_id = detection['track_id']
                match = detection['match']
                if track_id not in tracked_matches:
                    tracked_matches[track_id] = match
                    print(f'Match found: id {track_id} {match}')

        # Draw elements
        scanner.drawBoxes(image_copy, detections)
        scanner.drawMasks(image_copy, detections)
        scanner.writeCardLabels(image_copy, detections)
        scanner.writeTrackId(image_copy, detections)

        # Display the resized frame
        cv2.imshow('Result Image', image_copy)

        # Check for key press to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
