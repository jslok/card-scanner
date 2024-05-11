
# Run inference on a prerecorded video

import cv2
from scanner.tools import scanner
from scanner.tools import detector as detect
from scanner.tools import tracker as track
from scanner.tools import viewer
import multiprocessing
import datetime


model = '../../rtm_det_card_trainer.py'
weights = "../../work_dirs/rtm_det_card_trainer/epoch_10.pth"
video_path = '../media/binder3.mp4'
size = 1080
scoreThreshold = .5
use_object_tracking = True
show_video = False
multiprocessing_enabled = True

# detection = {
#     bbox,mask,score,label,track_id,hash, card_image,match
# }

def main():
    print("Starting process - inference with video")

    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not video.isOpened():
        print("Error: Unable to open video.")
        return

    # Initialize the detector
    detector = detect.Detector(model, weights)

    # Initialize tracker
    tracker = track.Tracker()

    # Keep track of track_ids matched to cards
    tracked_matches = {}

    # Initialize the output video
    # output_path = 'output_video.mp4'
    # frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(video.get(cv2.CAP_PROP_FPS))

    frame_builder = viewer.VideoFrameBuilder(fps=fps,rows=1,num_images_per_frame=1, is_video=True, max_frame_size=1080)
    # Define the codec and create a VideoWriter object
    # codec = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 video
    # output_video = cv2.VideoWriter(f'output/{output_path}', codec, fps, (size, size))

    # Create a multiprocessing Pool
    pool = multiprocessing.Pool(12)

    while True:
        # Read the next frame
        ret, image_original = video.read()

        # Check if the frame was read successfully
        if not ret:
            # End of video
            break

        # Resize
        image_original = cv2.resize(image_original, (size, size))

        image_copy = image_original.copy()

        # Get detected objects
        detections = detector.detect_objects(image_original, scoreThreshold)

        if use_object_tracking is True:
            # Add object tracking IDs to detections
            tracker.track_objects(detections)

        # Reinstate matches for already tracked objects
        for detection in detections:
            track_id = detection.get('track_id')
            if track_id in tracked_matches:
                detection['match'] = tracked_matches[track_id]

        if multiprocessing_enabled is True:
            # Make hashes and matches to detections threaded
            scanner.getMatchesThreaded(pool, image_original, detections, mirror=False)
        else:
            # Make hashes and matches to detections non-threaded
            scanner.processMasksToCards(image_original, detections, mirror=False)
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

        # frame_builder.add_image(image_original)
        # Draw elements
        scanner.drawBoxes(image_copy, detections)
        # frame_builder.add_image(image_copy)
        scanner.drawMasks(image_copy, detections)
        # frame_builder.add_image(image_copy)
        scanner.writeCardLabels(image_copy, detections)
        #frame_builder.add_image(image_copy)
        scanner.writeTrackId(image_copy, detections)
        frame_builder.add_image(image_copy)
        # Write the processed frame to the output video
        #output_video.write(image_copy)

        # if len(detections) > 0:
        #     if 'card_image' in detections[0]:
        #         cv2.imshow('Image', detections[0]['card_image'])
        #         if cv2.waitKey(1) & 0xFF == ord('q'):
        #             break

        if show_video is True:
            # Display the resized frame
            cv2.imshow('Result', image_copy)

            # Check for key press to exit the loop or save a screenshot
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print('Key pressed. Ending early.')
                break
            elif key == ord('s'):  # Check if 'S' key is pressed
                current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                # Write the image to a file
                screenshot_filename = f'output/screenshots/screenshot_{current_time}.png'
                cv2.imwrite(screenshot_filename, image_copy)
                print(f"Screenshot saved to '{screenshot_filename}'")

        frame_builder.create_frame()


    # Release the video capture and writer objects
    video.release()
    # output_video.release()
    frame_builder.release()
    print('Video saved')

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
