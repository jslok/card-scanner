
# Tracks the detections between frames

import numpy as np
from sort.sort import Sort

class Tracker:
    # Initialize the SORT tracker
    def __init__(self):
        self.tracker = Sort()

    def track_objects(self, detections):
        # Extract bounding boxes from the detected objects
        if len(detections) > 0:
            bboxes = np.array([obj['bbox'] for obj in detections])
        else:
            bboxes = np.empty((0, 5))

        # Pass the bounding boxes to the SORT tracker
        tracked_objects = self.tracker.update(bboxes)

        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = obj.astype(int)

            # Iterate through detections to find the one that matches the tracked object
            for detection in detections:
                bbox = detection['bbox']
                det_x1, det_y1, det_x2, det_y2 = bbox

                # Compute IoU (Intersection over Union) between the tracked object and detection
                intersection_area = max(0, min(x2, det_x2) - max(x1, det_x1)) * max(0, min(y2, det_y2) - max(y1, det_y1))
                area_track = (x2 - x1) * (y2 - y1)
                area_det = (det_x2 - det_x1) * (det_y2 - det_y1)
                iou = intersection_area / (area_track + area_det - intersection_area)

                # If IoU is above a threshold (e.g., 0.5), consider it a match and add track_id to detection
                if iou > 0.5:
                    detection['track_id'] = track_id
                    break

        return detections
