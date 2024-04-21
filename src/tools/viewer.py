import cv2
import numpy as np
import datetime
from scanner.tools import scanner

class VideoFrameBuilder:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


    def __init__(self, image_size=1080, max_frame_size=1080, num_images_per_frame=4, rows=2, fps=60, show_feed=True, is_video = True):
        if getattr(self, '_initialized', None) is None:
            self.image_size = image_size
            self.max_frame_size = max_frame_size
            self.num_images_per_frame = num_images_per_frame
            self.image_array = [None] * num_images_per_frame
            self.rows = rows
            self.last_frame = None
            self.show_feed = show_feed
            self.video_writer = None
            self.fps = fps
            self.is_video = 1 if is_video else 0
            self._initialized = True
            if show_feed is True:
                print('Showing feed. Press q to quit or s to take screenshot.')


    def add_image(self, image=None, i=None, label=None):
        width = self.image_size
        height = self.image_size
        if image is None:
            image = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            image = make_square_and_resize(image, width, height)
        if label is not None:
            font_scale = 2
            # Measure the size of the text
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
            # Calculate the position to place the text and the box
            text_x = (image.shape[1] - text_size[0]) // 2
            text_y = text_size[1] + 30
            box_top_left = (text_x - 5, text_y - text_size[1] - 5)
            box_bottom_right = (text_x + text_size[0] + 5, text_y + 5)
            # Draw the white box
            cv2.rectangle(image, box_top_left, box_bottom_right, (255, 255, 255), cv2.FILLED)
            # Add label as text at the top of the image
            cv2.putText(image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
        if i is not None:
            if type(i) != int or i >= len(self.image_array):
                print('Error adding image. Invalid value for index.')
                return
            self.image_array[i] = image
        else:
            for i, data in enumerate(self.image_array):
                if data is None:
                    self.image_array[i] = image
                    return
            print('Error adding image. Array is full.')


    def create_frame(self):
        if self.image_array:
            num_images = len(self.image_array)
            num_columns = int(np.ceil(num_images / self.rows))

            # Create a list to store images for each row
            rows_images = []

            # Split the images into rows
            for i in range(self.rows):
                start_index = i * num_columns
                end_index = min(start_index + num_columns, num_images)
                row_images = self.image_array[start_index:end_index]

                # Fill None images with black images
                row_images = [image if image is not None else np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8) for image in row_images]

                row_images = [cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image for image in row_images]

                rows_images.append(row_images)

            # Concatenate images for each row vertically
            row_frames = [np.hstack(row_images) for row_images in rows_images]

            # Concatenate row frames horizontally to form the final frame
            frame = np.vstack(row_frames)

            # Resize frame if necessary
            frame = self.resize_frame_maintain_aspect(frame)

            self.last_frame = frame

            self.save_frame(frame)

            if self.show_feed is True:
                self.show_last_frame()

            # Reset image_array
            self.image_array = [None] * self.num_images_per_frame


    def show_last_frame(self):
        # Display or save the frame
        if self.last_frame is not None:
            cv2.imshow('Frame', self.last_frame)
            # Check for key press to save a screenshot or quit
            key = cv2.waitKey(self.is_video) & 0xFF
            if key == ord('s'):  # Check if 'S' key is pressed
                scanner.save_screenshot(self.last_frame)
            elif key == ord('q'):  # Check if 'q' key is pressed
                exit()  # Close the window


    def save_frame(self, frame):
        if self.video_writer is None:
            height, width = frame.shape[:2]
            # Initialize the output video
            codec = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(f'output/output_video.mp4', codec, self.fps, (width, height))
        self.video_writer.write(frame)


    def release(self):
        self.video_writer.release()
        print('Saved video to output/output_video.mp4')


    def resize_frame_maintain_aspect(self, frame):
        width = self.max_frame_size
        height = self.max_frame_size
        inter = cv2.INTER_AREA

        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = frame.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return frame

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(frame, dim, interpolation = inter)

        # return the resized image
        return resized


def make_square_and_resize(frame, width, height):
    # Determine the dimensions of the frame
    frame_height, frame_width = frame.shape[:2]

    # Make the frame square by adding black padding
    max_dim = max(frame_height, frame_width)
    square_frame = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
    x_offset = (max_dim - frame_width) // 2
    y_offset = (max_dim - frame_height) // 2
    square_frame[y_offset:y_offset+frame_height, x_offset:x_offset+frame_width] = frame

    # Resize the frame to the given width and height
    resized_frame = cv2.resize(square_frame, (width, height))

    return resized_frame


# Example usage:
if __name__ == "__main__":
    frame_builder = VideoFrameBuilder()

    # Assuming you have a list of images
    image_list = [cv2.imread('media/cards1.jpg'), cv2.imread('media/cards2.jpg'), cv2.imread('media/cards3.jpg'), cv2.imread('media/cards4.jpg')]

    # Adding images to the frame builder
    for image in image_list:
        frame_builder.add_image(image)

    # Create frame from remaining images in the array
    frame_builder.create_frame()
