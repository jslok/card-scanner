## Deep Learning & Computer Vision Trading Card Scanner

<p align="center">
  <img src="https://github.com/jslok/cardscanner/blob/master/media/demo_binder.gif?raw=true" width="400" />
  <img src="https://raw.githubusercontent.com/jslok/cardscanner/master/media/demo_mat.webp" width="400" />
</p>

This is a Pokémon card scanner that can use any video input such as webcam or phone camera to scan a card or multiple cards real-time and identify them against the complete database of currently about 20,000 English Pokémon cards. It is able to identify all types including holos, reverse holos, full art cards, etc. purely via machine learning and imaging techniques. No OCR (text recognition) is used whatsoever. This is a working proof of concept built in Python with the goal of eventually integrating it into my existing Pokémon card collection tracking app built with React Native.

There are 4 major steps to the process:

1.  Object Detection and Segmentation
2.  Object Tracking
3.  Perspective Transform
4.  Image Hashing and Identification

**1. Object Detection and Segmentation**
OpenCV is a widely-used computer vision library that can perform image processing tasks including edge detection and contouring which can be used to locate individual objects in a frame. I found it works great in cases with adequate lighting, depth, a high contrast background, and/or cards with the typical yellow border. However, it struggles detecting cards in some less ideal scenarios such as full art cards against a high-noise or similar-colored background.

To solve this issue, I used a custom trained deep learning model to do the detection phase. Object detection models only locate the bounding box of the item in the frame, or in other words, a rectangular frame that contains the object. Unless the card image is viewed perfectly straight, the bounding box contains too much other content from the background. We need to take this a step further and find not just the bounding box, but the exact pixels representing the card, also known as the segmentation mask. After evaluating several different segmentation-capable models, I found RTM-Det to be the most viable for this situation since it is very fast an lightweight, ideal for mobile deployments. It’s also not bound by restrictive licenses as some other models are, which is important if I were to use it in my own closed-source projects. RTM-Det performs instance segmentation (as opposed to just semantic segmentation) which is capable of assigning a separate mask to each detection which is what I need.

I custom trained the model on a dataset of 50,000 images I generated by compositing 2-5 random card images onto a random background. Some skew, rotation, scaling, and even overlap was applied to each card to provide extra variance and mimic cards in a real life. Only 1 class is needed, “card”. Training time for 20 epochs was about 40 hours. I achieved a segmentation mask mean average precision (mAP) of .90, which is extremely good. The generated dataset and smooth edges of the cards has a lot to do with the high mAP.

<table style="width: 400px; text-align: center; border: 0;">
  <tr>
    <td><img src="https://raw.githubusercontent.com/jslok/cardscanner/master/media/train1-1.webp" alt="alt text" style="max-width: 100%; height: auto;"></td>
    <td><img src="https://raw.githubusercontent.com/jslok/cardscanner/master/media/train1-3.webp" alt="alt text" style="max-width: 100%; height: auto;"></td>
  </tr>
</table>

<table style="width: 100%; text-align: center; border: 0;">
  <tr>
    <td>
        <img src="https://raw.githubusercontent.com/jslok/cardscanner/master/media/train2-1.webp" alt="alt text" style="max-width: 100%; height: auto;">
    </td>
    <td>
        <img src="https://raw.githubusercontent.com/jslok/cardscanner/master/media/train2-2.webp" alt="alt text" style="max-width: 100%; height: auto;">
    </td>
  </tr>
</table>

You can see clear improvements after optimizations to reach .90 mask mAP (top right) and the model can even predict cards that are occluded (bottom right). The early version of the custom trained model was trained on a dataset of only 5,000 images having 1 card each, limiting its versatility. Later versions were trained on up to 50,000 images, each having up to 5 cards with more randomness and even considerable overlap.

<table style="width: 100%; text-align: center; border: 0;">
  <tr>
    <td>
        <img src="https://raw.githubusercontent.com/jslok/cardscanner/master/media/training1.webp" alt="alt text" style="max-width: 100%; height: auto;">
    </td>
    <td>
        <img src="https://raw.githubusercontent.com/jslok/cardscanner/master/media/training2.webp" alt="alt text" style="max-width: 100%; height: auto;">
    </td>
    <td>
        <img src="https://raw.githubusercontent.com/jslok/cardscanner/master/media/training3.webp" alt="alt text" style="max-width: 100%; height: auto;">
    </td>
    <td>
        <img src="https://raw.githubusercontent.com/jslok/cardscanner/master/media/training4.webp" alt="alt text" style="max-width: 100%; height: auto;">
    </td>
  </tr>
</table>

Examples of training images with segmentation annotations drawn

**2. Object Tracking**
The transformation and identification steps can be very resource intensive, especially when scanning multiple cards together in one video feed at 60 fps. It’s best if we only perform the whole process once for each detected card, and carryover the same results through the following frames. This is where object tracking comes in. When a card comes into frame, an ID number is assigned to it. In the succeeding frames, the same ID is assigned to the same object based on having a similar size and position when comparing bounding boxes. I use the SORT algorithm to do this because it is fast and lightweight. Now we can assign the final matched result to the ID and not need to reperform the card identification process if that same ID is observed again.

**3. Perspective Transform**
The ML model outputs a binary mask of each detected object, aka the segmentation mask. We need to isolate the card from the rest of the image based on the mask and correct its orientation and any skewing so we can compare it to the database. A mask is a matrix grid with dimensions matching the image dimensions in pixels. Each index contains a binary integer (or float representing confidence score) representing whether or not that pixel is part of the object detected. We use Open CV to take each mask and use edge detection and contouring against the mask to isolate the card from the original image. Since we expect cards to have four straight edges, we can filter out any detections that have more or less than four edges. The four corners of the card are identified and pulled to reach the corners of the image canvas. This is perspective transform. A card that was skewed or crooked has been corrected to appear as if it is being viewed perfectly straight on. The card image is also oriented to portrait in this process. It may or may not be upside down, but we can worry about this later.

**4. Image Hashing and Identification**
Now how can we take the fully processed image of a card and find its match in the database? A blurry image pulled, stretched, and with some inconsistent lighting will never perfectly match pixel to pixel with the official digital glam photos. The solution is image hashing. We use an algorithm to hash an image into a binary string which is stored in a database and compared using a hamming distance formula. Images that are visually similar will have similar hash strings. Small changes to an image will result in only small changes to the hash. This is different from cryptographic hashing such as with passwords where small changes to your input can and will result in big or even complete changes to your output so it will be indiscernible.

Using only a single hashing algorithm at the default 64 bit length resulted in too many hash collisions when using such a large database of almost 20,000 cards. Two cards may not look similar, but still end up with similar hash strings. This is because part of the process of hashing is resizing the image down to a consistent size (32x32 for pHash or 8x9 for dHash) so most of the detail is lost in exchange for speed and necessary consistency. To solve this issue, I combined two different hashing algorithms together, dHash and pHash, and increased the hash length of each. Combining two algorithms together eliminates issues with a single algorithm resulting in similar hashes for non-similar cards. 128 bit length provides double the granularity over 64 bit as there is simply more data to compare.

Using a simple hamming distance formula, hash strings are compared and, within a given threshold, the pair with the lowest score means they have the highest similarities and is returned as the match. If no match is found within the threshold, it may have no match or it may be because the card is upside down. It is flipped, hashed, and checked again against the database. The threshold is adjusted to provide a good balance between allowing some interference (i.e. overlap or fingers over cards) while not giving too many incorrect matches.

<img src="https://raw.githubusercontent.com/jslok/cardscanner/master/media/demo_stages.webp"/>

[![Pokémon Binder Full Video](https://img.youtube.com/vi/u4uVJfR20iw/maxresdefault.jpg)](https://www.youtube.com/watch?v=u4uVJfR20iw)

**Conclusion**
We combine the versatility of a machine learning model with the speed of OpenCV and Image hashing to create the fastest and most versatile card scanner to my knowledge. When new cards are released, the hash database is easily updated while the model and all detection logic remains unchanged. The database used in this proof of concept only consists of Pokémon cards, but it can easily be expended to include any other types of trading cards and other types of products as well.

**Why not use deep learning for the whole process including identification?**
This would mean training a model not on a single class as I did, but on almost 20,000 classes, one for each card. Even at a mere (and likely inadequate) 1000 images per class/card, that would equal a huge training dataset of 20 million images. Not only that, but each time new cards are released, the model would need to be retrained on those new cards and redeployed. Inference with large models requires enormous amounts of memory and CPU power and may still take several seconds to produce a result. Both inference and training time would be much longer.

Remember, the goal of this project is to be able to run on-device inference real-time directly on mobile, not with some delay through a remote server. The model (plus the rest of the scanning process) must be as light and as fast as possible to run at least a few frames per second to provide a good user experience.
