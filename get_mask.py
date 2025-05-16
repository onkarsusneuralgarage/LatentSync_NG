import cv2
from latentsync.utils.image_processor import ImageProcessor, load_fixed_mask
from einops import rearrange
import torch
import numpy as np
import matplotlib.pyplot as plt

# im = cv2.imread("/home/tmpuser/onkar/LatentSync/latentsync/utils/mask.png")
face_mask = load_fixed_mask(256, "/home/tmpuser/onkar/LatentSync/latentsync/utils/mask.png")
image_processor = ImageProcessor(256, mask="face", device='cuda', mask_image=face_mask)
video = cv2.VideoCapture("/home/tmpuser/data3/celeb4k_results/3710.mp4")

im = []
while True:
    ret, frame = video.read()
    if not ret:
        break

    cv2.imwrite("image.jpg", frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (256, 256))
    # cv2.imwrite("image.jpg", frame)
    # frame = rearrange(torch.Tensor(frame).type(torch.uint8), "h w c ->  c h w")
    # face, masked_face, _ = image_processor.preprocess_fixed_mask_image(frame, affine_transform=False)
    # face, masked_face, mask = image_processor.preprocess_one_masked_image(frame)
    # print(frame.shape)
    # face, _, _ = image_processor.affine_transform(frame)

    im.append(frame)

    break

    # break
# face, masked_face, mask = image_processor.preprocess_one_masked_image(frame)
im = np.array(im)
print(im.shape)
face, masked_face, mask = image_processor.prepare_masks_and_masked_images(im, affine_transform=False)
print(face.shape, masked_face.shape, mask.shape)

face = (rearrange(masked_face[0], "c h w -> h w c").detach().cpu().numpy())
face = ((face + 1) * 127.5).clip(0, 255).astype(np.uint8)
plt.imsave("masked_face.jpg", face)
