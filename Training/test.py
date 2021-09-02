from matplotlib import pyplot as plt
import numpy as np
import movenet_train
from movenet_train import draw_prediction_on_image
from movenet_train import display_image
from movenet_train import keypoints_with_scores

output_overlay = draw_prediction_on_image(np.squeeze(display_image.numpy(), axis=0), keypoints_with_scores)
plt.figure(figsize=(5,5))
plt.imshow(output_overlay)
plt.savefig('test2.png')
print("save complete")
_= plt.axis('off')