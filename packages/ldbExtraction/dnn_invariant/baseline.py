import sys
sys.path.insert(0, "/home/peter/CODE/dnn_interpretation")

from dnn_invariant.utilities.environ import *
from dnn_invariant.models.models4invariant import *
from dnn_invariant.utilities.datasets import *
from dnn_invariant.algorithms.gradcam import *
from dnn_invariant.algorithms.lime_image import *
import cv2

print('Module Loading Completed')

np.set_printoptions(threshold=np.inf, precision=20)
np.random.seed(0)
torch.set_printoptions(precision=6)
torch.manual_seed(0)

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

W = 224
H = 224
I = 34

model = VGG19Assira(num_classes_=2).cuda()
model.loadModel()

img = train_data.getCNNFeature(0).cuda()

# Save Original
img_np = img.cpu().numpy().copy()
img_np = np.squeeze(img_np)
for i in range(3):
    img_np[i, :, :] = img_np[i, :, :] * std[i] + mean[i]
img_np = np.moveaxis(img_np, 0, -1)
img_np = cv2.resize(img_np, (600, 600))
img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
cv2.imwrite('Original.jpg', np.uint8(255 * img_np))

# Save Grad-CAM
mask_gradcam, img_gradcam = Rule_GradCam(model, img)
cv2.imwrite('Grad-CAM.jpg', np.uint8(255 * img_gradcam))

# Save LIME
mask_lime, img_lime = Rule_LIME(model, img)
cv2.imwrite('LIME.jpg', np.uint8(255 * img_lime))