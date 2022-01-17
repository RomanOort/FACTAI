from dnn_invariant.utilities.datasets import *
from dnn_invariant.algorithms.gradcam import *

def get_image(idx, data):
    img = data.getCNNFeature(idx).numpy().copy()
    img = np.squeeze(img)
    img = process_img(img)
    return img

def save_numpy_img(fname, img):
    img = np.moveaxis(img, 0, -1)
    img = cv2.resize(img, (600, 600))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(fname, np.uint8(255 * img))

def savefig(idx, name, data):
    img = get_image(idx, data)
    save_numpy_img(name + '_' + str(idx) + '.jpg', img)

for i in range(20):
    savefig(i, 'testing', train_data)
