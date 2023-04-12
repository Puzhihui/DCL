import sys
sys.path.insert(0, '../')
import cv2
import numbers
import numpy as np
from cvtorchvision import cvtransforms



def cv_loader(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb


def cv_center_crop(img, size=[448, 448]):  # size [h,w]
    height = img.shape[0]
    width = img.shape[1]

    cropW = size[0]
    cropH = size[1]
    width_crop = cropW if width > cropW else width
    height_crop = cropH if height > cropH else height
    centerX = width / 2
    centerY = height / 2

    x0 = int(centerX - width_crop / 2)
    x1 = int(centerX + width_crop / 2)
    y0 = int(centerY - height_crop / 2)
    y1 = int(centerY + height_crop / 2)

    # cropped_image = img[124:956, 124:956]
    cropped_image = img[y0:y1, x0:x1]
    return cropped_image


def cv_resize(img, size=[448, 448]):  # size [h,w]
    height = size[0]
    width = size[1]
    img_resize = cv2.resize(img, (width, height))
    return img_resize

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def crop(img, x, y, h, w):
    """Crop the given CV Image.
    Args:
        img (np.ndarray): Image to be cropped.
        x: Upper pixel coordinate.
        y: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
    Returns:
        CV Image: Cropped image.
    """
    assert _is_numpy_image(img), 'img should be CV Image. Got {}'.format(type(img))
    assert h > 0 and w > 0, 'h={} and w={} should greater than 0'.format(h, w)

    x1, y1, x2, y2 = round(x), round(y), round(x+h), round(y+w)

    try:
        check_point1 = img[x1, y1, ...]
        check_point2 = img[x2-1, y2-1, ...]
    except IndexError:
        # warnings.warn('crop region is {} but image size is {}'.format((x1, y1, x2, y2), img.shape))
        img = cv2.copyMakeBorder(img, - min(0, x1), max(x2 - img.shape[0], 0),
                                 -min(0, y1), max(y2 - img.shape[1], 0), cv2.BORDER_CONSTANT, value=[0, 0, 0])
        y2 += -min(0, y1)
        y1 += -min(0, y1)
        x2 += -min(0, x1)
        x1 += -min(0, x1)

    finally:
        return img[x1:x2, y1:y2, ...].copy()


def center_crop(img, output_size):
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    h, w = img.shape[:2]
    th, tw = output_size
    i = int(round((h - th) * 0.5))
    j = int(round((w - tw) * 0.5))
    return crop(img, i, j, th, tw)


transform_cv = cvtransforms.Compose([
                              #cvtransforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 0)),
                              #cvtransforms.Resize(size=(350, 350), interpolation='BILINEAR'),
    # cvtransforms.CenterCrop((448, 448)),
    # cvtransforms.RandomVerticalFlip(1),
    # cvtransforms.RandomHorizontalFlip(1),
    # cvtransforms.ColorJitter(0.2, 0.1, 0.1, 0.01),
    # cvtransforms.RandomRotation(degrees=15),
    cvtransforms.Resize(size=(448, 448), interpolation='NEAREST'),
    cvtransforms.ToTensor(),
    cvtransforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

from transforms import transforms
from PIL import Image
transform_torch = transforms.Compose([
    # transforms.CenterCrop((448, 448)),
    # transforms.RandomVerticalFlip(1),
    # transforms.RandomHorizontalFlip(1),
    # transforms.ColorJitter(0.2, 0.1, 0.1, 0.01),
    # transforms.RandomRotation(degrees=15),
    transforms.Resize((448,448), interpolation=Image.NEAREST),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

if __name__ == '__main__':
    img1 = r'/data3/pzh/data/tensorrt/jpg/NPS2138155B01@D00003264-R51-C51-90979970--72960060#540-540-44231.jpg'
    img_rgb = cv_loader(img1)

    cv = transform_cv(img_rgb)

    img_cv2_PIL = Image.fromarray(img_rgb)
    torch_t = transform_torch(img_cv2_PIL)

    print(cv.equal(torch_t))