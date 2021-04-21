import collections
import cv2
import numpy as np
import sys
import random
import torch

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable


class ZeroPad(object):
    """Pads the numpy array with zeros by adding the input to the center of a numpy array with size specified

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
    """

    def __init__(self, size):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)

        if isinstance(size, int):
            self.size = [size, size]
        else:
            self.size = size

    def __call__(self, img):
        """
        Args:
            img (np.array)

        Returns:
            PIL Image: Image padded with zeros
        """

        if not _is_numpy_image(img):
            raise TypeError('img should be numpy image. Got {}'.format(type(img)))

        try:
            h, w, c = img.shape
        except:
            h, w = img.shape

        assert self.size[0] >= h and self.size[1] >= w
        # we need this dtype thing so that it doesnt convert our image.
        odtype = img.dtype
        new_img = np.zeros((self.size[0], self.size[1], c), dtype=odtype)

        new_img[(self.size[0] - h) // 2:(self.size[0] - h) // 2 + h,
        (self.size[1] - w) // 2:(self.size[1] - w) // 2 + w, :] = img

        return new_img


class ResizeKeepAspect(object):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=cv2.INTER_NEAREST):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)

        if isinstance(size, int):
            self.size = [size, size]
        else:
            self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """

        if not _is_numpy_image(img):
            img = np.array(img)
            # raise TypeError('img should be numpy image. Got {}'.format(type(img)))

        try:
            h, w, c = img.shape
        except:
            h, w = img.shape

        aspect_ratio = h / w

        new_aspect_ratio = self.size[1] / self.size[0]

        if new_aspect_ratio < aspect_ratio:
            h = self.size[1]
            w = int(h / aspect_ratio)
        elif new_aspect_ratio > aspect_ratio:
            w = self.size[0]
            h = int(w * aspect_ratio)
        else:
            w = self.size[0]
            h = self.size[1]

        size = w, h
        if w == 0 or h == 0:
            print('problem resizing to h,w = {},{}'.format(h, w))
            print('original size of {}'.format(img.shape))
            w = max(w, 1)
            h = max(h, 1)
            size = w, h
        if size == self.size:
            return img

        if self.size != size:
            img = cv2.resize(img, dsize=size, interpolation=self.interpolation)
            return img


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def get_crop_params(img, output_size):
    h, w, _ = img.shape
    th, tw = output_size
    if w == tw and h == th:
        return 0, 0, h, w

    i = random.randint(0, h - th)
    j = random.randint(0, w - tw)
    return i, j, th, tw


# opencv hflipi am e
def hflip(img):
    """Horizontally flip the given numpy ndarray.
    Args:
        img (numpy ndarray): image to be flipped.
    Returns:
        numpy ndarray:  Horizontally flipped image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy image. Got {}'.format(type(img)))
    # img[:,::-1] is much faster, but doesn't work with torch.from_numpy()!
    if img.shape[2] == 1:
        return cv2.flip(img, 1)[:, :, np.newaxis]
    else:
        return cv2.flip(img, 1)


def vflip(img):
    """Horizontally flip the given numpy ndarray.
    Args:
        img (numpy ndarray): image to be flipped.
    Returns:
        numpy ndarray:  Horizontally flipped image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy image. Got {}'.format(type(img)))
    # img[:,::-1] is much faster, but doesn't work with torch.from_numpy()!
    if img.shape[2] == 1:
        return cv2.flip(img, 0)[:, :, np.newaxis]
    else:
        return cv2.flip(img, 0)


def crop(img, i, j, h, w):
    """Crop the given PIL Image.
    Args:
        img (numpy ndarray): Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
    Returns:
        numpy ndarray: Cropped image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy image. Got {}'.format(type(img)))

    return img[i:i + h, j:j + w, :]


def resize(img, size, interpolation=cv2.INTER_NEAREST):
    r"""Resize the input numpy ndarray to the given size.
    Args:
        img (numpy ndarray): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaining
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            :math:`\left(\text{size} \times \frac{\text{height}}{\text{width}}, \text{size}\right)`
        interpolation (int, optional): Desired interpolation. Default is
            ``cv2.INTER_CUBIC``
    Returns:
        PIL Image: Resized image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy image. Got {}'.format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))
    w, h, = size

    if isinstance(size, int):
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            output = cv2.resize(img, dsize=(ow, oh), interpolation=interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            output = cv2.resize(img, dsize=(ow, oh), interpolation=interpolation)
    else:
        output = cv2.resize(img, dsize=tuple(size[::-1]), interpolation=interpolation)
    if img.shape[2] == 1:
        return output[:, :, np.newaxis]
    else:
        return output


class GrayScaleTransform(object):
    """
    This transform
    """

    def __init__(self):
        """
        Args:
            severity (int, optional): Strength of the corruption, with valid values being 1 <= severity <= 5
        """
        pass

    def __call__(self, img):
        """
        Save every batch..sometimes
        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW.
        Returns:
            ndarray: transformed image(s).
        """
        dt = img.dtype
        return np.dstack([img[:, :, 0] * 0.11 + img[:, :, 1] * 0.59 + img[:, :, 2] * 0.3,
                          img[:, :, 0] * 0.11 + img[:, :, 1] * 0.59 + img[:, :, 2] * 0.3,
                          img[:, :, 0] * 0.11 + img[:, :, 1] * 0.59 + img[:, :, 2] * 0.3]).astype(dtype=dt)


def zeropad(img, input_size):
    """Pads the numpy array with zeros by adding the input to the center of a numpy array with size specified

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
    """

    assert isinstance(input_size, int) or (isinstance(input_size, Iterable) and len(input_size) == 2)

    if isinstance(input_size, int):
        input_size = [input_size, input_size]
    else:
        input_size = input_size

    if not _is_numpy_image(img):
        raise TypeError('img should be numpy image. Got {}'.format(type(img)))

    try:
        h, w, c = img.shape
    except:
        h, w = img.shape

    assert input_size[0] >= h and input_size[1] >= w
    # we need this dtype thing so that it doesnt convert our image.
    odtype = img.dtype
    new_img = np.zeros((input_size[0], input_size[1], c), dtype=odtype)

    new_img[(input_size[0] - h) // 2:(input_size[0] - h) // 2 + h, (input_size[1] - w) // 2:(input_size[1] - w) // 2 + w, :] = img

    return new_img


def resizekeepaspect(img, input_size, interpolation=cv2.INTER_NEAREST):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    assert isinstance(input_size, int) or (isinstance(input_size, Iterable) and len(input_size) == 2)

    if isinstance(input_size, int):
        input_size = [input_size, input_size]
    else:
        input_size = input_size

    if not _is_numpy_image(img):
        img = np.array(img)
        # raise TypeError('img should be numpy image. Got {}'.format(type(img)))

    try:
        h, w, c = img.shape
    except:
        h, w = img.shape

    aspect_ratio = h / w

    new_aspect_ratio = input_size[1] / input_size[0]

    if new_aspect_ratio < aspect_ratio:
        h = input_size[1]
        w = int(h / aspect_ratio)
    elif new_aspect_ratio > aspect_ratio:
        w = input_size[0]
        h = int(w * aspect_ratio)
    else:
        w = input_size[0]
        h = input_size[1]

    size = w, h
    if w == 0 or h == 0:
        print('problem resizing to h,w = {},{}'.format(h, w))
        print('original size of {}'.format(img.shape))
        w = max(w, 1)
        h = max(h, 1)
        size = w, h
    if size == input_size:
        return img

    if input_size != size:
        img = cv2.resize(img, dsize=size, interpolation=interpolation)
        return img
