from typing import List

import cv2
import os
import numpy as np
from scipy.sparse import csr_matrix


class ImageProcessor:
    """
    ImageProcessor: Manipulates images from the binary file
    """

    def __init__(self, px: int, color_channels: int):
        """
        Args:
            px: number of pixels along x and y axes.
            color_channels: number of color channels (e.g. 3 for RGB,
                1 for grayscale, ...)
        """
        self.px = px
        self.color_channels = color_channels
        self.bytes_per_image = self.px * self.px * self.color_channels

    def load_from_binary_file_to_grayscale(
        self, binary_file_path: str, n_images: int, start_index: int = 0
    ) -> List[np.ndarray]:
        """
        Loads a set of images (in bytes) from the binary file.
        Returns a list of images in grayscale.

        Args:
            binary_file_path: path to the file.
            n_images: number of images to load.
            start_index: index of the first image. Defaults to zero.

        Returns:
            List of images in grayscale
        """

        with open(binary_file_path, "rb") as f:
            f.seek(start_index * self.bytes_per_image)
            image_bytes = f.read(self.bytes_per_image * n_images)
            f.close()
        gray_images = [
            self.convert_byte_image_to_gray(image_bytes[i : i + self.bytes_per_image])
            for i in range(0, len(image_bytes), self.bytes_per_image)
        ]

        return gray_images

    def convert_byte_image_to_gray(self, byte_image) -> np.ndarray:
        """
        Converts an image from byte format to gray scale.
        Args:
            byte_image: image in bytes.

        Returns:
            The image in gray scale.

        """
        nparr = np.frombuffer(byte_image, dtype=np.uint8)
        bgr_image = nparr.reshape(
            self.px, self.px, self.color_channels, order="F"
        ).copy()
        return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    def save_gray_images_to_png(self, gray_images: List[np.ndarray], path: str) -> None:
        """
        Save a list of grayscale images to png format.

        Args:
             gray_images: list of gray images.
             path: path to store the images.
        """

        if not os.path.isdir(path):
            os.mkdir(path)
        for i, img in enumerate(gray_images):
            cv2.imwrite(os.path.join(path, "{}.png".format(i)), img)

    def compute_sparse_dct_matrix(
        self, gray_images: List[np.ndarray], frequencies_to_keep: int = 20
    ) -> csr_matrix:
        """
        Compute the DCT of each gray image and keep only the top frequencies_to_keep elements.
        Return a matrix representation of the sparse transforms.

        Args:
             gray_images: list of images in grayscale.
             frequencies_to_keep: number of frequencies to keep.

        Returns:
            csr_matrix: A sparse matrix where each row corresponds to an image, and it has
                exactly frequencies_to_keep nonzero elements
        """
        dct_indices = []
        dct_data = []
        for image in gray_images:
            image_dct = cv2.dct(image.astype(float))
            indices = np.argsort(np.abs(image_dct.ravel()))[-frequencies_to_keep:][::-1]
            dct_indices.extend(list(indices))
            dct_data.extend(list(image_dct.ravel()[indices]))

        dim = self.px * self.px
        indptr = list(range(0, len(dct_indices) + 1, frequencies_to_keep))
        A = csr_matrix((dct_data, dct_indices, indptr), shape=(len(gray_images), dim))
        return A
