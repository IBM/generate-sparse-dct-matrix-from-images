import os
import shutil
import pytest
import numpy as np
import tempfile

from image_processor import ImageProcessor

tmp_dir = tempfile.mkdtemp()
n_images = 5
bytes_per_image = 32 * 32 * 3


def create_random_test_images():
    images = []
    with open(os.path.join(tmp_dir, "images.bin"), "wb") as f:
        for i in range(n_images):
            img = (np.random.randint(low=0, high=255, size=(32, 32, 3))).astype(
                np.uint8
            )
            images.append(img)
            f.write(img.data.tobytes())

    # Reconstruct to test that save was successful
    with open(os.path.join(tmp_dir, "images.bin"), "rb") as f:
        images_bytes = f.read(n_images * bytes_per_image)
        reconstructed_images = []
        for i in range(n_images):
            image_bytes = images_bytes[i * bytes_per_image : (i + 1) * bytes_per_image]
            reconstructed_images.append(
                np.frombuffer(image_bytes, dtype=np.uint8).reshape(32, 32, 3)
            )
            assert (images[i] == reconstructed_images[i]).all()


create_random_test_images()


@pytest.fixture(scope="session", autouse=True)
def cleanup(request):
    """Cleanup the tmp dir"""

    def remove_test_dir():
        shutil.rmtree(tmp_dir)

    request.addfinalizer(remove_test_dir)


def test_init():
    ip = ImageProcessor(32, 3)
    assert ip.px == 32
    assert ip.color_channels == 3
    assert ip.bytes_per_image == 3072


def test_convert_byte_image_to_gray():

    with open(os.path.join(tmp_dir, "images.bin"), "rb") as f:
        raw_bytes = f.read(n_images * bytes_per_image)
        byte_images = [
            raw_bytes[i * bytes_per_image : (i + 1) * bytes_per_image]
            for i in range(n_images)
        ]

    ip = ImageProcessor(32, 3)
    for byte_image in byte_images:
        gray_image = ip.convert_byte_image_to_gray(byte_image)
        assert gray_image.shape == (32, 32)
        assert (0 <= gray_image).all()
        assert (gray_image <= 255).all()


def test_load_from_binary_file_to_grayscale():
    ip = ImageProcessor(32, 3)
    gray_images = ip.load_from_binary_file_to_grayscale(
        os.path.join(tmp_dir, "images.bin"), n_images=n_images
    )
    assert len(gray_images) == n_images


def test_save_gray_images_to_png():
    ip = ImageProcessor(32, 3)
    gray_images = ip.load_from_binary_file_to_grayscale(
        os.path.join(tmp_dir, "images.bin"), n_images=n_images
    )
    ip.save_gray_images_to_png(gray_images=gray_images, path=tmp_dir)
    image_files = os.listdir(tmp_dir)
    for i in range(n_images):
        assert f"{i}.png" in image_files


def test_compute_sparse_dct_matrix():
    frequencies_to_keep = 13
    ip = ImageProcessor(32, 3)
    gray_images = ip.load_from_binary_file_to_grayscale(
        os.path.join(tmp_dir, "images.bin"), n_images=n_images
    )
    M = ip.compute_sparse_dct_matrix(
        gray_images, frequencies_to_keep=frequencies_to_keep
    )
    assert M.shape == (n_images, 32 * 32)
    assert M.nnz == n_images * frequencies_to_keep
