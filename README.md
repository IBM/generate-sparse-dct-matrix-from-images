# generate-sparse-dct-matrix-from-images
Snippet to generate a sparse matrix from a collection of images in binary format.
Each image is first converted in grayscale and then flattened to a vector.
A 2d Discrete Cosine Transform (DCT) is applied in order to isolate the top frequencies for each image, leading to a sparse vector representation in the frequency domain.
The sparse vectors are stacked to form a matrix where each row is the sparsified DCT of the corresponding image.
For details on the usage check the test file.

A preliminary version of this code was used to generate test matrices in the following publication:
https://epubs.siam.org/doi/abs/10.1137/20M1314471

If you use this code for related academic work consider citing this work.
