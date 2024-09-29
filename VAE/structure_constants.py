class StructureConstants:
    INPUT_SHAPE = (1, 256, 128)

    FILTER0 = INPUT_SHAPE[0]

    FILTER1 = 256
    FILTER2 = 128
    FILTER3 = 64
    FILTER4 = 32
    FILTER5 = 16

    KERNEL = 3
    STRIDE = 2

    H_DIM = 512  # dependent to input_shape, #filters, kernel, strides
    Z_DIM = 128

    SIZE_BEFORE_LATENT = (FILTER5, 8, 4)  # dependent to input_shape, #filters
