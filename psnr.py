import cv2
import math
import numpy


def psnr(target, ref):
    # assume RGB image
    target_data = numpy.array(target, dtype=float)
    ref_data = numpy.array(ref, dtype=float)

    diff = ref_data - target_data
    diff = diff.flatten('C')

    rmse = math.sqrt(numpy.mean(diff ** 2.))

    return 20 * math.log10(255. / rmse)


if __name__ == "__main__":
    im1 = cv2.imread("./input.jpg", cv2.IMREAD_COLOR)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2YCrCb)[6: -6, 6: -6, 0]
    im2 = cv2.imread("./butterfly_GT.bmp", cv2.IMREAD_COLOR)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2YCrCb)[6: -6, 6: -6, 0]
    im3 = cv2.imread("pre_adam2000.jpg", cv2.IMREAD_COLOR)
    im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2YCrCb)[6: -6, 6: -6, 0]
    im4 = cv2.imread("./pre.jpg", cv2.IMREAD_COLOR)
    im4 = cv2.cvtColor(im4, cv2.COLOR_BGR2YCrCb)[6: -6, 6: -6, 0]

    print "adam:"
    print psnr(im2, im3)
    print "bicubic:"
    print psnr(im2, im1)
    print "SRCNN:"
    print psnr(im2, im4)
