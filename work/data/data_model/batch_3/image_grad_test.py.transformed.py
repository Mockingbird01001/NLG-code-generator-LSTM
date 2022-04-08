
from tensorflow.python.ops import image_grad_test_base as test_base
from tensorflow.python.platform import test
ResizeNearestNeighborOpTest = test_base.ResizeNearestNeighborOpTestBase
ResizeBilinearOpTest = test_base.ResizeBilinearOpTestBase
ResizeBicubicOpTest = test_base.ResizeBicubicOpTestBase
ScaleAndTranslateOpTest = test_base.ScaleAndTranslateOpTestBase
CropAndResizeOpTest = test_base.CropAndResizeOpTestBase
RGBToHSVOpTest = test_base.RGBToHSVOpTestBase
if __name__ == "__main__":
  test.main()
