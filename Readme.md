## cvnp: pybind11 casts and transformers between numpy and OpenCV, with shared memory

cvnp provides automatic cast between OpenCV matrices and numpy arrays when using pybind11:

* `cv::Mat` and `cv::Mat_<Tp>`: standard OpenCV matrices are transformed to numpy array *with shared memory*
  (i.e. modification to matrices elements made from python are immediately visible to C++, and vice-versa).
* Sub-matrices:
  * Sub-matrices created from C++ can also be shared to python
  * Sub-matrices created from python will need to be transformed to a contiguous array before being shared to C++
* `cv::Matx`: small matrices are transformed to numpy arrays *without shared memory*
* Casts *without* shared memory for simple types, between `cv::Size`, `cv::Point`, `cv::Point3`, `cv::Scalar_<Tp>`, `cv::Rect_<Tp>` and python `tuple`

> Note: The API evolved on May 4th 2023: see [breaking changes](https://github.com/pthom/cvnp#breaking-changes)

### Explicit transformers

#### Explicit transformers between cv::Mat and numpy.ndarray, *with* shared memory

```cpp
pybind11::array mat_to_nparray(const cv::Mat& m);
cv::Mat         nparray_to_mat(pybind11::array& a);
```

#### Explicit transformers between cv::Matx and numpy.ndarray *without* shared memory

```cpp
template<typename _Tp, int _rows, int _cols> pybind11::array    matx_to_nparray(const cv::Matx<_Tp, _rows, _cols>& m);
template<typename _Tp, int _rows, int _cols> void               nparray_to_matx(pybind11::array &a, cv::Matx<_Tp, _rows, _cols>& out_matrix);
```

### Supported matrix types

Since OpenCV supports a subset of numpy types, here is the table of supported types:

```
➜ python
>>> import cvnp
>>> cvnp.print_types_synonyms()
  cv_depth   cv_depth_name   np_format   np_format_long
     0          CV_8U           B         np.uint8  
     1          CV_8S           b         np.int8   
     2          CV_16U          H        np.uint16  
     3          CV_16S          h         np.int16  
     4          CV_32S          i         np.int32  
     5          CV_32F          f          float    
     6          CV_64F          d        np.float64
```


## How to use it in your project

1. Add cvnp to your project. For example:

```bash
cd external
git submodule add https://github.com/pthom/cvnp.git
```

2. Link it to your python module:

In your python module CMakeLists, add:

```cmake
add_subdirectory(path/to/cvnp)
target_link_libraries(your_target PRIVATE cvnp)
```

3. In your module, include cvnp:

```cpp
#include "cvnp/cvnp.h"
```

4. (Optional) If you want to import the declared functions in your module:

Write this in your main module code:
```cpp
void pydef_cvnp(pybind11::module& m);

PYBIND11_MODULE(your_module, m)
{
    ....
    ....
    ....
    pydef_cvnp(m);
}
```

You will get two simple functions:
* cvnp.list_types_synonyms()
* cvnp.print_types_synonyms()

```python
import cvnp
import pprint
pprint.pprint(cvnp.list_types_synonyms(), indent=2, width=120)
[ {'cv_depth': 0, 'cv_depth_name': 'CV_8U', 'np_format': 'B', 'np_format_long': 'np.uint8'},
  {'cv_depth': 1, 'cv_depth_name': 'CV_8S', 'np_format': 'b', 'np_format_long': 'np.int8'},
  {'cv_depth': 2, 'cv_depth_name': 'CV_16U', 'np_format': 'H', 'np_format_long': 'np.uint16'},
  {'cv_depth': 3, 'cv_depth_name': 'CV_16S', 'np_format': 'h', 'np_format_long': 'np.int16'},
  {'cv_depth': 4, 'cv_depth_name': 'CV_32S', 'np_format': 'i', 'np_format_long': 'np.int32'},
  {'cv_depth': 5, 'cv_depth_name': 'CV_32F', 'np_format': 'f', 'np_format_long': 'float'},
  {'cv_depth': 6, 'cv_depth_name': 'CV_64F', 'np_format': 'd', 'np_format_long': 'np.float64'}]
```


### Demo with cv::Mat : shared memory and sub-matrices

Below is on extract from the test [test/test_cvnp.py](tests/test_cvnp.py):

```python
def test_cpp_sub_matrices():
  """
  We are playing with these bindings:
      struct CvNp_TestHelper {
          // m10 is a cv::Mat with 3 float channels
          cv::Mat m10 = cv::Mat(cv::Size(100, 100), CV_32FC3, cv::Scalar(0.f, 0.f, 0.f));
          // GetSubM10 returns a sub-matrix of m10
          cv::Mat GetSubM10() { return m10(cv::Rect(1, 1, 3, 3)); }
          // Utilities to trigger value changes made by C++ from python 
          void SetM10(int row, int col, cv::Vec3f v) { m10.at<cv::Vec3f>(row, col) = v; }
          cv::Vec3f GetM10(int row, int col) { return m10.at<cv::Vec3f>(row, col); }
          ...
      };
  """
  o = CvNp_TestHelper()

  #
  # 1. Transform cv::Mat and sub-matrices into numpy arrays / check that reference counts are handled correctly
  #
  # Transform the cv::Mat m10 into a linked numpy array (with shared memory) and assert that m10 now has 2 references
  m10: np.ndarray = o.m10
  assert o.m10_refcount() == 2
  # Also transform the m10's sub-matrix into a numpy array, and assert that m10's references count is increased
  sub_m10 = o.GetSubM10()
  assert o.m10_refcount() == 3

  #
  # 2. Modify values from C++ or python, and ensure that the data is shared
  #
  # Modify a value in m10 from C++, and ensure this is visible from python
  val00 = np.array([1, 2, 3], np.float32)
  o.SetM10(0, 0, val00)
  assert (m10[0, 0] == val00).all()
  # Modify a value in m10 from python and ensure this is visible from C++
  val10 = np.array([4, 5, 6], np.float32)
  o.m10[1, 1] = val10
  assert (o.m10[1, 1] == val10).all()

  #
  # 3. Check that values in sub-matrices are also changed
  #
  # Check that the sub-matrix is changed
  assert (sub_m10[0, 0] == val10).all()
  # Change a value in the sub-matrix from python
  val22 = np.array([7, 8, 9], np.float32)
  sub_m10[1, 1] = val22
  # And assert that the change propagated to the master matrix
  assert (o.m10[2, 2] == val22).all()

  #
  # 4. del python numpy arrays and ensure that the reference count is updated
  #
  del m10
  del sub_m10
  assert o.m10_refcount() == 1

  #
  # 5. Sub-matrices are supported from C++ to python, but not from python to C++!
  #
  # i. create a numpy sub-matrix
  full_matrix = np.ones([10, 10], np.float32)
  sub_matrix = full_matrix[1:5, 2:4]
  # ii. Try to copy it into a C++ matrix: this should raise a `ValueError`
  with pytest.raises(ValueError):
    o.m = sub_matrix
  # iii. However, we can update the C++ matrix by using a contiguous copy of the sub-matrix
  sub_matrix_clone = np.ascontiguousarray(sub_matrix)
  o.m = sub_matrix_clone
  assert o.m.shape == sub_matrix.shape
```


#### Demo with cv::Matx : no shared memory

Below is on extract from the test [test/test_cvnp.py](tests/test_cvnp.py):

```python
def test_matx_not_shared():
  """
  We are playing with these elements
      struct CvNp_TestHelper {
          cv::Matx32d mx_ns = cv::Matx32d::eye();
          void SetMX_ns(int row, int col, double v) { mx_ns(row, col) = v;}
          ...
      };
  """
  # create object
  o = CvNp_TestHelper()

  m_linked = o.mx_ns                   # Make a numy array that is a copy of mx_ns *without* shared memory
  assert m_linked.shape == (3, 2)      # check its shape
  m_linked[1, 1] = 3                   # a value change in the numpy array made from python
  assert o.mx_ns[1, 1] != 3            # is not visible from C++!

  o.SetMX_ns(2, 1, 15)                             # A C++ change a value in the matrix
  assert not are_float_close(m_linked[2, 1], 15)   # is not visible from python,
  m_linked = o.mx_ns                               # but becomes visible after we re-create the numpy array from
  assert are_float_close(m_linked[2, 1], 15)       # the cv::Matx
```


## Build and test

_These steps are only for development and testing of this package, they are not required in order to use it in a different project._

### install python dependencies (opencv-python, pytest, numpy)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Install C++ dependencies (pybind11, OpenCV)

You will need to have `OpenCV` installed on your system (you can use `vcpkg` or your package manager).

### Build

You need to specify the path to the python executable:

```bash
mkdir build && cd build
cmake .. -DPython_EXECUTABLE=../venv/bin/python
make
```

### Test

In the build dir, run:

```
cmake --build . --target test
```

(this will run native C++ tests and python tests)

### Deep clean

```
rm -rf build
rm -rf venv
rm -rf .pytest_cache
rm  *.so 
rm *.pyd
```


## Notes

Thanks to Dan Mašek who gave me some inspiration here:
https://stackoverflow.com/questions/60949451/how-to-send-a-cvmat-to-python-over-shared-memory

This code is intended to be integrated into your own pip package. As such, no pip tooling is provided.

## Breaking changes

This library was updated in May 2023, with breaking changes from previous versions:

* Previously, it was required to use the (now defunct) `cvnp::Mat_shared` in order to share memory between cv::Mat and a numpy array.
  The memory is now *always* shared between cv::Mat and numpy arrays, since it was shown that this is faster and safe.
* cv::Matx cannot share memory with a numpy array

For those interested in the previous API, it is still available in the [original_api](https://github.com/pthom/cvnp/tree/original_api) branch.
