## cvnp: pybind11 casts and transformers between numpy and OpenCV, possibly with shared memory


### Explicit transformers between cv::Mat / cv::Matx and numpy.ndarray, with or without shared memory

Notes:
- When going from Python to C++ (nparray_to_mat), the memory is *always* shared
- When going from C++ to Python (mat_to_nparray) , you have to _specify_ whether you want to share memory via 
the boolean parameter `share_memory`

````cpp
    pybind11::array mat_to_nparray(const cv::Mat& m, bool share_memory);
    cv::Mat         nparray_to_mat(pybind11::array& a);

        template<typename _Tp, int _rows, int _cols>
    pybind11::array matx_to_nparray(const cv::Matx<_Tp, _rows, _cols>& m, bool share_memory);
        template<typename _Tp, int _rows, int _cols>
    void            nparray_to_matx(pybind11::array &a, cv::Matx<_Tp, _rows, _cols>& out_matrix);
````


> __Warning__: be extremely cautious of the lifetime of your Matrixes when using shared memory!
For example, the code below is guaranted to be a definitive UB, and a may cause crash much later.

````cpp

pybind11::array make_array()
{
    cv::Mat m(cv::Size(10, 10), CV_8UC1);               // create a matrix on the stack
    pybind11::array a = cvnp::mat_to_nparray(m, true);  // create a pybind array from it, using
                                                        // shared memory, which is on the stack!
    return a;                                                        
}  // Here be dragons, when closing the scope!
   // m is now out of scope, it is thus freed, 
   // and the returned array directly points to the old address on the stack!
````


### Automatic casts:

#### Without shared memory

* Casts *without* shared memory between `cv::Mat`, `cv::Matx`, `cv::Vec` and `numpy.ndarray`
* Casts *without* shared memory for simple types, between `cv::Size`, `cv::Point`, `cv::Point3` and python `tuple`

#### With shared memory

* Casts *with* shared memory between `cvnp::Mat_shared`, `cvnp::Matx_shared`, `cvnp::Vec_shared` and `numpy.ndarray`

When you want to cast with shared memory, use these wrappers, which can easily be constructed from their OpenCV counterparts.
They are defined in [cvnp/cvnp_shared_mat.h](cvnp/cvnp_shared_mat.h).

Be sure that your matrixes lifetime if sufficient (_do not ever share the memory of a temporary matrix!_) 

### Supported matrix types

Since OpenCV supports a subset of numpy types, here is the table of supported types:

````
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
````


## How to use it in your project

1. Add cvnp to your project. For example:

````bash
cd external
git submodule add https://github.com/pthom/cvnp.git
````

2. Link it to your python module:

In your python module CMakeLists, add:

````cmake
add_subdirectory(path/to/cvnp)
target_link_libraries(your_target PRIVATE cvnp)
````

3. (Optional) If you want to import the declared functions in your module:

Write this in your main module code:
````cpp
void pydef_cvnp(pybind11::module& m);

PYBIND11_MODULE(your_module, m)
{
    ....
    ....
    ....
    pydef_cvnp(m);
}
````

You will get two simple functions:
* cvnp.list_types_synonyms()
* cvnp.print_types_synonyms()

````python
>>> import cvnp
>>> import pprint
>>> pprint.pprint(cvnp.list_types_synonyms(), indent=2, width=120)
[ {'cv_depth': 0, 'cv_depth_name': 'CV_8U', 'np_format': 'B', 'np_format_long': 'np.uint8'},
  {'cv_depth': 1, 'cv_depth_name': 'CV_8S', 'np_format': 'b', 'np_format_long': 'np.int8'},
  {'cv_depth': 2, 'cv_depth_name': 'CV_16U', 'np_format': 'H', 'np_format_long': 'np.uint16'},
  {'cv_depth': 3, 'cv_depth_name': 'CV_16S', 'np_format': 'h', 'np_format_long': 'np.int16'},
  {'cv_depth': 4, 'cv_depth_name': 'CV_32S', 'np_format': 'i', 'np_format_long': 'np.int32'},
  {'cv_depth': 5, 'cv_depth_name': 'CV_32F', 'np_format': 'f', 'np_format_long': 'float'},
  {'cv_depth': 6, 'cv_depth_name': 'CV_64F', 'np_format': 'd', 'np_format_long': 'np.float64'}]
````


### Shared and non shared matrices - Demo

Demo based on extracts from the tests:

We are using this struct:

````cpp
// CvNp_TestHelper is a test helper struct
struct CvNp_TestHelper
{
    // m is a *shared* matrix (i.e `cvnp::Mat_shared`)
    cvnp::Mat_shared m = cvnp::Mat_shared(cv::Mat::eye(cv::Size(4, 3), CV_8UC1));
    void SetM(int row, int col, uchar v) { m.Value.at<uchar>(row, col) = v; }

    // m_ns is a standard OpenCV matrix
    cv::Mat m_ns = cv::Mat::eye(cv::Size(4, 3), CV_8UC1);
    void SetM_ns(int row, int col, uchar v) { m_ns.at<uchar>(row, col) = v; }

    // ...
};
````

#### Shared matrices 

Changes propagate from Python to C++ and from C++ to Python

````python
def test_mat_shared():
    # CvNp_TestHelper is a test helper object
    o = CvNp_TestHelper()
    # o.m is a *shared* matrix i.e `cvnp::Mat_shared` in the object
    assert o.m.shape == (3, 4)

    # From python, change value in the C++ Mat (o.m) and assert that the changes are visible from python and C++
    o.m[0, 0] = 2
    assert o.m[0, 0] == 2

    # Make a python linked copy of the C++ Mat, named m_linked.
    # Values of m_mlinked and the C++ mat should change together
    m_linked = o.m
    m_linked[1, 1] = 3
    assert o.m[1, 1] == 3

    # Ask C++ to change a value in the matrix, at (0,0)
    # and verify that m_linked as well as o.m are impacted
    o.SetM(0, 0, 10)
    o.SetM(2, 3, 15)
    assert m_linked[0, 0] == 10
    assert m_linked[2, 3] == 15
    assert o.m[0, 0] == 10
    assert o.m[2, 3] == 15
````

#### Non shared matrices 

Changes propagate from C++ to Python, but not the other way.

````python
def test_mat_not_shared():
    # CvNp_TestHelper is a test helper object
    o = CvNp_TestHelper()
    # o.m_ns is a bare `cv::Mat`. Its memory is *not* shared
    assert o.m_ns.shape == (3, 4)

    # From python, change value in the C++ Mat (o.m) and assert that the changes are *not* applied
    o.m_ns[0, 0] = 2
    assert o.m_ns[0, 0] != 2 # No shared memory!

    # Ask C++ to change a value in the matrix, at (0,0) and verify that the change is visible from python
    o.SetM_ns(2, 3, 15)
    assert o.m_ns[2, 3] == 15
````

### Non continuous matrices

#### From C++
The conversion of non continuous matrices from C++ to python will fail. You need to clone them to make them continuous beforehand.

Example:

````cpp
    cv::Mat m(cv::Size(10, 10), CV_8UC1);
    cv::Mat sub_matrix = m(cv::Rect(3, 0, 3, m.cols));

    TEST_NAME("Try to convert a non continuous Mat to py::array, ensure it throws");
    TEST_ASSERT_THROW(
        cvnp::mat_to_nparray(sub_matrix, share_memory)
    );

    TEST_NAME("Clone the mat, ensure the clone can now be converted to py::array");
    cv::Mat sub_matrix_clone = sub_matrix.clone();
    py::array a = cvnp::mat_to_nparray(sub_matrix_clone, share_memory);
    TEST_ASSERT(a.shape()[0] == 10);
````

#### From python

The conversion of non continuous matrices from python to python will work, with or without shared memory.

````python
# import test utilities
>>> from cvnp import CvNp_TestHelper, cvnp_roundtrip, cvnp_roundtrip_shared, short_lived_matx, short_lived_mat
>>> o=CvNp_TestHelper()
# o.m is of type `cvnp::Mat_shared`
>>> o.m
array([[1, 0, 0, 0],
       [0, 1, 0, 0],
       [0, 0, 1, 0]], dtype=uint8)

# Create a non continuous array
>>> m = np.zeros((10,10))
>>> sub_matrix = m[4:6, :]
>>> sub_matrix.flags['F_CONTIGUOUS']
False

# Assign it to a `cvnp::Mat_shared`
>>> o.m = m
# Check that memory sharing works
>>> m[0,0]=42
>>> o.m[0,0]
42.0
````


## Build and test

_These steps are only for development and testing of this package, they are not required in order to use it in a different project._

### Build

````bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

mkdir build
cd build

# if you do not have a global install of OpenCV and pybind11
conan install .. --build=missing
# if you do have a global install of OpenCV, but not pybind11
conan install ../conanfile_pybind_only.txt --build=missing

cmake ..
make
````

### Test

In the build dir, run:

````
cmake --build . --target test
````

### Deep clean

````
rm -rf build
rm -rf venv
rm -rf .pytest_cache
rm  *.so 
rm *.pyd
````


## Notes

Thanks to Dan Mašek who gave me some inspiration here:
https://stackoverflow.com/questions/60949451/how-to-send-a-cvmat-to-python-over-shared-memory

This code is intended to be integrated into your own pip package. As such, no pip tooling is provided.
