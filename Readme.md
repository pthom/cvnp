cv_np: pybind11 casts and transformers between numpy and OpenCV.


### Automatic casts:

1. Casts with shared memory between `cv::Mat`, `cv::Matx`, `cv::Vec` and `numpy.ndarray`
2. Casts without shared memory for simple types, between `cv::Size`, `cv::Point`, `cv::`Point3` and pythonic `tuple`


### Explicit transformers between cv::Mat / cv::Matx and numpy.ndarray, with shared memory

````cpp
    pybind11::array mat_to_nparray(const cv::Mat& m);
    cv::Mat         nparray_to_mat(pybind11::array& a);

        template<typename _Tp, int _rows, int _cols>
    pybind11::array matx_to_nparray(const cv::Matx<_Tp, _rows, _cols>& m);
        template<typename _Tp, int _rows, int _cols>
    void            nparray_to_matx(pybind11::array &a, cv::Matx<_Tp, _rows, _cols>& out_matrix);
````

### Supported matrix types

Since OpenCV supports a subset of numpy types, here is the table of supported types:

````
python
>>> import cv_np
>>> print(cv_np.list_types_synonyms())
  cv_depth  cv_depth_name np_format  
     0         CV_8U         B      
     1         CV_8S         b      
     2         CV_16U        H      
     3         CV_16S        h      
     4         CV_32S        i      
     5         CV_32F        f      
     6         CV_64F        d      
````

### Notes

Thanks to Dan Mašek who gave me some inspiration here:
https://stackoverflow.com/questions/60949451/how-to-send-a-cvmat-to-python-over-shared-memory


### Build and test

#### Build
````bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

mkdir build
cd build
conan install .. --build=missing
cmake ..
make
````

#### Test

In the main repository dir:

````
pytest tests
````

