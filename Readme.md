### cvnp: pybind11 casts and transformers between numpy and OpenCV with shared memory


#### Automatic casts:

1. Casts with shared memory between `cv::Mat`, `cv::Matx`, `cv::Vec` and `numpy.ndarray`
2. Casts without shared memory for simple types, between `cv::Size`, `cv::Point`, `cv::`Point3` and pythonic `tuple`


#### Explicit transformers between cv::Mat / cv::Matx and numpy.ndarray, with shared memory

````cpp
    pybind11::array mat_to_nparray(const cv::Mat& m);
    cv::Mat         nparray_to_mat(pybind11::array& a);

        template<typename _Tp, int _rows, int _cols>
    pybind11::array matx_to_nparray(const cv::Matx<_Tp, _rows, _cols>& m);
        template<typename _Tp, int _rows, int _cols>
    void            nparray_to_matx(pybind11::array &a, cv::Matx<_Tp, _rows, _cols>& out_matrix);
````

#### Supported matrix types

Since OpenCV supports a subset of numpy types, here is the table of supported types:

````
python
>>> import cvnp
>>> print(cvnp.list_types_synonyms())
  cv_depth  cv_depth_name np_format  
     0         CV_8U         B    (np.uint8) 
     1         CV_8S         b    (np.int8)
     2         CV_16U        H    (np.uint16)
     3         CV_16S        h    (np.int16)
     4         CV_32S        i    (np.int32)
     5         CV_32F        f    (float)
     6         CV_64F        d    (np.float64)
````

### Notes

Thanks to Dan Mašek who gave me some inspiration here:
https://stackoverflow.com/questions/60949451/how-to-send-a-cvmat-to-python-over-shared-memory


### Build and test

#### Build / standard mode
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

#### Build with pip

This code is intended to be integrated into your own pip package. 
As such, no pip tooling is provided.


#### Test

In the main repository dir:

````
pytest tests
````

#### Deep clean

````
rm -rf build
rm -rf venv
rm -rf .pytest_cache
rm  *.so 
rm *.pyd
````