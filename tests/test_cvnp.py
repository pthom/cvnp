import pytest
import numpy as np
import math
import random
import pytest

import sys

sys.path.append(".")
sys.path.append("..")


from cvnp import CvNp_TestHelper, cvnp_roundtrip, short_lived_matx, short_lived_mat  # type: ignore


def are_float_close(x: float, y: float):
    return math.fabs(x - y) < 1e-5


def test_mat_shared():
    """
    Test cv::Mat
    We are playing with these elements
        cv::Mat m = cv::Mat::eye(cv::Size(4, 3), CV_8UC1);
        void SetM(int row, int col, uchar v) { m.at<uchar>(row, col) = v; }
    """
    # CvNp_TestHelper is a test helper object
    o = CvNp_TestHelper()
    # o.m is a *shared* matrix i.e `cvnp::Mat_shared` in the object
    assert o.m.shape == (3, 4)

    # play with its internal cv::Mat

    # From python, change value in the C++ Mat (o.m) and assert that the changes are applied, and visible from python
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

    # Make a clone of the C++ mat and change a value in it
    # => Make sure that the C++ mat is not impacted
    m_clone = np.copy(o.m)
    m_clone[1, 1] = 18
    assert o.m[1, 1] != 18

    # Change the whole C++ mat, by assigning to it a new matrix of different type and dimension
    # check that the shape has changed, and that values are ok
    new_shape = (3, 4, 2)
    new_type = np.float32
    new_mat = np.zeros(new_shape, new_type)
    new_mat[0, 0, 0] = 42.1
    new_mat[1, 0, 1] = 43.1
    new_mat[0, 1, 1] = 44.1
    o.m = new_mat
    assert o.m.shape == new_shape
    assert o.m.dtype == new_type
    assert are_float_close(o.m[0, 0, 0], 42.1)
    assert are_float_close(o.m[1, 0, 1], 43.1)
    assert are_float_close(o.m[0, 1, 1], 44.1)


def test_mat__shared():
    """
    Test cv::Mat_<Tp>
    We are playing with these elements
        cv::Mat_<uint8_t> m_uint8 = cv::Mat_<uint8_t>::eye(cv::Size(4, 3));
        cv::Mat_<int8_t> m_int8 = cv::Mat_<int8_t>::eye(cv::Size(4, 3));
        cv::Mat_<uint16_t> m_uint16 = cv::Mat_<uint16_t>::eye(cv::Size(4, 3));
        cv::Mat_<int16_t> m_int16 = cv::Mat_<int16_t>::eye(cv::Size(4, 3));
        cv::Mat_<int32_t> m_int32 = cv::Mat_<int32_t>::eye(cv::Size(4, 3));
        cv::Mat_<float> m_float = cv::Mat_<float>::eye(cv::Size(4, 3));
        cv::Mat_<double> m_double = cv::Mat_<double>::eye(cv::Size(4, 3));
        void set_m_double(int row, int col, double v) { m_double(row, col) = v; }
    """
    # CvNp_TestHelper is a test helper object
    o = CvNp_TestHelper()

    # Test 1: shapes and types
    assert o.m_uint8.shape == (3, 4)
    assert o.m_uint8.dtype.name == "uint8"
    assert o.m_int8.shape == (3, 4)
    assert o.m_int8.dtype.name == "int8"

    assert o.m_uint16.shape == (3, 4)
    assert o.m_uint16.dtype.name == "uint16"
    assert o.m_int16.shape == (3, 4)
    assert o.m_int16.dtype.name == "int16"

    assert o.m_int32.shape == (3, 4)
    assert o.m_int32.dtype.name == "int32"

    assert o.m_float.shape == (3, 4)
    assert o.m_float.dtype.name == "float32"

    assert o.m_double.shape == (3, 4)
    assert o.m_double.dtype.name == "float64"

    # Test 2: make a linked python copy, modify it, and assert that changes are visible in C++
    m_uint8 = o.m_uint8 # m_uint8 is a python linked numpy matrix
    m_uint8[1, 2] = 3   # modify the python matrix
    assert(o.m_uint8[1, 2] == 3)  # assert that changes are propagated to C++

    m_int32 = o.m_int32 # m_uint8 is a python linked numpy matrix
    m_int32[1, 2] = 3   # modify the python matrix
    assert(o.m_int32[1, 2] == 3)  # assert that changes are propagated to C++

    m_double = o.m_double
    m_double[1, 2] = 3.5
    assert(o.m_double[1, 2] == 3.5)

    # Test 3: make changes from c++
    o.set_m_double(2, 1, 4.5)
    assert(m_double[2, 1] == 4.5)
    assert(o.m_double[2, 1] == 4.5)


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

    # Make a clone of the C++ mat and change a value in it
    # => Make sure that the C++ mat is not impacted
    m_clone = np.copy(o.mx_ns)
    m_clone[1, 1] = 18
    assert not are_float_close(o.mx_ns[1, 1], 18)

    # Change the whole C++ matx, by assigning to it a new matrix with different values
    # check that values are ok
    new_shape = o.mx_ns.shape
    new_type = o.mx_ns.dtype
    new_mat = np.zeros(new_shape, new_type)
    new_mat[0, 0] = 42.1
    new_mat[1, 0] = 43.1
    new_mat[0, 1] = 44.1
    o.mx_ns = new_mat
    assert o.mx_ns.shape == new_shape
    assert o.mx_ns.dtype == new_type
    assert are_float_close(o.mx_ns[0, 0], 42.1)
    assert are_float_close(o.mx_ns[1, 0], 43.1)
    assert are_float_close(o.mx_ns[0, 1], 44.1)

    # Try to change the shape of the Matx (not allowed)
    new_mat = np.zeros([100, 100, 10], new_type)
    with pytest.raises(RuntimeError):
        o.mx_ns = new_mat


def test_vec_not_shared():
    """
    We are playing with these elements
        cv::Vec3f v3_ns = {1.f, 2.f, 3.f};
        void SetV3_ns(int idx, float v) { v3_ns(idx) = v; }
    """
    o = CvNp_TestHelper()
    assert o.v3_ns.shape == (3, 1)
    assert o.v3_ns[0] == 1.

    o.v3_ns[0] = 10
    assert o.v3_ns[0] != 10. # Vec are not shared

    o.SetV3_ns(0, 10)
    assert o.v3_ns[0] == 10.


def test_size():
    """
    we are playing with these elements
        cv::Size s = cv::Size(123, 456);
        void SetWidth(int w) { s.width = w;}
        void SetHeight(int h) { s.height = h;}
    """
    o = CvNp_TestHelper()
    assert o.s[0] == 123
    assert o.s[1] == 456
    o.SetWidth(789)
    assert o.s[0] == 789
    o.s = (987, 654)
    assert o.s[0] == 987
    assert o.s[1] == 654


def test_point():
    """
    we are playing with these elements
        cv::Point2i pt = cv::Point2i(42, 43);
        void SetX(int x) { pt.x = x; }
        void SetY(int y) { pt.y = y; }
    """
    o = CvNp_TestHelper()
    assert o.pt[0] == 42
    assert o.pt[1] == 43
    o.SetX(789)
    assert o.pt[0] == 789
    o.pt = (987, 654)
    assert o.pt[0] == 987
    assert o.pt[1] == 654

    """
    we are playing with these elements
        cv::Point3d pt3 = cv::Point3d(41.5, 42., 42.5);
        void SetX3(double x) { pt3.x = x; }
        void SetY3(double y) { pt3.y = y; }
        void SetZ3(double z) { pt3.z = z; }
    """
    o = CvNp_TestHelper()
    assert are_float_close(o.pt3[0], 41.5)
    assert are_float_close(o.pt3[1], 42.0)
    assert are_float_close(o.pt3[2], 42.5)
    o.SetX3(789.0)
    assert are_float_close(o.pt3[0], 789.0)
    o.pt3 = (987.1, 654.2, 321.0)
    assert are_float_close(o.pt3[0], 987.1)
    assert are_float_close(o.pt3[1], 654.2)
    assert are_float_close(o.pt3[2], 321.0)


def test_cvnp_round_trip():
    m = np.zeros([5, 6, 7])
    m[3, 4, 5] = 156
    m2 = cvnp_roundtrip(m)
    assert (m == m2).all()

    possible_types = [np.uint8, np.int8, np.uint16, np.int16, np.int32, float, np.float64]
    for test_idx in range(300):
        ndim = random.choice([2, 3])
        shape = []
        for dim in range(ndim):
            if dim < 2:
                shape.append(random.randrange(2, 1000))
            else:
                shape.append(random.randrange(2, 10))
        type = random.choice(possible_types)

        m = np.zeros(shape, dtype=type)

        i = random.randrange(shape[0])
        j = random.randrange(shape[1])
        if ndim == 2:
            m[i, j] = random.random()
        elif ndim == 3:
            k = random.randrange(shape[2])
            m[i, j, k] = random.random()
        else:
            raise RuntimeError("Should not happen")

        m2 = cvnp_roundtrip(m)

        if not (m == m2).all():
            print("argh")
        assert (m == m2).all()


def test_short_lived_matx():
    """
    We are calling the function ShortLivedMatx():

        // Returns a short lived matrix: sharing memory for this matrix makes *no sense at all*,
        // since its pointer lives on the stack and is deleted as soon as we exit the function!
        cv::Matx33d ShortLivedMatx()
        {
            auto mat = cv::Matx33d::eye();
            return mat;
        }
    """
    m = short_lived_matx()
    assert are_float_close(m[0, 0], 1.0)


def test_short_lived_mat():
    """
    We are calling the function ShortLivedMat():

        // Returns a short lived Mat: sharing memory for this matrix makes *no sense at all*,
        // since its pointer lives on the stack and is deleted as soon as we exit the function!
        cv::Mat ShortLivedMat()
        {
            auto mat = cv::Mat(cv::Size(300, 200), CV_8UC4);
            mat = cv::Scalar(12, 34, 56, 78);
            return mat;
        }
    """
    m = short_lived_mat()
    assert m.shape == (200, 300, 4)
    assert (m[0, 0] == (12, 34, 56, 78)).all()


def test_empty_mat():
    m = np.zeros(shape=(0, 0, 3))
    m2 = cvnp_roundtrip(m)
    assert (m == m2).all()


def test_refcount():
    """
    We are playing with these bindings
        cv::Mat m10 = cv::Mat(cv::Size(100, 100), CV_32FC3, cv::Scalar(0.f, 0.f, 0.f));
        int m10_refcount()
        {
            if (m10.u)
                return m10.u->refcount;
            else
                { printf("m10.u is null!\n"); return 0; }
        }
    """
    o = CvNp_TestHelper()
    m = o.m10
    assert o.m10_refcount() == 2
    m2 = o.m10
    assert o.m10_refcount() == 3
    del(m)
    assert o.m10_refcount() == 2
    del(m2)
    assert o.m10_refcount() == 1


def test_sub_matrices():
    """
    We are playing with these bindings
        struct CvNp_TestHelper {
            cv::Mat m10 = cv::Mat(cv::Size(100, 100), CV_32FC3, cv::Scalar(0.f, 0.f, 0.f));
            void SetM10(int row, int col, cv::Vec3f v) { m10.at<cv::Vec3f>(row, col) = v; }
            cv::Vec3f GetM10(int row, int col) { return m10.at<cv::Vec3f>(row, col); }
            cv::Mat GetSubM10() { return m10(cv::Rect(1, 1, 3, 3)); }

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


def test_scalar():
    """
    We are playing with this:
        cv::Scalar scalar_double = cv::Scalar(1.);
        cv::Scalar_<float> scalar_float = cv::Scalar_<float>(1.f, 2.f);
        cv::Scalar_<int32_t> scalar_int32 = cv::Scalar_<int32_t>(1, 2, 3);
        cv::Scalar_<uint8_t> scalar_uint8 = cv::Scalar_<uint8_t>(1, 2, 3, 4);
    """
    o = CvNp_TestHelper()

    assert o.scalar_double == (1.0, 0.0, 0.0, 0.0)
    o.scalar_double = (4.0, 5.0)
    assert o.scalar_double == (4.0, 5.0, 0.0, 0.0)

    assert o.scalar_float == (1.0, 2.0, 0.0, 0.0)
    o.scalar_float = (4.0, 5.0, 6.0)
    assert o.scalar_float == (4.0, 5.0, 6.0, 0.0)

    assert o.scalar_int32 == (1, 2, 3, 0)
    o.scalar_int32 = (4, 5, 6, 7)
    assert o.scalar_int32 == (4, 5, 6, 7)

    assert o.scalar_uint8 == (1, 2, 3, 4)
    o.scalar_uint8 = (4, 5, 6, 7)
    assert o.scalar_uint8 == (4, 5, 6, 7)

    # Check that setting float values to an int scalar raises an error:
    with pytest.raises(RuntimeError):
        o.scalar_int32 = (1.23, 4.56)


def test_rect():
    """
    We are playing with:
        cv::Rect  rect_int = cv::Rect(1, 2, 3, 4);
        cv::Rect_<double> rect_double = cv::Rect_<double>(5., 6., 7., 8.);
    """
    o = CvNp_TestHelper()

    assert o.rect_int == (1, 2, 3, 4)
    o.rect_int = (50, 55, 60, 65)
    assert o.rect_int == (50, 55, 60, 65)
    with pytest.raises(ValueError):
        o.rect_int = (1, 2) # We should give 4 values!
    with pytest.raises(RuntimeError):
        o.rect_int = (1.1, 2.1, 3.1, 4.1) # We should int values!

    assert o.rect_double == (5.0, 6.0, 7.0, 8.0)
    o.rect_double = (50.1, 55.2, 60.3, 65.4)
    assert o.rect_double == (50.1, 55.2, 60.3, 65.4)
    with pytest.raises(ValueError):
        o.rect_double = (1, 2) # We should give 4 values!



def main():
    # Todo: find a way to call pytest for this file
    test_refcount()
    test_mat_shared()
    test_mat__shared()
    test_sub_matrices()

    test_vec_not_shared()
    test_matx_not_shared()
    test_point()

    test_cvnp_round_trip()
    test_short_lived_mat()
    test_short_lived_matx()
    test_empty_mat()
    test_scalar()
    test_rect()


if __name__ == "__main__":
    main()
