import pytest
import numpy as np
import math
import random
import pytest

import sys

sys.path.append(".")
sys.path.append("..")


from cvnp import CvNp_TestHelper, cvnp_roundtrip, cvnp_roundtrip_shared, short_lived_matx, short_lived_mat


def are_float_close(x: float, y: float):
    return math.fabs(x - y) < 1e-5


"""
We are playing with this C++ class

struct CvNp_TestHelper
{
    //
    // Shared Matrixes
    //
    // Create a Mat_shared with 3 rows, 4 columns and 1 channel
    // its shape for numpy should be (3, 4)
    cvnp::Mat_shared m = cvnp::Mat_shared(cv::Mat::eye(cv::Size(4, 3), CV_8UC1));
    void SetM(int row, int col, uchar v) { m.Value.at<uchar>(row, col) = v; }

    cvnp::Matx_shared32d mx = cv::Matx32d::eye();
    void SetMX(int row, int col, double v) { mx.Value(row, col) = v;}

    cvnp::Vec_shared3d vx = cv::Vec3d(1., 2., 3.);

    //
    // *Not* shared Matrixes
    //
    cv::Mat m_ns = cv::Mat::eye(cv::Size(4, 3), CV_8UC1);
    void SetM_ns(int row, int col, uchar v) { m_ns.at<uchar>(row, col) = v; }

    cv::Matx32d mx_ns = cv::Matx32d::eye();
    void SetMX_ns(int row, int col, double v) { mx_ns(row, col) = v;}

    cv::Vec3d vx_ns = cv::Vec3d(1., 2., 3.);


    //
    // *Not* shared simple structs (Size, Point2 and Point3)
    //
    cv::Size s = cv::Size(123, 456);
    void SetWidth(int w) { s.width = w;}
    void SetHeight(int h) { s.height = h;}

    cv::Point2i pt = cv::Point2i(42, 43);
    void SetX(int x) { pt.x = x; }
    void SetY(int y) { pt.y = y; }

    cv::Point3d pt3 = cv::Point3d(41.5, 42., 42.5);
    void SetX3(double x) { pt3.x = x; }
    void SetY3(double y) { pt3.y = y; }
    void SetZ3(double z) { pt3.z = z; }
};
"""


def test_mat_shared():
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


def test_mat_not_shared():
    # CvNp_TestHelper is a test helper object
    o = CvNp_TestHelper()
    # o.m_ns is a bare `cv::Mat`. Its memory is *not* shared
    assert o.m_ns.shape == (3, 4)

    # play with its internal cv::Mat

    # From python, change value in the C++ Mat (o.m) and assert that the changes are *not* applied
    o.m_ns[0, 0] = 2
    assert o.m_ns[0, 0] != 2  # No shared memory!

    # Make a python linked copy of the C++ Mat, named m_linked.
    # Values of m_mlinked and the C++ mat should change together
    m_linked = o.m_ns
    m_linked[1, 1] = 3
    assert o.m_ns[1, 1] != 3  # No shared memory!

    # Ask C++ to change a value in the matrix, at (0,0)
    # and verify that m_linked as well as o.m are impacted
    o.SetM_ns(0, 0, 10)
    o.SetM_ns(2, 3, 15)
    assert m_linked[0, 0] != 10  # No shared memory!
    assert m_linked[2, 3] != 15
    assert o.m_ns[0, 0] == 10  # But we can modify by calling C++ methods
    assert o.m_ns[2, 3] == 15

    # Make a clone of the C++ mat and change a value in it
    # => Make sure that the C++ mat is not impacted
    m_clone = np.copy(o.m_ns)
    m_clone[1, 1] = 18
    assert o.m_ns[1, 1] != 18

    # Change the whole C++ mat, by assigning to it a new matrix of different type and dimension
    # check that the shape has changed, and that values are ok
    new_shape = (3, 4, 2)
    new_type = np.float32
    new_mat = np.zeros(new_shape, new_type)
    new_mat[0, 0, 0] = 42.1
    new_mat[1, 0, 1] = 43.1
    new_mat[0, 1, 1] = 44.1
    o.m_ns = new_mat
    assert o.m_ns.shape == new_shape
    assert o.m_ns.dtype == new_type
    assert are_float_close(o.m_ns[0, 0, 0], 42.1)
    assert are_float_close(o.m_ns[1, 0, 1], 43.1)
    assert are_float_close(o.m_ns[0, 1, 1], 44.1)


def test_matx_shared():
    # create object
    o = CvNp_TestHelper()
    assert o.mx.shape == (3, 2)

    # play with its internal cv::Mat

    # From python, change value in the C++ Mat (o.m) and assert that the changes are applied, and visible from python
    o.mx[0, 0] = 2
    assert o.mx[0, 0] == 2

    # Make a python linked copy of the C++ Mat, named m_linked.
    # Values of m_mlinked and the C++ mat should change together
    m_linked = o.mx
    m_linked[1, 1] = 3
    assert o.mx[1, 1] == 3

    # Ask C++ to change a value in the matrix, at (0,0)
    # and verify that m_linked as well as o.m are impacted
    o.SetMX(0, 0, 10)
    o.SetMX(2, 1, 15)
    assert are_float_close(m_linked[0, 0], 10)
    assert are_float_close(m_linked[2, 1], 15)
    assert are_float_close(o.mx[0, 0], 10)
    assert are_float_close(o.mx[2, 1], 15)

    # Make a clone of the C++ mat and change a value in it
    # => Make sure that the C++ mat is not impacted
    m_clone = np.copy(o.mx)
    m_clone[1, 1] = 18
    assert not are_float_close(o.mx[1, 1], 18)

    # Change the whole C++ matx, by assigning to it a new matrix with different values
    # check that values are ok
    new_shape = o.mx.shape
    new_type = o.mx.dtype
    new_mat = np.zeros(new_shape, new_type)
    new_mat[0, 0] = 42.1
    new_mat[1, 0] = 43.1
    new_mat[0, 1] = 44.1
    o.mx = new_mat
    assert o.mx.shape == new_shape
    assert o.mx.dtype == new_type
    assert are_float_close(o.mx[0, 0], 42.1)
    assert are_float_close(o.mx[1, 0], 43.1)
    assert are_float_close(o.mx[0, 1], 44.1)

    # Try to change the shape of the Matx (not allowed)
    new_mat = np.zeros([100, 100, 10], new_type)
    with pytest.raises(RuntimeError):
        o.mx = new_mat


def test_matx_not_shared():
    # create object
    o = CvNp_TestHelper()
    assert o.mx_ns.shape == (3, 2)

    # play with its internal cv::Mat

    # From python, change value in the C++ Mat (o.m) and assert that the changes are applied, and visible from python
    o.mx_ns[0, 0] = 2
    assert o.mx_ns[0, 0] != 2  # No shared memory!

    # Make a python linked copy of the C++ Mat, named m_linked.
    # Values of m_mlinked and the C++ mat should change together
    m_linked = o.mx_ns
    m_linked[1, 1] = 3
    assert o.mx_ns[1, 1] != 3  # No shared memory!

    # Ask C++ to change a value in the matrix, at (0,0)
    # and verify that m_linked as well as o.m are impacted
    o.SetMX_ns(0, 0, 10)
    o.SetMX_ns(2, 1, 15)
    assert not are_float_close(m_linked[0, 0], 10)  # No shared memory!
    assert not are_float_close(m_linked[2, 1], 15)
    assert are_float_close(o.mx_ns[0, 0], 10)  # But we can modify by calling C++ methods
    assert are_float_close(o.mx_ns[2, 1], 15)

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


def test_size():
    o = CvNp_TestHelper()
    assert o.s[0] == 123
    assert o.s[1] == 456
    o.SetWidth(789)
    assert o.s[0] == 789
    o.s = (987, 654)
    assert o.s[0] == 987
    assert o.s[1] == 654


def test_point():
    o = CvNp_TestHelper()
    assert o.pt[0] == 42
    assert o.pt[1] == 43
    o.SetX(789)
    assert o.pt[0] == 789
    o.pt = (987, 654)
    assert o.pt[0] == 987
    assert o.pt[1] == 654


def test_point3():
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


def test_cvnp_round_trip_shared():
    m = np.zeros([5, 6, 7])
    m[3, 4, 5] = 156
    m2 = cvnp_roundtrip_shared(m)
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

        m2 = cvnp_roundtrip_shared(m)

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


def main():
    # Todo: find a way to call pytest for this file
    test_mat_shared()
    test_matx_shared()
    test_cvnp_round_trip()


if __name__ == "__main__":
    main()
