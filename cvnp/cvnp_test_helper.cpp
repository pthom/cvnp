#include "cvnp.h"

#include <pybind11/pybind11.h>
#include <opencv2/core.hpp>


// This simple function will call
// * the cast Python->C++ (for the input parameter)
// * the cast C++->Python (for the returned value)
// The unit tests check that the values and types are unmodified
cv::Mat cvnp_roundtrip(const cv::Mat& m)
{
    return m;
}

cvnp::Mat_shared cvnp_roundtrip_shared(const cvnp::Mat_shared& m)
{
    return m;
}



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

    //
    // *Not* shared Matrixes
    //
    cv::Mat m_ns = cv::Mat::eye(cv::Size(4, 3), CV_8UC1);
    void SetM_ns(int row, int col, uchar v) { m_ns.at<uchar>(row, col) = v; }

    cv::Matx32d mx_ns = cv::Matx32d::eye();
    void SetMX_ns(int row, int col, double v) { mx_ns(row, col) = v;}


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


// Returns a short lived Matx: sharing memory for this matrix makes *no sense at all*,
// since its pointer lives on the stack and is deleted as soon as we exit the function!
cv::Matx33d ShortLivedMatx()
{
    auto mat = cv::Matx33d::eye();
    return mat;
}


// Returns a short lived Mat: sharing memory for this matrix makes *no sense at all*,
// since its pointer lives on the stack and is deleted as soon as we exit the function!
cv::Mat ShortLivedMat()
{
    auto mat = cv::Mat(cv::Size(300, 200), CV_8UC4);
    mat = cv::Scalar(12, 34, 56, 78);
    return mat;
}


void pydef_cvnp_test(pybind11::module& m)
{
    pybind11::class_<CvNp_TestHelper>(m, "CvNp_TestHelper")
        .def(pybind11::init<>())

        .def_readwrite("m", &CvNp_TestHelper::m)
        .def("SetM", &CvNp_TestHelper::SetM)

        .def_readwrite("mx", &CvNp_TestHelper::mx)
        .def("SetMX", &CvNp_TestHelper::SetMX)

        .def_readwrite("m_ns", &CvNp_TestHelper::m_ns)
        .def("SetM_ns", &CvNp_TestHelper::SetM_ns)

        .def_readwrite("mx_ns", &CvNp_TestHelper::mx_ns)
        .def("SetMX_ns", &CvNp_TestHelper::SetMX_ns)

        .def_readwrite("s", &CvNp_TestHelper::s)
        .def("SetWidth", &CvNp_TestHelper::SetWidth)
        .def("SetHeight", &CvNp_TestHelper::SetHeight)

        .def_readwrite("pt", &CvNp_TestHelper::pt)
        .def("SetX", &CvNp_TestHelper::SetX)
        .def("SetY", &CvNp_TestHelper::SetY)

        .def_readwrite("pt3", &CvNp_TestHelper::pt3)
        .def("SetX3", &CvNp_TestHelper::SetX3)
        .def("SetY3", &CvNp_TestHelper::SetY3)
        .def("SetZ3", &CvNp_TestHelper::SetZ3)
        ;

    m.def("cvnp_roundtrip", cvnp_roundtrip);
    m.def("cvnp_roundtrip_shared", cvnp_roundtrip_shared);

    m.def("short_lived_matx", ShortLivedMatx);
    m.def("short_lived_mat", ShortLivedMat);
}


