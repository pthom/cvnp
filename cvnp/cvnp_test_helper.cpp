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

struct CvNp_TestHelper
{
    //
    // cv::Mat (shared)
    //
    // Create a Mat with 3 rows, 4 columns and 1 channel. Its shape for numpy should be (3, 4)
    cv::Mat m = cv::Mat::eye(cv::Size(4, 3), CV_8UC1);
    void SetM(int row, int col, uchar v) { m.at<uchar>(row, col) = v; }

    // cv::Mat_<Tp> (shared)
    cv::Mat_<uint8_t> m_uint8 = cv::Mat_<uint8_t>::eye(cv::Size(4, 3));
    cv::Mat_<int8_t> m_int8 = cv::Mat_<int8_t>::eye(cv::Size(4, 3));
    cv::Mat_<uint16_t> m_uint16 = cv::Mat_<uint16_t>::eye(cv::Size(4, 3));
    cv::Mat_<int16_t> m_int16 = cv::Mat_<int16_t>::eye(cv::Size(4, 3));
    cv::Mat_<int32_t> m_int32 = cv::Mat_<int32_t>::eye(cv::Size(4, 3));
    cv::Mat_<float> m_float = cv::Mat_<float>::eye(cv::Size(4, 3));
    cv::Mat_<double> m_double = cv::Mat_<double>::eye(cv::Size(4, 3));
    void set_m_double(int row, int col, double v) { m_double(row, col) = v; }

    //
    // cv::Matx (not shared)
    //
    cv::Matx32d mx_ns = cv::Matx32d::eye();
    void SetMX_ns(int row, int col, double v) { mx_ns(row, col) = v;}

    //
    // cv::Vec not shared
    //
    cv::Vec3f v3_ns = {1.f, 2.f, 3.f};
    void SetV3_ns(int idx, float v) { v3_ns(idx) = v; }

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

    //
    // cv::Mat and sub matrices
    //
    cv::Mat m10 = cv::Mat(cv::Size(100, 100), CV_32FC3, cv::Scalar(0.f, 0.f, 0.f));
    void SetM10(int row, int col, cv::Vec3f v) { m10.at<cv::Vec3f>(row, col) = v; }
    cv::Vec3f GetM10(int row, int col) { return m10.at<cv::Vec3f>(row, col); }
    cv::Mat GetSubM10() {
        cv::Mat sub = m10(cv::Rect(1, 1, 3, 3));
        return sub;
    }
    int m10_refcount() 
    {
        if (m10.u)
            return m10.u->refcount;
        else
        {
            printf("m10.u is null!\n");
            return 0;
        }
    }

    //
    // cv::Scalar_
    //
    cv::Scalar scalar_double = cv::Scalar(1.);
    cv::Scalar_<float> scalar_float = cv::Scalar_<float>(1.f, 2.f);
    cv::Scalar_<int32_t> scalar_int32 = cv::Scalar_<int32_t>(1, 2, 3);
    cv::Scalar_<uint8_t> scalar_uint8 = cv::Scalar_<uint8_t>(1, 2, 3, 4);

    //
    // cv::Rect
    //
    cv::Rect  rect_int = cv::Rect(1, 2, 3, 4);
    cv::Rect_<double> rect_double = cv::Rect_<double>(5., 6., 7., 8.);
};


// Returns a short lived Matx: sharing memory for this matrix makes *no sense at all*,
// since its pointer lives on the stack and is deleted as soon as we exit the function!
cv::Matx33d ShortLivedMatx()
{
    auto mat = cv::Matx33d::eye();
    return mat;
}


// Returns a short lived Mat: sharing memory for this matrix *makes sense*
// since the capsule will add to its reference count
cv::Mat ShortLivedMat()
{
    auto mat = cv::Mat(cv::Size(300, 200), CV_8UC4);
    mat = cv::Scalar(12, 34, 56, 78);
    return mat;
}


cv::Matx33d make_eye()
{
    return cv::Matx33d::eye();
}


void display_eye(cv::Matx33d m = cv::Matx33d::eye())
{
    printf("display_eye\n");
    for(int row=0; row < 3; ++row)
        printf("%lf, %lf, %lf\n", m(row, 0), m(row, 1), m(row, 2));
}



void pydef_cvnp_test(pybind11::module& m)
{
    m.def("make_eye", make_eye);
    m.def("display_eye", display_eye);

    pybind11::class_<CvNp_TestHelper>(m, "CvNp_TestHelper")
        .def(pybind11::init<>())

        .def_readwrite("m", &CvNp_TestHelper::m)
        .def("SetM", &CvNp_TestHelper::SetM)

        .def_readwrite("m_uint8", &CvNp_TestHelper::m_uint8)
        .def_readwrite("m_int8", &CvNp_TestHelper::m_int8)
        .def_readwrite("m_uint16", &CvNp_TestHelper::m_uint16)
        .def_readwrite("m_int16", &CvNp_TestHelper::m_int16)
        .def_readwrite("m_int16", &CvNp_TestHelper::m_int16)
        .def_readwrite("m_int32", &CvNp_TestHelper::m_int32)
        .def_readwrite("m_float", &CvNp_TestHelper::m_float)
        .def_readwrite("m_double", &CvNp_TestHelper::m_double)
        .def("set_m_double", &CvNp_TestHelper::set_m_double)

        .def_readwrite("mx_ns", &CvNp_TestHelper::mx_ns)
        .def("SetMX_ns", &CvNp_TestHelper::SetMX_ns)

        .def_readwrite("v3_ns", &CvNp_TestHelper::v3_ns)
        .def("SetV3_ns", &CvNp_TestHelper::SetV3_ns)

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

        .def_readwrite("m10", &CvNp_TestHelper::m10)
        .def("SetM10", &CvNp_TestHelper::SetM10)
        .def("GetM10", &CvNp_TestHelper::GetM10)
        .def("GetSubM10", &CvNp_TestHelper::GetSubM10)
        .def("m10_refcount", &CvNp_TestHelper::m10_refcount)

        .def_readwrite("scalar_double", &CvNp_TestHelper::scalar_double)
        .def_readwrite("scalar_float", &CvNp_TestHelper::scalar_float)
        .def_readwrite("scalar_int32", &CvNp_TestHelper::scalar_int32)
        .def_readwrite("scalar_uint8", &CvNp_TestHelper::scalar_uint8)

        .def_readwrite("rect_int", &CvNp_TestHelper::rect_int)
        .def_readwrite("rect_double", &CvNp_TestHelper::rect_double)
        ;

    m.def("cvnp_roundtrip", cvnp_roundtrip);

    m.def("short_lived_matx", ShortLivedMatx);
    m.def("short_lived_mat", ShortLivedMat);
}


