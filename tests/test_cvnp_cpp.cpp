#include "cvnp/cvnp.h"

#include <pybind11/embed.h>
#include <iostream>
#include <sstream>

namespace py = pybind11;

#include <algorithm>
#include <cctype>

std::string trim(const std::string& s)
{
    auto start = std::find_if_not(s.begin(), s.end(), ::isspace);
    auto end = std::find_if_not(s.rbegin(), s.rend(), ::isspace).base();

    return start < end ? std::string(start, end) : std::string();
}

// Poor man's unit test macros, that do not require to add an external dependency
#define TEST_NAME(s) printf("\n%s\n", s);
#define TEST_ASSERT(v)                                                 \
{                                                                      \
    if ( ! (v) )                                                       \
    {                                                                  \
        std::stringstream ss;                                          \
        ss << "TEST_ASSERT failed at " << __FILE__ << ":" << __LINE__; \
        throw std::runtime_error(ss.str().c_str());                    \
    }                                                                  \
}

#define TEST_ASSERT_THROW(expression)                                  \
{                                                                      \
    bool received_exception = false;                                   \
    try                                                                \
    {                                                                  \
        expression;                                                    \
    }                                                                  \
    catch(const std::exception& e)                                     \
    {                                                                  \
        received_exception = true;                                     \
    }                                                                  \
    if ( ! (received_exception) )                                      \
    {                                                                  \
        std::stringstream ss;                                          \
        ss << "TEST_ASSERT failed at " << __FILE__ << ":" << __LINE__; \
        throw std::runtime_error(ss.str().c_str());                    \
    }                                                                  \
}


void display_mat(const std::string& title, const cv::Mat& m)
{
    std::cout << title << "\n" << cv::format(m, cv::Formatter::FMT_PYTHON) << "\n";
}

bool is_matrix_content_equal_to(const cv::Mat& m, const std::string& expectedFormattedContent)
{
    std::stringstream ss;
    ss << cv::format(m, cv::Formatter::FMT_PYTHON);
    std::string m_str = trim(ss.str());
    bool is_equal = trim(expectedFormattedContent) == ss.str();
    return is_equal;
}


void test_submatrix()
{
    // We create two cv::Mat: m and sub_m (sub_m is a submatrix of m)
    // Then we create two pybind11::array:
    //     a is linked to m
    //    sub_a is a submatrix of a, expressed with python slices
    //
    //  => all modification made to a and sub_a should be transferred transparently to m and sub_m

    cv::Mat m(cv::Size(4, 3), CV_8UC1);
    m = cv::Scalar(0);
    m.at<uchar>(2, 1) = 255;

    cv::Mat sub_m = m(cv::Rect(1, 1, 3, 2));
    sub_m.at<uchar>(0, 0) = 10;

    pybind11::array a = cvnp::mat_to_nparray(m);

    // Now, let write the equivalent of
    //      sub_a = a[1:, 1:] # in python
    // but with C++
    pybind11::array sub_a;
    {
        // Create the py::slice objects for the rows and columns
        py::slice row_slice(py::int_(1), py::none(), py::none());
        py::slice col_slice(py::int_(1), py::none(), py::none());
        //pybind11::array sub_a = a[row_slice][col_slice];
        // Create the subarray b using the attr() function
        sub_a = a.attr("__getitem__")(py::make_tuple(row_slice, col_slice));
    }

    // Set the value at location (0, 0) in a to 12
    a.attr("__setitem__")(py::make_tuple(0, 0), 12);
    // Set the value at location (1, 2) in sub_a to 3
    sub_a.attr("__setitem__")(py::make_tuple(1, 2), 3);

    // py::print("a:\n", a);
    // py::print("sub_a:\n", sub_a);
    // display_mat("m", m);
    // display_mat("sub_m", sub_m);

    TEST_ASSERT(is_matrix_content_equal_to(m, R"(
[[ 12,   0,   0,   0],
 [  0,  10,   0,   0],
 [  0, 255,   0,   3]]
)"));

    TEST_ASSERT(is_matrix_content_equal_to(sub_m, R"(
[[ 10,   0,   0],
 [255,   0,   3]]
)"));
}



void test_non_continuous_mat()
{
    cv::Mat m(cv::Size(10, 10), CV_8UC1);
    cv::Mat sub_matrix = m(cv::Rect(3, 0, 4, 5));

    py::array a = cvnp::mat_to_nparray(sub_matrix);
    TEST_ASSERT(a.shape()[0] == 5);
    TEST_ASSERT(a.shape()[1] == 4);
}


void test_nparray_to_mat()
{
    std::vector<pybind11::ssize_t> a_shape{3, 4, 2};
    std::vector<pybind11::ssize_t> a_strides{};
    pybind11::dtype a_dtype = pybind11::dtype(pybind11::format_descriptor<int32_t>::format());
    pybind11::array a(
        a_dtype,
        a_shape,
        a_strides
        );

    cv::Mat m = cvnp::nparray_to_mat(a);

    pybind11::array aa = cvnp::mat_to_nparray(m);
    auto aa_shape = aa.shape();
    cv::Mat mm = cvnp::nparray_to_mat(aa);
    printf("Aghh");
}


int main()
{
    // If numpy is not installed globally, set site_packages_path
    std::string site_packages_path;
    {
        // In this example, we are using a virtual env
        std::string venv_path = "../venv";
        site_packages_path = venv_path + "/lib/python3.11/site-packages";
    }

    // We need to instantiate an interpreter before the tests,
    // and give it the path to the site_packages folder, so that pybind11 can access numpy
    // (which will be required)
    py::scoped_interpreter guard{};

    // Append the site-packages directory to sys.path
    py::module sys = py::module::import("sys");
    sys.attr("path").attr("append")(site_packages_path);

    test_nparray_to_mat();
    test_submatrix();
    test_non_continuous_mat();
}