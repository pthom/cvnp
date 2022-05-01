#include "cvnp/cvnp.h"

#include <pybind11/embed.h>
#include <iostream>
#include <filesystem>

namespace py = pybind11;


#define TEST_CAT(s) printf("\n%s\n", s);


void test_mat_shared()
{
    {
        TEST_CAT("Create an empty Mat_shared, ensure it is ok");
        cvnp::Mat_shared m(cv::Size(10, 10), CV_8UC3);
        bool isContinuous = m.isContinuous();
        assert(isContinuous);
        assert(m.size() == cv::Size(10, 10));
    }

    {
        TEST_CAT("Create an empty Mat_shared from an lvalue Mat, ensure it is ok");
        cv::Mat mcv(cv::Size(10, 10), CV_8UC3);
        cvnp::Mat_shared m(mcv);
        bool isContinuous = m.isContinuous();
        assert(m.size() == cv::Size(10, 10));
        assert(isContinuous);
    }

    {
        TEST_CAT("Create an empty Mat_shared from an rvalue Mat, ensure it is ok");
        cvnp::Mat_shared m(cv::Mat(cv::Size(10, 10), CV_8UC3));
        bool isContinuous = m.isContinuous();
        assert(isContinuous);
        assert(m.size() == cv::Size(10, 10));
    }

    {
        TEST_CAT("Create an empty Mat_shared, copy a Mat inside, ensure it is ok");
        cv::Mat m(cv::Size(10, 10), CV_8UC3);
        cvnp::Mat_shared ms;
        ms = m;
        bool isContinuous = m.isContinuous();
        assert(isContinuous);
        assert(m.size() == cv::Size(10, 10));
    }


    {
        TEST_CAT("Create an empty Mat_shared, fill it with a cv::MatExpr, ensure it is ok");
        cvnp::Mat_shared m;
        m = cv::Mat::eye(cv::Size(4, 3), CV_8UC1);
        bool isContinuous= m.isContinuous();
        assert(m.size() == cv::Size(4, 3));
        assert(isContinuous);
    }

    {
        TEST_CAT("Create an Mat_shared from a cv::MatExpr, ensure it is ok");
        cvnp::Mat_shared m(cv::Mat::eye(cv::Size(4, 3), CV_8UC1));
        bool isContinuous= m.isContinuous();
        assert(m.size() == cv::Size(4, 3));
        assert(isContinuous);
    }


    /////////////////
    {
//        cv::Mat m = cv::Mat::eye(cv::Size(4, 3), CV_8UC1);
//        auto array = cvnp::mat_to_nparray(m, false);
    }
}


int main()
{

    // We need to instantiate an interpreter before the tests,
    // so that pybind11 is fully initialized
    py::scoped_interpreter guard{};
    std::string cmd = R"(
print("hello from python")
    )";
    py::exec(cmd);


    test_mat_shared();
}