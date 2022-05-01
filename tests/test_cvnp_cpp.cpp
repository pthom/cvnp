#include "cvnp/cvnp.h"

#include <pybind11/embed.h>
#include <iostream>
#include <sstream>

namespace py = pybind11;


// Poor man's unit test macros, that do not require to add an external dependency
#define TEST_SHOUT_NAME(s) printf("\n%s\n", s);
#define TEST_ASSERT(v)                                                 \
{                                                                      \
    if ( ! (v) )                                                       \
    {                                                                  \
        std::stringstream ss;                                          \
        ss << "TEST_ASSERT failed at " << __FILE__ << ":" << __LINE__; \
        throw std::runtime_error(ss.str().c_str());                    \
    }                                                                  \
}


void test_mat_shared()
{
    {
        TEST_SHOUT_NAME("Create an empty Mat_shared, ensure it is ok");
        cvnp::Mat_shared m;
        TEST_ASSERT(m.Value.size() == cv::Size(0, 0));
    }

    {
        TEST_SHOUT_NAME("Create an empty Mat_shared from an lvalue Mat, ensure it is ok");
        cv::Mat mcv(cv::Size(10, 10), CV_8UC3);
        cvnp::Mat_shared m(mcv);
        bool isContinuous = m.Value.isContinuous();
        TEST_ASSERT(m.Value.size() == cv::Size(10, 10));
        TEST_ASSERT(isContinuous);
    }

    {
        TEST_SHOUT_NAME("Create an empty Mat_shared from an rvalue Mat, ensure it is ok");
        cvnp::Mat_shared m(cv::Mat(cv::Size(10, 10), CV_8UC3));
        bool isContinuous = m.Value.isContinuous();
        TEST_ASSERT(isContinuous);
        TEST_ASSERT(m.Value.size() == cv::Size(10, 10));
    }

    {
        TEST_SHOUT_NAME("Create an empty Mat_shared, copy a Mat inside, ensure it is ok");
        cv::Mat mcv(cv::Size(10, 10), CV_8UC3);
        cvnp::Mat_shared m;
        m = mcv;
        bool isContinuous = m.Value.isContinuous();
        TEST_ASSERT(isContinuous);
        TEST_ASSERT(m.Value.size() == cv::Size(10, 10));
    }


    {
        TEST_SHOUT_NAME("Create an empty Mat_shared, fill it with a cv::MatExpr, ensure it is ok");
        cvnp::Mat_shared m;
        m.Value = cv::Mat::eye(cv::Size(4, 3), CV_8UC1);
        bool isContinuous= m.Value.isContinuous();
        TEST_ASSERT(m.Value.size() == cv::Size(4, 3));
        TEST_ASSERT(isContinuous);
    }

    {
        TEST_SHOUT_NAME("Create an Mat_shared from a cv::MatExpr, ensure it is ok");
        cvnp::Mat_shared m(cv::Mat::eye(cv::Size(4, 3), CV_8UC1));
        bool isContinuous= m.Value.isContinuous();
        TEST_ASSERT(m.Value.size() == cv::Size(4, 3));
        TEST_ASSERT(isContinuous);
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