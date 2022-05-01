#include "cvnp/cvnp.h"


void test_mat_shared()
{
//    {
//        cvnp::Mat_shared m = cv::Mat::eye(cv::Size(4, 3), CV_8UC1);
//        auto array = cvnp::mat_to_nparray(m, true);
//    }
    {
        cv::Mat m = cv::Mat::eye(cv::Size(4, 3), CV_8UC1);
        auto array = cvnp::mat_to_nparray(m, false);
    }
}


int main()
{
    test_mat_shared();
}