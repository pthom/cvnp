#pragma once

#include <opencv2/core/core.hpp>


namespace cvnp
{
    //
    // Wrappers for cv::Matx, cv::Mat and cv::Vec when you explicitly intend to share memory
    //

    struct Mat_shared
    {
        Mat_shared(const cv::Mat value = cv::Mat()): Value(value) {};
        cv::Mat Value;
    };
} // namespace cvnp
