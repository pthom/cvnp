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


    template<typename Tp, int m, int n>
    struct Matx_shared
    {
        Matx_shared(const cv::Matx<Tp, m, n>& value = cv::Matx<Tp, m, n>()): Value(value) {};
        cv::Matx<Tp, m, n> Value;
    };


    typedef Matx_shared<float, 1, 2> Matx_shared12f;
    typedef Matx_shared<double, 1, 2> Matx_shared12d;
    typedef Matx_shared<float, 1, 3> Matx_shared13f;
    typedef Matx_shared<double, 1, 3> Matx_shared13d;
    typedef Matx_shared<float, 1, 4> Matx_shared14f;
    typedef Matx_shared<double, 1, 4> Matx_shared14d;
    typedef Matx_shared<float, 1, 6> Matx_shared16f;
    typedef Matx_shared<double, 1, 6> Matx_shared16d;

    typedef Matx_shared<float, 2, 1> Matx_shared21f;
    typedef Matx_shared<double, 2, 1> Matx_shared21d;
    typedef Matx_shared<float, 3, 1> Matx_shared31f;
    typedef Matx_shared<double, 3, 1> Matx_shared31d;
    typedef Matx_shared<float, 4, 1> Matx_shared41f;
    typedef Matx_shared<double, 4, 1> Matx_shared41d;
    typedef Matx_shared<float, 6, 1> Matx_shared61f;
    typedef Matx_shared<double, 6, 1> Matx_shared61d;

    typedef Matx_shared<float, 2, 2> Matx_shared22f;
    typedef Matx_shared<double, 2, 2> Matx_shared22d;
    typedef Matx_shared<float, 2, 3> Matx_shared23f;
    typedef Matx_shared<double, 2, 3> Matx_shared23d;
    typedef Matx_shared<float, 3, 2> Matx_shared32f;
    typedef Matx_shared<double, 3, 2> Matx_shared32d;

    typedef Matx_shared<float, 3, 3> Matx_shared33f;
    typedef Matx_shared<double, 3, 3> Matx_shared33d;

    typedef Matx_shared<float, 3, 4> Matx_shared34f;
    typedef Matx_shared<double, 3, 4> Matx_shared34d;
    typedef Matx_shared<float, 4, 3> Matx_shared43f;
    typedef Matx_shared<double, 4, 3> Matx_shared43d;

    typedef Matx_shared<float, 4, 4> Matx_shared44f;
    typedef Matx_shared<double, 4, 4> Matx_shared44d;
    typedef Matx_shared<float, 6, 6> Matx_shared66f;
    typedef Matx_shared<double, 6, 6> Matx_shared66d;
} // namespace cvnp
