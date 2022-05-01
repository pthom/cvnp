#pragma once

#include <opencv2/core/core.hpp>


namespace cvnp
{
    //
    // "Distinct Synonyms" for cv::Matx, cv::Mat and cv::Vec when you explicitly intend to share memory
    //
    template<typename Tp, int m, int n>
    class Matx_shared : public cv::Matx<Tp, m, n>
    {
        using Matxxx = cv::Matx<Tp, m, n>;
        using Matxxx_shared = cvnp::Matx_shared<Tp, m, n>;
    public:
        Matx_shared()
        {

        }

        Matx_shared &operator=(Matxxx &&other) noexcept
        {
            Matxxx::operator=(std::forward<Matxxx>(other));
            return *this;
        }

//        Matx_shared& operator=(cv::MatExpr&& matexpr) noexcept
//        {
//            Matxxx other = std::forward<cv::MatExpr>(matexpr);
//            Matxxx::operator=(other);
//            return *this;
//        }
        Matx_shared(Matxxx &&other)

        noexcept
        {
            static_cast<Matxxx>(*this) = std::forward<Matxxx>(other);
        }

        Matx_shared(cv::MatExpr &&matexpr)

        noexcept
        {
            static_cast<Matxxx>(*this) = std::forward<cv::MatExpr>(matexpr);
        }
    };


#define LOG_DBG(s) printf("    %s\n", s);

    class Mat_shared : public cv::Mat
    {
    public:
        Mat_shared() : cv::Mat()
        {
            LOG_DBG("Mat_shared() : cv::Mat()");
        }
        Mat_shared(int rows, int cols, int type): cv::Mat(rows, cols, type)
        {
            LOG_DBG("Mat_shared(int rows, int cols, int type): cv::Mat(rows, cols, type)");
        }
        Mat_shared(int rows, int cols, int type, const cv::Scalar& s) : cv::Mat(rows, cols, type, s)
        {
            LOG_DBG("Mat_shared(int rows, int cols, int type, const cv::Scalar& s) : cv::Mat(rows, cols, type, s)");
        }
        Mat_shared(cv::Size size, int type): cv::Mat(size, type)
        {
            LOG_DBG("Mat_shared(cv::Size size, int type): cv::Mat(size, type)");
        }
        Mat_shared(cv::Size size, int type, const cv::Scalar& s) : cv::Mat(size, type, s)
        {
            LOG_DBG("Mat_shared(cv::Size size, int type, const cv::Scalar& s) : cv::Mat(size, type, s)");
        }

        Mat_shared(const cv::Mat& other) noexcept
        {
            LOG_DBG("Mat_shared(const cv::Mat& other) noexcept");
            static_cast<cv::Mat>(*this) = other;
        }

        Mat_shared(cv::Mat&& other) noexcept
        {
            LOG_DBG("Mat_shared(cv::Mat&& other) noexcept");
            static_cast<cv::Mat>(*this) = std::forward<cv::Mat>(other);
        }

        Mat_shared(const cv::MatExpr &e) noexcept
        {
            LOG_DBG("Mat_shared(const cv::MatExpr &e) noexcept");
            static_cast<cv::Mat>(*this) = e;
        }

        Mat_shared& operator=(cv::Mat&& other) noexcept
        {
            LOG_DBG("Mat_shared& operator=(cv::Mat&& other) noexcept");
            cv::Mat::operator=(std::forward<cv::Mat>(other));
            return *this;
        }

        Mat_shared& operator=(const cv::MatExpr& e) noexcept
        {
            LOG_DBG("Mat_shared& operator=(const cv::MatExpr& e) noexcept");
            e.op->assign(e, *this);
            return *this;
        }

    };


    template<typename Tp, int m>
    class Vec_shared : public cv::Vec<Tp, m>
    {
        using Vecxx = cv::Vec<Tp, m>;
        using Vecxx_shared = cvnp::Vec_shared<Tp, m>;
    public:
        Vec_shared()
        {}

        Vec_shared &operator=(Vecxx &&other)

        noexcept
        {
            Vecxx::operator=(std::forward<Vecxx>(other));
            return *this;
        }

//        Vec_shared& operator=(cv::MatExpr&& matexpr) noexcept
//        {
//            Vecxx other = std::forward<cv::MatExpr>(matexpr);
//            Vecxx::operator=(other);
//            return *this;
//        }
        Vec_shared(Vecxx &&other)

        noexcept
        {
            static_cast<Vecxx>(*this) = std::forward<Vecxx>(other);
        }

        Vec_shared(cv::MatExpr &&matexpr)

        noexcept
        {
            static_cast<Vecxx>(*this) = std::forward<cv::MatExpr>(matexpr);
        }
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


    typedef Vec_shared<uchar, 2> Vec_shared2b;
    typedef Vec_shared<uchar, 3> Vec_shared3b;
    typedef Vec_shared<uchar, 4> Vec_shared4b;

    typedef Vec_shared<short, 2> Vec_shared2s;
    typedef Vec_shared<short, 3> Vec_shared3s;
    typedef Vec_shared<short, 4> Vec_shared4s;

    typedef Vec_shared<ushort, 2> Vec_shared2w;
    typedef Vec_shared<ushort, 3> Vec_shared3w;
    typedef Vec_shared<ushort, 4> Vec_shared4w;

    typedef Vec_shared<int, 2> Vec_shared2i;
    typedef Vec_shared<int, 3> Vec_shared3i;
    typedef Vec_shared<int, 4> Vec_shared4i;
    typedef Vec_shared<int, 6> Vec_shared6i;
    typedef Vec_shared<int, 8> Vec_shared8i;

    typedef Vec_shared<float, 2> Vec_shared2f;
    typedef Vec_shared<float, 3> Vec_shared3f;
    typedef Vec_shared<float, 4> Vec_shared4f;
    typedef Vec_shared<float, 6> Vec_shared6f;

    typedef Vec_shared<double, 2> Vec_shared2d;
    typedef Vec_shared<double, 3> Vec_shared3d;
    typedef Vec_shared<double, 4> Vec_shared4d;
    typedef Vec_shared<double, 6> Vec_shared6d;

} // namespace cvnp
