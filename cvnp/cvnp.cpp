#include "cvnp/cvnp.h"

// Thanks to Dan Mašek who gave me some inspiration here:
// https://stackoverflow.com/questions/60949451/how-to-send-a-cvmat-to-python-over-shared-memory

namespace cvnp
{

    namespace detail
    {
        namespace py = pybind11;
        
        py::dtype determine_np_dtype(int cv_depth)
        {
            for (auto format_synonym : cvnp::sTypeSynonyms)
                if (format_synonym.cv_depth == cv_depth)
                    return format_synonym.dtype();

            std::string msg = "numpy does not support this OpenCV depth: " + std::to_string(cv_depth) +  " (in determine_np_dtype)";
            throw std::invalid_argument(msg.c_str());
        }

        int determine_cv_depth(pybind11::dtype dt)
        {
            for (auto format_synonym : cvnp::sTypeSynonyms)
                if (format_synonym.np_format[0] == dt.char_())
                    return format_synonym.cv_depth;

            std::string msg = std::string("OpenCV does not support this numpy format: ") + dt.char_() +  " (in determine_np_dtype)";
            throw std::invalid_argument(msg.c_str());
        }

        int determine_cv_type(const py::array& a, int depth)
        {
            if (a.ndim() < 2)
                throw std::invalid_argument("determine_cv_type needs at least two dimensions");
            if (a.ndim() > 3)
                throw std::invalid_argument("determine_cv_type needs at most three dimensions");
            if (a.ndim() == 2)
                return CV_MAKETYPE(depth, 1);
            //We now know that shape.size() == 3
            return CV_MAKETYPE(depth, a.shape()[2]);
        }

        cv::Size determine_cv_size(const py::array& a)
        {
            if (a.ndim() < 2)
                throw std::invalid_argument("determine_cv_size needs at least two dimensions");
            return cv::Size(static_cast<int>(a.shape()[1]), static_cast<int>(a.shape()[0]));
        }

        std::vector<std::size_t> determine_shape(const cv::Mat& m)
        {
            if (m.channels() == 1) {
                return {
                    static_cast<size_t>(m.rows)
                    , static_cast<size_t>(m.cols)
                };
            }
            return {
                static_cast<size_t>(m.rows)
                , static_cast<size_t>(m.cols)
                , static_cast<size_t>(m.channels())
            };
        }

        py::capsule make_capsule_mat(const cv::Mat& m)
        {
            return py::capsule(new cv::Mat(m)
                , [](void *v) { delete reinterpret_cast<cv::Mat*>(v); }
            );
        }


    } // namespace detail

    pybind11::array mat_to_nparray(const cv::Mat& m, bool share_memory)
    {
        // note: empty mats aren't continuous
        if (!m.isContinuous() && !m.empty())
            throw std::invalid_argument("cvnp::mat_to_nparray / Only contiguous Mats supported / You can clone() your matrix to obtain a contiguous copy.");
        if (share_memory)
            return pybind11::array(detail::determine_np_dtype(m.depth())
                , detail::determine_shape(m)
                , m.data
                , detail::make_capsule_mat(m)
                );
        else
            return pybind11::array(detail::determine_np_dtype(m.depth())
                , detail::determine_shape(m)
                , m.data
                );
    }


    bool is_array_contiguous(const pybind11::array& a)
    {
        pybind11::ssize_t expected_stride = a.itemsize();
        for (int i = a.ndim() - 1; i >=0; --i)
        {
            pybind11::ssize_t current_stride = a.strides()[i];
            if (current_stride != expected_stride)
                return false;
            expected_stride = expected_stride * a.shape()[i];
        }
        return true;
    }


    cv::Mat nparray_to_mat(pybind11::array& a)
    {
        // note: empty arrays are not contiguous, but that's fine. Just
        //       make sure to not access mutable_data
        bool is_contiguous = is_array_contiguous(a);
        bool is_not_empty = a.size() != 0;
        if (! is_contiguous && is_not_empty) {
            throw std::invalid_argument("cvnp::nparray_to_mat / Only contiguous numpy arrays are supported. / Please use np.ascontiguousarray() to convert your matrix");
        }

        int depth = detail::determine_cv_depth(a.dtype());
        int type = detail::determine_cv_type(a, depth);
        cv::Size size = detail::determine_cv_size(a);
        cv::Mat m(size, type, is_not_empty ? a.mutable_data(0) : nullptr);
        return m;
    }

} // namespace cvnp
