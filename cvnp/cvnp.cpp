#include "cvnp/cvnp.h"
#include <thread>

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

        int determine_cv_depth(const pybind11::dtype& dt)
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
            return { static_cast<int>(a.shape()[1]), static_cast<int>(a.shape()[0]) };
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

        std::vector<std::size_t> determine_strides(const cv::Mat& m) {
            if (m.channels() == 1) {
                return {
                    static_cast<size_t>(m.step[0]), // row stride (in bytes)
                    static_cast<size_t>(m.step[1])  // column stride (in bytes)
                };
            }
            return {
                static_cast<size_t>(m.step[0]), // row stride (in bytes)
                static_cast<size_t>(m.step[1]), // column stride (in bytes)
                static_cast<size_t>(m.elemSize1()) // channel stride (in bytes)
            };
        }
 
        py::capsule make_capsule_mat(const cv::Mat& m)
        {
            return py::capsule(new cv::Mat(m)
                , [](void *v) { delete reinterpret_cast<cv::Mat*>(v); }
            );
        }


    } // namespace detail

    pybind11::array mat_to_nparray(const cv::Mat& m)
    {
        return pybind11::array(detail::determine_np_dtype(m.depth())
            , detail::determine_shape(m)
            , detail::determine_strides(m)
            , m.data
            , detail::make_capsule_mat(m)
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


    namespace // anonymous namespace to hide MatAllocator_LinkArray from other translation units
    {
        // #define DEBUG_ALLOCATOR
        #ifdef DEBUG_ALLOCATOR
        #define LOG_ALLOCATOR(fmt, ...) printf(fmt, ##__VA_ARGS__)
        #else
        #define LOG_ALLOCATOR(fmt, ...)
        #endif
        int nbInstances = 0;

        //
        // MatAllocator_LinkArray is a MatAllocator that uses a numpy array as the data pointer
        // --------------------------------------------------------------------------------------------
        // The implementation is quite tricky:
        // - A collection of all instances is kept in a static vector (see register_instance and unregister_instance)
        // - The constructor:
        //     * keeps a reference to the numpy array
        //     * registers itself via register_instance()
        // - allocate():
        //     * creates a new UMatData object
        //     * "steals" the data pointer from the numpy array
        // - deallocate():
        //     * decrements the reference count and deletes the UMatData object if no more references
        //     * unregisters itself via unregister_instance() which will *destroy* this instance!
        //
        //          **The destructor is called via deallocate()**
        //
        // As a consequence, MatAllocator_LinkArray is created via a *naked* new() and will be destroyed via deallocate()
        //
        class MatAllocator_LinkArray: public cv::MatAllocator
        {
        public:
            explicit MatAllocator_LinkArray(pybind11::array& a) : m_linked_array(a)
            {
                register_instance(this);

                ++nbInstances;
                LOG_ALLOCATOR("MatAllocator_LinkArray constructor %p / nbInstances=%d\n", this, nbInstances);
            }

            ~MatAllocator_LinkArray() override
            {
                --nbInstances;
                LOG_ALLOCATOR("MatAllocator_LinkArray destructor %p / nbInstances=%d\n", this, nbInstances);
            }

            cv::UMatData* allocate(
                int dims, const int* sizes, int type,
                void* data, size_t* step, cv::AccessFlag flags, cv::UMatUsageFlags usageFlags) const override
            {
                // This is the allocation that will be called by cv::Mat::create
                LOG_ALLOCATOR("Allocate 1\n");

                // Create a new UMatData object
                cv::UMatData *u = new cv::UMatData(this);
                // "Steal" the data pointer from the numpy array
                u->data = (uchar*)m_linked_array.mutable_data(0);

                // Set the reference counts to 0
                // (since UMatData is not documented, this is based on a reverse engineering of the OpenCV code)
                // (deallocate won't be called if refcount or urefcount is set to 1 here)
                u->refcount = u->urefcount = 0; // What is the difference between refcount and urefcount? this is undocumented

                return u;
            }

            bool allocate(cv::UMatData* data, cv::AccessFlag accessflags, cv::UMatUsageFlags usageFlags) const override
            {
                // We never reach here (I guess)
                LOG_ALLOCATOR("Allocate 2\n");
                data->urefcount++;
                return true;
            }

            void deallocate(cv::UMatData* data) const override
            {
                LOG_ALLOCATOR("Deallocate\n");
                // Decrement the reference count
                data->urefcount--;

                // If no more references, delete the UMatData object
                if (data->urefcount <= 0)
                {
                    delete data;
                    unregister_instance(const_cast<MatAllocator_LinkArray*>(this));
                }
            }


        private:
            mutable pybind11::array m_linked_array;

        private:
            static std::vector<MatAllocator_LinkArray*> m_all_instances;
            static std::mutex m_all_instances_mutex;


            static void register_instance(MatAllocator_LinkArray* instance)
            {
                LOG_ALLOCATOR("register_instance %p\n", instance);
                std::lock_guard<std::mutex> lock(m_all_instances_mutex);
                m_all_instances.push_back(instance);
            }
            static void unregister_instance(MatAllocator_LinkArray* instance)
            {
                LOG_ALLOCATOR("unregister_instance %p\n", instance);
                std::lock_guard<std::mutex> lock(m_all_instances_mutex);
                delete instance;
                auto it = std::find(m_all_instances.begin(), m_all_instances.end(), instance);
                if (it != m_all_instances.end())
                    m_all_instances.erase(it);
                else
                    throw std::runtime_error("MatAllocator_LinkArray::unregister_instance / instance not found");
            }
        };
        std::vector<MatAllocator_LinkArray*> MatAllocator_LinkArray::m_all_instances; // C++ at its best syntactic terseness
        std::mutex MatAllocator_LinkArray::m_all_instances_mutex;                     // with static members
    } // anonymous namespace


    cv::Mat nparray_to_mat(pybind11::array& a)
    {
        // note: empty arrays are not contiguous, but that's fine. Just
        //       make sure to not access mutable_data
        bool is_contiguous = is_array_contiguous(a);
        bool is_empty = (a.size() == 0);
        if (! is_contiguous && !is_empty) {
            throw std::invalid_argument("cvnp::nparray_to_mat / Only contiguous numpy arrays are supported. / Please use np.ascontiguousarray() to convert your matrix");
        }

        int depth = detail::determine_cv_depth(a.dtype());
        int type = detail::determine_cv_type(a, depth);
        cv::Size size = detail::determine_cv_size(a);

        if (is_empty)
        {
            return cv::Mat(size, type, nullptr);
        }
        else
        {
            // Black magic here, we are creating a new MatAllocator_LinkArray
            // which will
            // - keep a reference to the numpy array
            // - create a cv::UMatData object that uses the numpy array data pointer
            // - auto-destruct when the cv::Mat is destroyed
            auto allocator = new MatAllocator_LinkArray(a);
            cv::Mat m;
            m.allocator = allocator;
            m.create(size, type);
            return m;
        }

    }

    // this version tries to handle strides and sub-matrices
    // this is WIP, currently broken, and not used
    cv::Mat nparray_to_mat_with_strides_broken(pybind11::array& a)
    {
        int depth = detail::determine_cv_depth(a.dtype());
        int type = detail::determine_cv_type(a, depth);
        cv::Size size = detail::determine_cv_size(a);

        auto buffer_info = a.request();

        // Get the array strides (convert from pybind11::ssize_t to size_t)
        std::vector<size_t> strides;
        for (auto v : buffer_info.strides)
            strides.push_back(static_cast<size_t>(v));

        // Get the number of dimensions
        int ndims = static_cast<int>(buffer_info.ndim);
        //if ((ndims != 2) && (ndims != 3))
        //    throw std::invalid_argument("nparray_to_mat needs support only 2 or 3 dimension matrices");

        // Convert the shape (sizes) to a vector of int
        std::vector<int> sizes;
        for (auto v : buffer_info.shape)
            sizes.push_back(static_cast<int>(v));

        // Create the cv::Mat with the specified strides (steps)
        // We are calling this Mat constructor:
        //     Mat(const std::vector<int>& sizes, int type, void* data, const size_t* steps=0)
        cv::Mat m(sizes, type, a.mutable_data(0), strides.data());
        return m;
    }

} // namespace cvnp
