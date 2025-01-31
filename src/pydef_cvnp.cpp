#include "cvnp/cvnp_synonyms.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

py::object list_types_synonyms_as_dict()
{
    py::list r;
    for (const auto& format: cvnp::sTypeSynonyms)
    {
        py::dict d;
        d["cv_depth"] = format.cv_depth;
        d["cv_depth_name"] = format.cv_depth_name;
        d["np_format"] = format.np_format;
        d["np_format_long"] = format.np_format_long;
        r.append(d);
    }
    return r;
}

void pydef_cvnp(pybind11::module& m)
{
    using namespace cvnp;
    m.def("print_types_synonyms", cvnp::print_types_synonyms);
    m.def("list_types_synonyms", list_types_synonyms_as_dict);
}
