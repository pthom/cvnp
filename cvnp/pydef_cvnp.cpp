#include "cvnp/cvnp_synonyms.h"

#include <pybind11/pybind11.h>


void pydef_cvnp(pybind11::module& m)
{
    m.def("list_types_synonyms", cvnp::list_types_synonyms);
}
