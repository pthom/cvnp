#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "cvnp/cvnp.h"

void pydef_cvnp(pybind11::module& m);
void pydef_cvnp_shared_test(pybind11::module& m);


PYBIND11_MODULE(cvnp, m)
{
    m.doc() = "...blablabla";

    pydef_cvnp_shared_test(m);
    pydef_cvnp(m);
}
