#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "cvnp/cvnp.h"

void pydef_cvnp(pybind11::module& m);
void pydef_cvnp_test(pybind11::module& m);


PYBIND11_MODULE(cvnp, m)
{
    pydef_cvnp_test(m);
    pydef_cvnp(m);
}
