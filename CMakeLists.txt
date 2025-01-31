cmake_minimum_required(VERSION 3.26)
project(cvnp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)

option(CVNP_BUILD_TESTS "Build tests" ON)
option(CVNP_ENABLE_ASAN "Enable Address Sanitizer" OFF)

#
# check if pybind11 is installed via pip
#
find_package(pybind11)

if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not installed. Downloading it now.")
    include(FetchContent)

    FetchContent_Declare(pybind11 URL https://github.com/pybind/pybind11/archive/refs/heads/master.zip)
    FetchContent_MakeAvailable(pybind11)
endif()

#
# Requirements: python3, opencv
#
find_package(Python3 REQUIRED COMPONENTS Development Interpreter)
find_package(OpenCV REQUIRED)

#
# Define the main library
#
file(GLOB sources src/*.cpp)
file(GLOB headers include/cvnp/*.h)
add_library(cvnp ${sources})
set_target_properties(cvnp PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(cvnp PUBLIC pybind11::pybind11 ${OpenCV_LIBS} Python3::Python)
target_include_directories(
    cvnp
    PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)
set_target_properties(cvnp PROPERTIES PUBLIC_HEADER "${headers}")
install(TARGETS cvnp
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include/cvnp
)

if(CVNP_BUILD_TESTS)
    find_program(Python_EXECUTABLE python3)
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import numpy"
    )
    if(NOT Python_EXECUTABLE)
        message(FATAL_ERROR "
        Please set the Python_EXECUTABLE variable to a python interpreter
        where you installed numpy and opencv. For example:

            cmake .. -DPython_EXECUTABLE=/venv/bin/python
    ")
    endif()

    # Find site-packages directory (needed for tests)
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import site; print(site.getsitepackages()[0])"
        OUTPUT_VARIABLE PYTHON_SITE_PACKAGES_DIRECTORY
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if (CVNP_ENABLE_ASAN)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
        add_compile_definitions(CVNP_ENABLE_ASAN)
    endif()

    #
    # Build cvnp python module
    #
    # Copy the .so or .pyd module into this dir, to facilitate the tests
    add_custom_command(
        TARGET cvnp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:cvnp> ${CMAKE_CURRENT_LIST_DIR}
    )
    file(GLOB pybind_sources main/*.cpp)
    pybind11_add_module(py_cvnp ${pybind_sources})
    target_link_libraries(py_cvnp PRIVATE cvnp)

    #
    # c++ test target
    #
    add_executable(test_cvnp_cpp tests/test_cvnp_cpp.cpp)
    target_link_libraries(test_cvnp_cpp PUBLIC pybind11::pybind11 cvnp)
    enable_testing()
    add_test(NAME test_cvnp_cpp COMMAND test_cvnp_cpp ${PYTHON_SITE_PACKAGES_DIRECTORY} )

    #
    # python test target
    #
    message("lib path: " ${CMAKE_BINARY_DIR})
    add_test(
        NAME test_cvnp_python
        COMMAND pytest ${CMAKE_SOURCE_DIR}/tests)
    # just for testing we can set the PYTHONPATH
    set_property(TEST test_cvnp_python PROPERTY ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}")
endif()
