cmake_minimum_required(VERSION 3.6)
project(cvnp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)

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


# if (PROJECT_IS_TOP_LEVEL)
#     # Add address sanitizer
#     option(CVNP_ENABLE_ASAN "Enable Address Sanitizer" OFF)
#     if (CVNP_ENABLE_ASAN)
#         set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
#         add_compile_definitions(CVNP_ENABLE_ASAN)
#     endif()

#     #
#     # Requirements: python3, opencv
#     #
#     find_package(Python3 REQUIRED COMPONENTS Development Interpreter)
#     find_package(OpenCV REQUIRED)

#     # Requirement: pybind11
#     # Add pybind11 via add_subdirectory (since vcpkg's pybind11 fails on linux + python 3.12)
#     if (NOT IS_DIRECTORY(${CMAKE_CURRENT_LIST_DIR}/pybind11))
#         message(STATUS "cloning pybind11")
#         execute_process(
#             COMMAND git clone https://github.com/pybind/pybind11.git
#             WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
#         )
#     endif()
#     add_subdirectory(pybind11)

#     #
#     # Build cvnp python module
#     #
#     file(GLOB sources src/*.cpp)
#     pybind11_add_module(cvnp ${sources} main/cvnp_main.cpp)
#     target_link_libraries(cvnp PUBLIC opencv_core)
#     target_include_directories(cvnp PUBLIC ${CMAKE_SOURCE_DIR}/include)
#     # Copy the .so or .pyd module into this dir, to facilitate the tests
#     add_custom_command(
#         TARGET cvnp POST_BUILD
#         COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:cvnp> ${CMAKE_CURRENT_LIST_DIR}
#     )

#     #
#     # c++ test target
#     #
#     add_executable(test_cvnp_cpp tests/test_cvnp_cpp.cpp ${sources})
#     target_include_directories(test_cvnp_cpp PRIVATE ${CMAKE_SOURCE_DIR}/include)
#     target_link_libraries(test_cvnp_cpp PRIVATE opencv_core)
#     target_link_libraries(test_cvnp_cpp PRIVATE pybind11::pybind11 Python3::Python)
#     enable_testing()
#     add_test(NAME test_cvnp_cpp COMMAND test_cvnp_cpp ${PYTHON_SITE_PACKAGES_DIRECTORY} )

#     #
#     # python test target
#     #
#     add_test(NAME test_cvnp_python COMMAND pytest ${CMAKE_SOURCE_DIR}/tests)

#     #
#     # setup installation
#     #
#     target_sources(
#         cvnp
#         PUBLIC
#             FILE_SET public-headers
#             TYPE HEADERS
#             FILES
#             cvnp_synonyms.h
#             cvnp.h
#     )

#     install(TARGETS cvnp FILE_SET public-headers)
# else()
    # in order to use cvnp from another project, do this:   
    #       add_subdirectory(path/to/cvnp)
    #       target_link_libraries(your_target PRIVATE cvnp)

    find_package(OpenCV REQUIRED)
    find_package(pybind11 REQUIRED)
    file(GLOB sources src/*.cpp)
    message(STATUS ${sources})
    add_library(cvnp ${sources})
    target_link_libraries(cvnp PUBLIC pybind11::pybind11 ${OpenCV_LIBS})
    target_include_directories(
        cvnp
        PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
        )
# endif()
