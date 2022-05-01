cmake_minimum_required(VERSION 3.4)
project(cvnp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)

if (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
    # For conan, add binary dir to module search path
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_BINARY_DIR})
    find_package(pybind11 REQUIRED)

    find_package(OpenCV REQUIRED)

    file(GLOB sources cvnp/*.h cvnp/*.cpp)
    pybind11_add_module(cvnp ${sources} main/cvnp_main.cpp)
    target_link_libraries(cvnp PUBLIC opencv_core)
    target_include_directories(cvnp PUBLIC ${CMAKE_CURRENT_LIST_DIR})
    # Copy the .so or .pyd module into this dir, to facilitate the tests
    add_custom_command(
        TARGET cvnp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:cvnp> ${CMAKE_CURRENT_LIST_DIR}
    )

    #
    # c++ test target
    #
    add_executable(test_cvnp_cpp tests/test_cvnp_cpp.cpp ${sources})
    target_include_directories(test_cvnp_cpp PRIVATE ${CMAKE_CURRENT_LIST_DIR})
    target_link_libraries(test_cvnp_cpp PRIVATE opencv_core)
    find_package(Python3 REQUIRED COMPONENTS Development)
    target_link_libraries(test_cvnp_cpp PRIVATE pybind11::pybind11)
    enable_testing()
    add_test(NAME test_cvnp_cpp COMMAND test_cvnp_cpp)
    add_test(NAME test_cvnp_python COMMAND pytest ${CMAKE_SOURCE_DIR}/tests)

else()
    # in order to use cvnp from another project, do this:   
    #       add_subdirectory(path/to/cvnp)
    #       target_link_libraries(your_target PRIVATE cvnp)
    file(GLOB sources cvnp/*.h cvnp/*.cpp)
    add_library(cvnp INTERFACE)
    target_sources(cvnp INTERFACE ${sources})
    target_include_directories(cvnp INTERFACE ${CMAKE_CURRENT_LIST_DIR})
endif()
