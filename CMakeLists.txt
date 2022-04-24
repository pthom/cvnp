cmake_minimum_required(VERSION 3.4)
project(cvnp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)

if (${PROJECT_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
    # For conan, add binary dir to module search path
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_BINARY_DIR})
    find_package(pybind11 REQUIRED)
endif()

find_package(OpenCV REQUIRED)

#
# Library cvnp_lib that you can link into another python module
#
file(GLOB sources cvnp/*.h cvnp/*.cpp)
pybind11_add_module(cvnp_lib OBJECT ${sources})
target_link_libraries(cvnp_lib PUBLIC opencv_core)
target_include_directories(cvnp_lib PUBLIC ${CMAKE_CURRENT_LIST_DIR})

#
# Standalone module cvnp that you can import into python
# (Mainly used for the tests: for the integration with your code,
# it is easier to link with cvnp_lib object library)
#
pybind11_add_module(cvnp  main/cvnp_main.cpp)
target_link_libraries(cvnp PRIVATE cvnp_lib)
# Copy the .so or .pyd module into this dir, to facilitate the tests
add_custom_command(
    TARGET cvnp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:cvnp> ${CMAKE_CURRENT_LIST_DIR}
)
