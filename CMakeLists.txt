cmake_minimum_required(VERSION 3.4)
project(cvnp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)

if (${PROJECT_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
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
else()
    # in order to use cvnp from another project, do this   
    #       add_subdirectory(path/to/cvnp)
    #       target_sources(your_target PRIVATE ${cvnp_SOURCES})
    #       target_include_directories(your_target PRIVATE ${cvnp_INCLUDE_DIRS})
    file(GLOB sources cvnp/*.h cvnp/*.cpp)
    set(cvnp_SOURCES ${sources} PARENT_SCOPE)
    set(cvnp_INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR} PARENT_SCOPE)
endif()

