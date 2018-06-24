cmake_minimum_required(VERSION 3.1)

project(xt_util)


##############################
# Check and enable C++ 14
##############################

include(CheckCXXCompilerFlag)
# make sure the compiler supports c++14
# FIXME I think this won't work for MSVC
CHECK_CXX_COMPILER_FLAG("-std=c++14" HAS_CPP14_FLAG)
if(NOT HAS_CPP14_FLAG)
    message(FATAL_ERROR "Unsupported compiler -- xt_util requires C++14 support!")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")


# set default build type
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
        "MinSizeRel" "RelWithDebInfo")
endif()
string(TOUPPER "${CMAKE_BUILD_TYPE}" U_CMAKE_BUILD_TYPE)


# add gtest external project and include the directories
add_subdirectory(external/googletest/googletest)
include_directories(${gtest_SOURCE_DIR/include} ${gtest_SOURCE_DIR})


# Find the current conda env and set it as CMAKE_PREFIX_PATH
execute_process(
    COMMAND bash -c "conda info | grep 'active env location' | awk '{print $5}'"
    OUTPUT_VARIABLE CMAKE_PREFIX_PATH
)
string(REGEX REPLACE "\n$" "" CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}")

# Set CMAKE_PREFIX_PATH to the conda env, but allow changing it
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} CACHE PATH "")

MESSAGE(STATUS "Setting cmake prefix path to ${CMAKE_PREFIX_PATH}")
include_directories("${CMAKE_PREFIX_PATH}/include")


# find global headers
file(GLOB_RECURSE headers include/*.hxx)
file(GLOB_RECURSE headers ${CMAKE_INSTALL_PREFIX}/include/*.hxx)
include_directories(include)

# define test libraries
SET(TEST_LIBS
    gtest
    gtest_main
    pthread
)

# add subdirectories
add_subdirectory(test)
