project(mybridge)
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# AVX does not do much benefit at the moment, but breaks compilation on some platforms.
# Disabled for now until AVX becomes important enough to reconsider.
SET(EMBREE_MAX_ISA "SSE4.2" CACHE STRING "Selects highest ISA to support.")
set(USE_AVX FALSE CACHE BOOL "Use AVX.")
set(USE_AVX2 FALSE CACHE BOOL "Use AVX2.")

include(OptimizeForArchitecture)
OptimizeForArchitecture()

if (USE_AVX)
    message(STATUS "Compiling with AVX support")
    set(__AVX__ 1)
elseif (USE_SSE4_2)
    message(STATUS "Compiling with SSE4.2 support")
elseif (USE_SSSE3)
    message(STATUS "Compiling with SSE3 support")
else()
    message(FATAL_ERROR "The target machine does not support SSE3. At least SSE3 is required")
endif()

foreach(flag ${Vc_ARCHITECTURE_FLAGS})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag}")
endforeach()
message("Optimized Flags for this machine: ${CMAKE_CXX_FLAGS}")

FIND_PACKAGE(OpenMP)
if (OPENMP_FOUND)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
endif (OPENMP_FOUND)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")


add_library(my_tvl1flow_lib 3rdparty/tvl1flow/tvl1flow_lib)

set(src
    libBridge.cpp

    #TV-L1 Optical Flow
    3rdparty/tvl1flow/bicubic_interpolation.c
    3rdparty/tvl1flow/mask.c 
    3rdparty/tvl1flow/zoom.c
    3rdparty/tvl1flow/xmalloc.c
    3rdparty/tvl1flow/tvl1flow_lib.c
)

include_directories(. ./3rdparty/tvl1flow)
add_library(Bridge MODULE ${src})
TARGET_LINK_LIBRARIES(Bridge my_tvl1flow_lib)

message("***************** C FLAGS *****************")
message("${CMAKE_C_FLAGS}")
message("***************** CXX FLAGS *****************")
message("${CMAKE_CXX_FLAGS}")
