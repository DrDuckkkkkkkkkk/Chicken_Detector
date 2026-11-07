cmake_minimum_required(VERSION 3.20)

# Use vcpkg from VCPKG_ROOT or %USERPROFILE%\vcpkg unless a toolchain file is given.
# This has to happen before project().
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    if(DEFINED ENV{VCPKG_ROOT})
        set(_vcpkg_root "$ENV{VCPKG_ROOT}")
    else()
        set(_vcpkg_root "$ENV{USERPROFILE}/vcpkg")
    endif()
    if(EXISTS "${_vcpkg_root}/scripts/buildsystems/vcpkg.cmake")
        set(CMAKE_TOOLCHAIN_FILE "${_vcpkg_root}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
    endif()
endif()

project(ChickenDetector VERSION 1.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui videoio dnn)
find_package(Threads REQUIRED)

add_executable(ChickenDetector src/main.cpp src/Detector.h)
target_link_libraries(ChickenDetector PRIVATE ${OpenCV_LIBS} Threads::Threads)

if(MSVC)
    target_compile_options(ChickenDetector PRIVATE /W4 /permissive- /utf-8 $<$<CONFIG:Release>:/O2 /GL>)
    target_link_options(ChickenDetector PRIVATE $<$<CONFIG:Release>:/LTCG>)
else()
    target_compile_options(ChickenDetector PRIVATE -Wall -Wextra -Wpedantic $<$<CONFIG:Release>:-O3>)
endif()
