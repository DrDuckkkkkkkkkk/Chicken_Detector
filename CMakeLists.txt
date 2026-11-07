cmake_minimum_required(VERSION 3.20)

# vcpkg-Toolchain: zuerst VCPKG_ROOT, sonst ~/vcpkg (muss VOR project() gesetzt werden)
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    if(DEFINED ENV{VCPKG_ROOT})
        set(_vcpkg "$ENV{VCPKG_ROOT}")
    else()
        set(_vcpkg "$ENV{USERPROFILE}/vcpkg")
    endif()
    if(EXISTS "${_vcpkg}/scripts/buildsystems/vcpkg.cmake")
        set(CMAKE_TOOLCHAIN_FILE "${_vcpkg}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
    endif()
endif()

project(ChickenDetector LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Ohne Angabe immer optimiert bauen (betrifft Single-Config-Generatoren wie Ninja/Make)
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui videoio dnn)

add_executable(ChickenDetector src/main.cpp src/Detector.h)
target_link_libraries(ChickenDetector PRIVATE ${OpenCV_LIBS})

find_package(Threads REQUIRED)
target_link_libraries(ChickenDetector PRIVATE Threads::Threads)

if(MSVC)
    # /W4 Warnungen, /permissive- strenger Standard, /utf-8 Quelltext als UTF-8
    target_compile_options(ChickenDetector PRIVATE /W4 /permissive- /utf-8 $<$<CONFIG:Release>:/O2 /GL>)
    target_link_options(ChickenDetector PRIVATE $<$<CONFIG:Release>:/LTCG>)
else()
    target_compile_options(ChickenDetector PRIVATE -Wall -Wextra -Wpedantic $<$<CONFIG:Release>:-O3>)
endif()
