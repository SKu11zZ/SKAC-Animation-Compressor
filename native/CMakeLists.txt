cmake_minimum_required(VERSION 3.20)
project(skac_runtime_beta VERSION 0.2.0 LANGUAGES CXX)

string(CONCAT SKAC_CXX_17_FEATURE "cxx_st" "d_17")

option(SKAC_RUNTIME_WITH_ZLIB "Enable direct .skac container inflation with zlib" OFF)
option(SKAC_RUNTIME_BUILD_TESTS "Build native runtime smoke tests" ON)
option(SKAC_RUNTIME_BUILD_SHARED "Build skac_runtime as a shared library" ON)

if(SKAC_RUNTIME_BUILD_SHARED)
    set(SKAC_RUNTIME_LIBRARY_TYPE SHARED)
else()
    set(SKAC_RUNTIME_LIBRARY_TYPE STATIC)
endif()

add_library(skac_runtime ${SKAC_RUNTIME_LIBRARY_TYPE} src/skac_runtime.cpp)
add_library(SKAC::Runtime ALIAS skac_runtime)
target_compile_features(skac_runtime PUBLIC ${SKAC_CXX_17_FEATURE})
target_compile_definitions(skac_runtime PRIVATE SKAC_RUNTIME_BUILD=1)
if(NOT SKAC_RUNTIME_BUILD_SHARED)
    target_compile_definitions(skac_runtime PUBLIC SKAC_RUNTIME_STATIC=1)
endif()
target_include_directories(
    skac_runtime
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

if(MSVC)
    target_compile_options(skac_runtime PRIVATE /W4 /permissive-)
else()
    target_compile_options(skac_runtime PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(SKAC_RUNTIME_WITH_ZLIB)
    find_package(ZLIB REQUIRED)
    target_compile_definitions(skac_runtime PRIVATE SKAC_RUNTIME_WITH_ZLIB=1)
    target_link_libraries(skac_runtime PRIVATE ZLIB::ZLIB)
endif()

if(SKAC_RUNTIME_BUILD_TESTS)
    enable_testing()
    add_executable(skac_runtime_smoke tests/runtime_smoke.cpp)
    target_compile_features(skac_runtime_smoke PRIVATE ${SKAC_CXX_17_FEATURE})
    target_link_libraries(skac_runtime_smoke PRIVATE SKAC::Runtime)
    add_test(NAME skac_runtime_smoke COMMAND skac_runtime_smoke)

    add_executable(skac_runtime_probe tests/runtime_probe.cpp)
    target_compile_features(skac_runtime_probe PRIVATE ${SKAC_CXX_17_FEATURE})
    target_link_libraries(skac_runtime_probe PRIVATE SKAC::Runtime)

    add_executable(skac_runtime_v2_probe tests/runtime_v2_probe.cpp)
    target_compile_features(skac_runtime_v2_probe PRIVATE ${SKAC_CXX_17_FEATURE})
    target_link_libraries(skac_runtime_v2_probe PRIVATE SKAC::Runtime)

    add_executable(skac_runtime_benchmark tests/runtime_benchmark.cpp)
    target_compile_features(skac_runtime_benchmark PRIVATE ${SKAC_CXX_17_FEATURE})
    target_link_libraries(skac_runtime_benchmark PRIVATE SKAC::Runtime)
endif()
