cmake_minimum_required(VERSION 3.20)
project(qfin_native LANGUAGES CXX)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(
    _qfin_native
    MODULE
    bindings/python.cpp
    src/alm.cpp
    src/curves.cpp
    src/fixed_income.cpp
    src/mortality.cpp
    src/projections.cpp
    src/risk.cpp
    src/scenarios.cpp
)

target_compile_features(_qfin_native PRIVATE cxx_std_20)
set_target_properties(
    _qfin_native
    PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO
)
target_include_directories(_qfin_native PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(
    _qfin_native
    PRIVATE QFIN_CXX_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
)

if(MSVC)
    target_compile_options(_qfin_native PRIVATE /O2 /W4 /permissive-)
else()
    target_compile_options(_qfin_native PRIVATE -O3 -Wall -Wextra -Wpedantic)
endif()

install(TARGETS _qfin_native LIBRARY DESTINATION qfin RUNTIME DESTINATION qfin)
