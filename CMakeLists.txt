cmake_minimum_required(VERSION 3.16)
project(custats_backend LANGUAGES CXX CUDA)

find_package(pybind11 REQUIRED)
find_package(CUDA REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES "75")
set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_CUDA_FLAGS "-O3")

set(SOURCE_DIR "custats/backend")
include_directories(${SOURCE_DIR})

set(SOURCES 
  "${SOURCE_DIR}/functions.cu"
  "${SOURCE_DIR}/kernels.cu"
)

if(${CUDA_ERROR_CHECK} MATCHES "True")
  add_compile_definitions(__CUDA_ERROR_CHECK__)
endif()

pybind11_add_module(${PROJECT_NAME} ${SOURCES} "${SOURCE_DIR}/bindings.cpp")
