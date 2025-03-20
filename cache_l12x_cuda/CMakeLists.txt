cmake_minimum_required(VERSION 3.17)
set(nom_projet "cache_L12x_cuda")

PROJECT (${nom_projet} LANGUAGES CXX CUDA )

find_package(CUDAToolkit REQUIRED)

file(GLOB SourceFile
    "*.h"
    "cache_L12x_cuda.cu")
ADD_EXECUTABLE (${nom_projet} ${SourceFile})
set_target_properties(${nom_projet} PROPERTIES CUDA_ARCHITECTURES "86")
# GTX 730 cuda 11.4 ubuntu cuda 11.8 windows
# set_target_properties(${nom_projet} PROPERTIES CUDA_ARCHITECTURES "86")
# GTX 1050 cuda 11.4 windows11
# set_target_properties(${nom_projet} PROPERTIES CUDA_ARCHITECTURES "61")
target_compile_features(${nom_projet} PRIVATE cxx_std_20)
target_link_libraries(${nom_projet} CUDA::cuda_driver)


