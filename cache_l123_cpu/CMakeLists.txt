cmake_minimum_required(VERSION 3.10)
set(nom_projet "cache_L123_c++20")

PROJECT (${nom_projet})

file(GLOB SourceFile
    "*.h"
    "cache_L123_c++20.cpp")
ADD_EXECUTABLE (${nom_projet} ${SourceFile})
target_compile_features(${nom_projet} PRIVATE cxx_std_20)


