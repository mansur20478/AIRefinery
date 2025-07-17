cmake_minimum_required(VERSION 3.10)
project(airefinery)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Ofast -DNDEBUG -pthread -march=native -fno-finite-math-only -DGGML_USE_LLAMAFILE")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -DNDEBUG -pthread -march=native -fno-finite-math-only -DGGML_USE_LLAMAFILE")

set(GGML_NATIVE ON CACHE BOOL "GGML NATIVE" FORCE)
set(GGML_LTO ON CACHE BOOL "Link-Time Optimizations" FORCE)
set(GGML_OPENBLAS ON CACHE BOOL "BLAS" FORCE)

set( GGML_MACHINE_SUPPORTS_dotprod_EXITCODE 
     "0"
     CACHE STRING "Result from TRY_RUN" FORCE)

set( GGML_MACHINE_SUPPORTS_dotprod_EXITCODE__TRYRUN_OUTPUT 
     "PLEASE_FILL_OUT-NOTFOUND"
     CACHE STRING "Output from TRY_RUN" FORCE)

add_subdirectory("llama.cpp")

set(SOURCES
        causal_lm.cpp
        llama_jni.cpp
)

add_library(airefinery SHARED ${SOURCES})

find_library(log-lib log)

target_link_libraries(airefinery PRIVATE ${log-lib} llama)

target_include_directories(airefinery PRIVATE
        llama
)
