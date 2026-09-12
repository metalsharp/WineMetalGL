cmake_minimum_required(VERSION 3.24)

project(WineMetalGL VERSION 0.3.0 LANGUAGES C CXX OBJC OBJCXX)

if(NOT APPLE)
    message(FATAL_ERROR "WineMetalGL builds only on macOS.")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_OSX_DEPLOYMENT_TARGET "15.0" CACHE STRING "Minimum macOS deployment target" FORCE)
set(CMAKE_OSX_ARCHITECTURES "x86_64" CACHE STRING "Host architecture" FORCE)
if(NOT CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64")
    message(FATAL_ERROR "WineMetalGL only supports x86_64 Mach-O output; universal and ARM targets are not supported.")
endif()

set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
set(SPIRV_CROSS_CLI OFF CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_GLSL ON CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_MSL ON CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_HLSL OFF CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_CPP OFF CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_REFLECT OFF CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_C_API ON CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_UTIL OFF CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(SPIRV_CROSS_SKIP_INSTALL ON CACHE BOOL "" FORCE)

set(ENABLE_GLSLANG_BINARIES OFF CACHE BOOL "" FORCE)
set(ENABLE_SPVREMAPPER OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(ENABLE_OPT OFF CACHE BOOL "" FORCE)
set(ENABLE_HLSL OFF CACHE BOOL "" FORCE)
set(GLSLANG_TESTS OFF CACHE BOOL "" FORCE)
set(GLSLANG_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)

add_subdirectory(vendor/SPIRV-Cross)
add_subdirectory(vendor/glslang)

find_library(APPKIT_FRAMEWORK AppKit REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(OPENGL_FRAMEWORK OpenGL REQUIRED)
find_library(QUARTZCORE_FRAMEWORK QuartzCore REQUIRED)

add_library(winemetalgl SHARED
    src/opengl/EntryPoint.cpp
    src/opengl/GLErrorTracker.cpp
    src/opengl/GLMetalRenderer.mm
    src/opengl/GLShaderCache.cpp
    src/opengl/GLShaderTracker.cpp
    src/opengl/GLSLCompiler.cpp
    src/opengl/OpenGLBridge.mm
    src/opengl/WGLContextManager.mm
    src/opengl/WGLShim.cpp
    src/runtime/Logger.cpp
)

target_include_directories(winemetalgl
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_compile_options(winemetalgl PRIVATE
    $<$<COMPILE_LANGUAGE:OBJC>:-fobjc-arc>
    $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>
)

target_link_libraries(winemetalgl
    PRIVATE
        glslang
        glslang-default-resource-limits
        SPIRV
        spirv-cross-core
        spirv-cross-glsl
        spirv-cross-msl
        "${APPKIT_FRAMEWORK}"
        "${FOUNDATION_FRAMEWORK}"
        "${METAL_FRAMEWORK}"
        "${OPENGL_FRAMEWORK}"
        "${QUARTZCORE_FRAMEWORK}"
)

set_target_properties(winemetalgl PROPERTIES
    OUTPUT_NAME "metalsharp-opengl"
    PREFIX ""
    INSTALL_NAME_DIR "@rpath"
    OSX_ARCHITECTURES "x86_64"
)

install(TARGETS winemetalgl LIBRARY DESTINATION lib)
