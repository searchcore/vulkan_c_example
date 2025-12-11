cmake_minimum_required(VERSION 4.0)

set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)

project(vulkan_c_example)

add_executable(vulkan_c_example WIN32 main.c)

target_compile_options(vulkan_c_example PRIVATE
    /Zi
    /Od
    /Ob0
)


find_package(SDL3 REQUIRED CONFIG REQUIRED COMPONENTS SDL3-shared)

# Create your game executable target as usual

target_include_directories(vulkan_c_example PRIVATE "$ENV{VK_SDK_PATH}/Include")

message(STATUS "$ENV{VK_SDK_PATH}")

# Link to the actual SDL3 library.
target_link_libraries(vulkan_c_example PRIVATE SDL3::SDL3)
target_link_libraries(vulkan_c_example PUBLIC "$ENV{VK_SDK_PATH}/Lib/vulkan-1.lib")