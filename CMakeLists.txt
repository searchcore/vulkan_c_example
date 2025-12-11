cmake_minimum_required(VERSION 4.0)

set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)
set(SLANGC_EXECUTABLE slangc)
file(GLOB SHADER_SLANG_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/shaders/shader.slang)

project(vulkan_c_example)

add_executable(vulkan_c_example WIN32 main.c)

target_compile_options(vulkan_c_example PRIVATE)

function (add_slang_shader_target TARGET)
    cmake_parse_arguments ("SHADER" "" "" "SOURCES" ${ARGN})
    set (SHADERS_DIR ${CMAKE_CURRENT_LIST_DIR}/shaders)
    set (ENTRY_POINTS -entry vertMain -entry fragMain)
    add_custom_command (
        OUTPUT ${SHADERS_DIR}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADERS_DIR}
    )
    add_custom_command (
        OUTPUT  ${SHADERS_DIR}/slang.spv
        COMMAND ${SLANGC_EXECUTABLE} ${SHADER_SOURCES} -target spirv -profile spirv_1_4 -emit-spirv-directly -fvk-use-entrypoint-name ${ENTRY_POINTS} -o ${CMAKE_BINARY_DIR}/slang.spv
        WORKING_DIRECTORY ${SHADERS_DIR}
        DEPENDS ${SHADERS_DIR} ${SHADER_SOURCES}
        COMMENT "Compiling Slang Shaders"
        VERBATIM
    )
    add_custom_target (${TARGET} DEPENDS ${SHADERS_DIR}/slang.spv)
endfunction()

add_slang_shader_target(shader SOURCES ${SHADER_SLANG_SOURCES})
add_dependencies(vulkan_c_example shader)

find_package(SDL3 REQUIRED CONFIG REQUIRED COMPONENTS SDL3-shared)

# Create your game executable target as usual

target_include_directories(vulkan_c_example PRIVATE "$ENV{VK_SDK_PATH}/Include")

message(STATUS "$ENV{VK_SDK_PATH}")

# Link to the actual SDL3 library.
target_link_libraries(vulkan_c_example PRIVATE SDL3::SDL3)
target_link_libraries(vulkan_c_example PUBLIC "$ENV{VK_SDK_PATH}/Lib/vulkan-1.lib")