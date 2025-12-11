1. Build requirements
1.1. CMake >= 4.0
1.2. Ninja as cmake generator
1.3. Clang (You can use different compiler but I didn't test it)
1.4. VulkanSDK (VK_SDK_PATH must be set to root of VulkanSDK and VulkanSDK/bin/ must be in path)

2. Build for Windows
2.1 Create directory "build"
2.2 From project root directory generate build files: "cmake -G "Ninja" -S . -B ./build/"
2.3 From project root directory build project: "cmake --build ./build/"
2.4 Go to ./build/ and run vulkan_c_example.exe
