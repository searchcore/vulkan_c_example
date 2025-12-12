#include <stdint.h>
#include <stdio.h>

#include <vulkan/vulkan.h>

#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <SDL3/SDL_vulkan.h>


typedef enum Constants {
    MAX_FRAMES_IN_FLIGHT = 2,
    MAX_IMAGES_COUNT = 8,
    LAYER_COUNT = 1,
} Constants;

static const char* pInstanceEnabledLayers[LAYER_COUNT] = {
    "VK_LAYER_KHRONOS_validation"
};

static const char* deviceExtensions[] = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_KHR_SPIRV_1_4_EXTENSION_NAME,
    VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
    VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME,
    VK_KHR_SHADER_DRAW_PARAMETERS_EXTENSION_NAME,
};
const uint32_t deviceExtensionsCount = sizeof(deviceExtensions) / sizeof(char*);

typedef struct GraphicsContext {
    SDL_Window* window;

    VkInstance instance;
    VkDevice device;
    VkPhysicalDevice physicalDevice;

    uint32_t queueCount;
    float* queuePriorities;

    VkQueue graphicsQueue;
    VkQueue presentQueue;
    VkSurfaceKHR surface;

    const char* const* deviceExtensions;
    uint32_t deviceExtensionsCount;

    const char* const* instanceExtensions;
    uint32_t instanceExtensionsCount;

    uint32_t presentQueueFamilyIdx;
    uint32_t graphicsQueueFamilyIdx;

    VkSwapchainKHR swapChain;
    uint32_t swapChainImagesCount;
    VkImage* swapChainImages;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;

    uint32_t imageViewCount;
    VkImageView* imageViews;

    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;
    VkCommandPool commandPool;

    uint32_t currentFrameIndex;
    VkCommandBuffer commandBuffer[MAX_FRAMES_IN_FLIGHT];

    VkSemaphore presentCompleteSemaphore[MAX_FRAMES_IN_FLIGHT];
    VkSemaphore renderFinishedSemaphore[MAX_IMAGES_COUNT];
    VkFence drawFence[MAX_FRAMES_IN_FLIGHT];

    VkDebugUtilsMessengerEXT debugMessanger;

    VkBool32 framebufferResized;
} GraphicsContext;


int initVulkan(GraphicsContext* ctx);
void getVkInstanceExtensions(const char*const** ext_out, uint32_t* extensionsCount);
int isDeviceSuitable(GraphicsContext* ctx, VkPhysicalDevice device);
void cleanup(GraphicsContext* ctx);
void drawFrame(GraphicsContext* ctx);

int main(int argc, char **argv) {
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        SDL_Log("Couldn't initialize SDL: %s", SDL_GetError());
        return -1;
    }

    SDL_Vulkan_LoadLibrary(NULL);

    const char* const* extensions = NULL;
    uint32_t extensionsCount = 0;
    getVkInstanceExtensions(&extensions, &extensionsCount);

    GraphicsContext ctx = {
        .framebufferResized = VK_FALSE,
        .currentFrameIndex = 0,
        .queueCount = 1,
        .renderFinishedSemaphore = { VK_NULL_HANDLE },
        .presentCompleteSemaphore = { VK_NULL_HANDLE },
        .drawFence = { VK_NULL_HANDLE },
        .instanceExtensions = extensions,
        .instanceExtensionsCount = extensionsCount,
        .deviceExtensions = deviceExtensions,
        .deviceExtensionsCount = deviceExtensionsCount,
    };

    float queuePriorities[1] = { 0.5 };
    ctx.queuePriorities = queuePriorities;

    ctx.window = SDL_CreateWindow(
        "Hello Vulkan",
        640, 480,
        SDL_WINDOW_RESIZABLE | SDL_WINDOW_VULKAN
    );

    if (ctx.window == NULL) {
        SDL_Log("Couldn't create window: %s", SDL_GetError());
        return -1;
    }

    if (initVulkan(&ctx) != 0) {
        SDL_Log("Failed to init vulkan!");
        return -1;
    }

    SDL_Log("Vulkan initialized!");

    int shouldClose = 0;
    while(!shouldClose) {
        for (SDL_Event event; SDL_PollEvent(&event);) {
            switch (event.type){
            case SDL_EVENT_QUIT:
                shouldClose = 1;
                break;
            case SDL_EVENT_WINDOW_RESIZED:
                ctx.framebufferResized = VK_TRUE;
                break;
            }
        }

        drawFrame(&ctx);
    }

    SDL_Log("Program end!");

    cleanup(&ctx);

    return 0;
}


void getVkInstanceExtensions(const char*const** deviceExtensions_out, uint32_t* extensionsCount_out) {
    uint32_t countInstanceExtensions;
    const char * const *instanceExtensions = SDL_Vulkan_GetInstanceExtensions(&countInstanceExtensions);

    char* pBaseExtensions[] = {
        VK_EXT_DEBUG_UTILS_EXTENSION_NAME
    };

    const uint32_t EXTENSIONS_AMOUNT = sizeof(pBaseExtensions) / sizeof(pBaseExtensions[0]);

    uint32_t countExtensions = countInstanceExtensions + EXTENSIONS_AMOUNT;
    SDL_Log("count_extensions: total: %i instance: %i base: %i", countExtensions, countInstanceExtensions, EXTENSIONS_AMOUNT);
    char** extensions = SDL_malloc(countExtensions * sizeof(const char *));
    memcpy(&extensions[0], instanceExtensions, countInstanceExtensions * sizeof(instanceExtensions[0]));
    memcpy(&extensions[countInstanceExtensions], pBaseExtensions, EXTENSIONS_AMOUNT * sizeof(pBaseExtensions[0]));

    SDL_Log("Need extensions:");
    for(uint32_t i = 0; i < countExtensions; i++) {
        SDL_Log("%s", extensions[i]);
    }

    *extensionsCount_out = countExtensions;
    *deviceExtensions_out = (const char*const*)extensions;
}

int isDeviceSuitable(GraphicsContext* ctx, VkPhysicalDevice device) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(device, &props);

    VkPhysicalDeviceFeatures features;
    vkGetPhysicalDeviceFeatures(device, &features);

    if (props.apiVersion < VK_API_VERSION_1_3) 
    {
        SDL_Log("Device [%s] Vulkan version less than 1.3!", props.deviceName);
        return 0;
    }

    uint32_t queuePropertiesCount;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queuePropertiesCount, NULL);
    VkQueueFamilyProperties pQueueProperties[queuePropertiesCount];
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queuePropertiesCount, pQueueProperties);

    VkBool32 hasGraphicsQueue = VK_FALSE;
    for(uint32_t i = 0; i < queuePropertiesCount; i++) 
    {
        VkBool32 isSupportSurface = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, ctx->surface, &isSupportSurface);

        if(pQueueProperties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT && isSupportSurface == VK_TRUE) {
            hasGraphicsQueue = VK_TRUE;
            break;
        }
    }

    if (!hasGraphicsQueue) return 0;

    uint32_t extensionPropertiesCount;
    vkEnumerateDeviceExtensionProperties(device, NULL, &extensionPropertiesCount, NULL);
    VkExtensionProperties extensionProperties[extensionPropertiesCount];
    vkEnumerateDeviceExtensionProperties(device, NULL, &extensionPropertiesCount, extensionProperties);

    uint32_t extFoundCount = 0;
    for(uint32_t i = 0; i < extensionPropertiesCount; i++) {
        for(uint32_t j = 0; j < ctx->deviceExtensionsCount; j++) {
            if(strcmp(extensionProperties[i].extensionName, ctx->deviceExtensions[j]) == 0) {
                extFoundCount += 1;
                break;
            }
        }
    }

    if (extFoundCount != ctx->deviceExtensionsCount) return 0;

    return 1;
}

uint32_t findQueueIdx(GraphicsContext* ctx, VkPhysicalDevice device) {
    uint32_t queuePropertiesCount;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queuePropertiesCount, NULL);
    VkQueueFamilyProperties* queueProperties = calloc(queuePropertiesCount, sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queuePropertiesCount, queueProperties);

    for(uint32_t i = 0; i < queuePropertiesCount; i++) {
        VkBool32 isSurfaceSupported;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, ctx->surface, &isSurfaceSupported);

        if((queueProperties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && (isSurfaceSupported == VK_TRUE)) {
            free(queueProperties);
            return i;
        }
    }

    free(queueProperties);

    return UINT32_MAX;
}

int createLogicalDevice(GraphicsContext* ctx, VkPhysicalDevice device) {
    uint32_t suitableQueueFamilyIdx = findQueueIdx(ctx, device);

    if (suitableQueueFamilyIdx == UINT32_MAX) {
        return -1;
    }

    ctx->presentQueueFamilyIdx = suitableQueueFamilyIdx;
    ctx->graphicsQueueFamilyIdx = suitableQueueFamilyIdx;

    VkDeviceQueueCreateInfo deviceQueueCI = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .queueFamilyIndex = suitableQueueFamilyIdx,
        .pQueuePriorities = ctx->queuePriorities,
        .queueCount = ctx->queueCount,
    };

    VkDeviceCreateInfo deviceCI = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &deviceQueueCI,
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = NULL,
        .pEnabledFeatures = NULL,
        .ppEnabledExtensionNames = ctx->deviceExtensions,
        .enabledExtensionCount = ctx->deviceExtensionsCount,
    };
    
    VkPhysicalDeviceFeatures2 physicalFeatures2 = { 
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = NULL,
    };

    VkPhysicalDeviceVulkan13Features physicalDeviceVk13Features = { 
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .pNext = NULL,
        .synchronization2 = VK_TRUE,
        .dynamicRendering = VK_TRUE,
    };

    VkPhysicalDeviceExtendedDynamicStateFeaturesEXT phyDeviceExtDynStateFeatures = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT,
        .pNext = NULL,
        .extendedDynamicState = 1,
    };

    deviceCI.pNext = &physicalFeatures2;
    physicalFeatures2.pNext = &physicalDeviceVk13Features;
    physicalDeviceVk13Features.pNext = &phyDeviceExtDynStateFeatures;

    if (vkCreateDevice(device, &deviceCI, NULL, &ctx->device) != VK_SUCCESS) {
        return -1;
    }

    SDL_Log("Logical device created!");

    vkGetDeviceQueue(ctx->device, suitableQueueFamilyIdx, 0, &ctx->graphicsQueue);
    vkGetDeviceQueue(ctx->device, ctx->presentQueueFamilyIdx, 0, &ctx->presentQueue);

    return 0;
}


VkSurfaceFormatKHR chooseSwapSurfaceFormat(VkSurfaceFormatKHR* surfaceFormats, uint32_t surfaceFormatsCount) {
    for(uint32_t i = 0; i < surfaceFormatsCount; i++) {
        if(surfaceFormats[i].format == VK_FORMAT_B8G8R8A8_SRGB && surfaceFormats[i].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return surfaceFormats[i];
        }
    }

    return surfaceFormats[0];
}

VkPresentModeKHR chooseSwapPresentMode(VkPresentModeKHR* presentModes, uint32_t presentModesCount) {
    for(uint32_t i = 0; i < presentModesCount; i++) {
        if (presentModes[i] == VK_PRESENT_MODE_MAILBOX_KHR) {
            return presentModes[i];
        }
    }

    return VK_PRESENT_MODE_FIFO_KHR;
}

int clamp(int d, int min, int max) {
  const int t = d < min ? min : d;
  return t > max ? max : t;
}

VkExtent2D chooseSwapExtent(GraphicsContext* ctx, VkSurfaceCapabilitiesKHR* caps) {
    if (caps->currentExtent.height != UINT32_MAX) {
        return caps->currentExtent;
    }

    int width, height;
    SDL_GetWindowSizeInPixels(ctx->window, &width, &height);

    return (struct VkExtent2D) {
        .width = clamp(width, caps->minImageExtent.width, caps->maxImageExtent.width),
        .height = clamp(height, caps->minImageExtent.height, caps->maxImageExtent.height)
    };
}

int createSwapChain(GraphicsContext* ctx, VkPhysicalDevice physical_device) {
    VkSurfaceCapabilitiesKHR surfaceCaps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, ctx->surface, &surfaceCaps);

    uint32_t surfaceFormatsCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, ctx->surface, &surfaceFormatsCount, NULL);
    VkSurfaceFormatKHR* surfaceFormats = calloc(surfaceFormatsCount, sizeof(VkSurfaceFormatKHR));
    vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, ctx->surface, &surfaceFormatsCount, surfaceFormats);

    uint32_t presentModesCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, ctx->surface, &presentModesCount, NULL);
    VkPresentModeKHR* presentModes = calloc(surfaceFormatsCount, sizeof(VkSurfaceFormatKHR));
    vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, ctx->surface, &presentModesCount, presentModes);

    VkSurfaceFormatKHR swapSurfaceFormat = chooseSwapSurfaceFormat(surfaceFormats, surfaceFormatsCount);
    VkPresentModeKHR presentMode = chooseSwapPresentMode(presentModes, presentModesCount);
    VkExtent2D swapExtent = chooseSwapExtent(ctx, &surfaceCaps);

    uint32_t minImageCount = 3 > surfaceCaps.minImageCount ? 3 : surfaceCaps.minImageCount;
    minImageCount = 0 < surfaceCaps.maxImageCount && surfaceCaps.maxImageCount < minImageCount ? surfaceCaps.maxImageCount : minImageCount;

    uint32_t imageCount = surfaceCaps.minImageCount + 1;

    if (surfaceCaps.maxImageCount > 0 && imageCount > surfaceCaps.maxImageCount) {
        imageCount = surfaceCaps.maxImageCount;
    }

    free(surfaceFormats);
    free(presentModes);

    VkSwapchainCreateInfoKHR swapchain_create_info = {
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .surface = ctx->surface,
        .minImageCount = minImageCount,
        .imageFormat = swapSurfaceFormat.format,
        .imageColorSpace = swapSurfaceFormat.colorSpace,
        .imageExtent = swapExtent,
        .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .preTransform = surfaceCaps.currentTransform,
        .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = presentMode,
        .clipped = VK_TRUE,
        .oldSwapchain = VK_NULL_HANDLE,
    };

    uint32_t pQueueFamilyIndicies[2] = {ctx->presentQueueFamilyIdx, ctx->graphicsQueueFamilyIdx};
    if(ctx->graphicsQueueFamilyIdx == ctx->presentQueueFamilyIdx) {
        swapchain_create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        swapchain_create_info.queueFamilyIndexCount = 2;
        swapchain_create_info.pQueueFamilyIndices = pQueueFamilyIndicies;
    } else {
        swapchain_create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        swapchain_create_info.queueFamilyIndexCount = 0;
        swapchain_create_info.pQueueFamilyIndices = VK_NULL_HANDLE;
    }

    VkSwapchainKHR swapChain;
    vkCreateSwapchainKHR(
        ctx->device,
        &swapchain_create_info,
        VK_NULL_HANDLE,
        &swapChain
    );

    uint32_t imagesCount;
    if(vkGetSwapchainImagesKHR(ctx->device, swapChain, &imagesCount, VK_NULL_HANDLE) != VK_SUCCESS) {
        return -1;
    }

    VkImage* images = calloc(imagesCount, sizeof(VkImage));
    if(vkGetSwapchainImagesKHR(ctx->device, swapChain, &imagesCount, images) != VK_SUCCESS) {
        return -1;
    }

    ctx->swapChain = swapChain;
    ctx->swapChainImages = images;
    ctx->swapChainImagesCount = imagesCount;
    ctx->swapChainImageFormat = swapSurfaceFormat.format;
    ctx->swapChainExtent = swapExtent;

    return 0;
}

int cleanupSwapChain(GraphicsContext* ctx) {
    for(uint32_t i = 0; i < ctx->imageViewCount; i++){
        vkDestroyImageView(ctx->device, ctx->imageViews[i], NULL);
    }
    ctx->imageViews = NULL;
    ctx->imageViewCount = 0;

    vkDestroySwapchainKHR(ctx->device, ctx->swapChain, NULL);
    ctx->swapChain = NULL;

    return 0;
}

int createSurface(GraphicsContext* ctx) {
    if(!SDL_Vulkan_CreateSurface(ctx->window, ctx->instance, NULL, &ctx->surface)) {
        SDL_Log("Failed to create surface: %s", SDL_GetError());
        return -1;
    }

    SDL_Log("Created surface!");

    return 0;
}

int createImageViews(GraphicsContext* ctx) {
    VkImageViewCreateInfo image_view_create_info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .flags = 0,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = ctx->swapChainImageFormat,
        .subresourceRange = (VkImageSubresourceRange) {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1
        },
        .components = (VkComponentMapping) {
            .r = VK_COMPONENT_SWIZZLE_IDENTITY,
            .g = VK_COMPONENT_SWIZZLE_IDENTITY,
            .b = VK_COMPONENT_SWIZZLE_IDENTITY,
            .a = VK_COMPONENT_SWIZZLE_IDENTITY
        }
    };

    ctx->imageViewCount = ctx->swapChainImagesCount;
    ctx->imageViews = calloc(ctx->swapChainImagesCount, sizeof(VkImageView));

    for (uint32_t i = 0; i < ctx->swapChainImagesCount; i++) {
        image_view_create_info.image = (ctx->swapChainImages)[i];

        if(vkCreateImageView(ctx->device, &image_view_create_info, VK_NULL_HANDLE, ctx->imageViews + i) != VK_SUCCESS) {
            return -1;
        }
    }
    
    return 0;
}

int recreateSwapChain(GraphicsContext* ctx) {
    vkDeviceWaitIdle(ctx->device);

    cleanupSwapChain(ctx);

    createSwapChain(ctx, ctx->physicalDevice);
    createImageViews(ctx);

    return 0;
}

int readShader(char* path, uint32_t* shaderSize_out, char** shaderCode_out){
    FILE* shaderFile = NULL;

    errno_t fileOpenError = fopen_s(&shaderFile, path, "rb+");

    if(fileOpenError) {
        return -1;
    }

    fseek(shaderFile, 0, SEEK_END);
    uint32_t shaderSize = ftell(shaderFile);

    char* shaderCode = NULL;
    shaderCode = calloc(shaderSize, sizeof(char));

    rewind(shaderFile);
    fread(shaderCode, 1, shaderSize, shaderFile);
    fclose(shaderFile);

    *shaderSize_out = shaderSize;
    *shaderCode_out = shaderCode;

    return 0;
}

int createShaderModule(GraphicsContext* ctx, char* path, VkShaderModule* shaderModule) {
    uint32_t shaderSize;
    char* shaderCode = NULL;

    if (readShader(path, &shaderSize, &shaderCode)) {
        SDL_Log("Failed to read shader!");
        return -1;
    }

    SDL_Log("Readed shader! Size: %i bytes", shaderSize);
 
    VkShaderModuleCreateInfo shaderModuleCI = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = shaderSize,
        .pCode = (const uint32_t*)shaderCode,
        .pNext = NULL,
        .flags = 0
    };

    if(vkCreateShaderModule(ctx->device, &shaderModuleCI, VK_NULL_HANDLE, shaderModule) != VK_SUCCESS) {
        SDL_Log("Failed create shader module!");
        return -1;
    }

    return 0;
}

int createGraphicsPipeline(GraphicsContext* ctx) {
    VkShaderModule shaderModule;
    
    if(createShaderModule(ctx, "./slang.spv", &shaderModule)) {
        SDL_Log("Error creating shader!");
        return -1;
    }

    SDL_Log("Shader created!");

    VkPipelineShaderStageCreateInfo shaderStageCI_vertex = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT,
        .module = shaderModule,
        .pName = "vertMain" 
    };

    VkPipelineShaderStageCreateInfo shaderStageCI_frag = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
        .module = shaderModule,
        .pName = "fragMain" 
    };

    VkPipelineShaderStageCreateInfo shaderStages[] = {shaderStageCI_vertex, shaderStageCI_frag};

    VkDynamicState pDynamicStates[] = {VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_VIEWPORT};

    VkPipelineDynamicStateCreateInfo dynamicStateCI = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = 2,
        .pDynamicStates = pDynamicStates,
    };

    VkPipelineVertexInputStateCreateInfo vertexInputStateCI = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .vertexBindingDescriptionCount = 0,
        .pVertexBindingDescriptions = NULL,
        .vertexAttributeDescriptionCount = 0,
        .pVertexAttributeDescriptions = NULL
    };

    VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCI = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    };

    VkViewport viewport = {
        .x = 0,
        .y = 0,
        .width = (float)ctx->swapChainExtent.width,
        .height = (float)ctx->swapChainExtent.height,
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };

    VkRect2D scissor = {
        .extent = ctx->swapChainExtent,
        .offset = (VkOffset2D){
            .x = 0, .y = 0
        }
    };

    VkPipelineViewportStateCreateInfo viewportStateCI = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .flags = 0,
        .pNext = NULL,
        .pViewports = &viewport,
        .pScissors = &scissor,
        .viewportCount = 1,
        .scissorCount = 1
    };

    VkPipelineRasterizationStateCreateInfo rasterizationStateCI = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .depthClampEnable = VK_FALSE,
        .rasterizerDiscardEnable = VK_FALSE,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_BACK_BIT,
        .frontFace = VK_FRONT_FACE_CLOCKWISE,
        .depthBiasEnable = VK_FALSE,
        .depthBiasConstantFactor = 0.0f,
        .depthBiasClamp = 0.0f,
        .depthBiasSlopeFactor = 1.0f,
        .lineWidth = 1.0f,
    };

    VkPipelineMultisampleStateCreateInfo multisampleStateCI = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        .sampleShadingEnable = VK_FALSE,
        .minSampleShading = 1.0f,
        .pSampleMask = NULL,
        .alphaToCoverageEnable = VK_FALSE,
        .alphaToOneEnable = VK_FALSE,
    };

    VkPipelineColorBlendAttachmentState colorBlendAttachmentState = {
        .blendEnable = VK_FALSE,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
        .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
        .colorBlendOp = VK_BLEND_OP_ADD,
        .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
        .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
        .alphaBlendOp = VK_BLEND_OP_ADD,
    };

    VkPipelineColorBlendStateCreateInfo colorBlendStateCI = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .logicOpEnable = VK_FALSE,
        .logicOp = VK_LOGIC_OP_COPY,
        .attachmentCount = 1,
        .pAttachments = &colorBlendAttachmentState,
        .blendConstants = {0.0f, 0.0f, 0.0f, 0.0f},
        .flags = 0,
        .pNext = NULL,
    };

    VkPipelineLayoutCreateInfo pipeLayoutCI = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .setLayoutCount = 0,
        .pushConstantRangeCount = 0,
        .pSetLayouts = NULL,
        .pPushConstantRanges = NULL,
    };

    if(vkCreatePipelineLayout(ctx->device, &pipeLayoutCI, VK_NULL_HANDLE, &ctx->pipelineLayout) != VK_SUCCESS) {
        SDL_Log("Failed to create pipeline layout!");
        return -1;
    }

    VkPipelineRenderingCreateInfo pipeRenderingCI = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .pNext = NULL,
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &ctx->swapChainImageFormat,
        .depthAttachmentFormat = VK_FORMAT_UNDEFINED,
        .stencilAttachmentFormat = VK_FORMAT_UNDEFINED,
    };

    VkGraphicsPipelineCreateInfo pipelineCI = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &pipeRenderingCI,
        .flags = 0,
        .stageCount = 2,
        .pStages = shaderStages,
        .pVertexInputState = &vertexInputStateCI,
        .pInputAssemblyState = &inputAssemblyStateCI,
        .pViewportState = &viewportStateCI,
        .pRasterizationState = &rasterizationStateCI,
        .pMultisampleState = &multisampleStateCI,
        .pColorBlendState = &colorBlendStateCI,
        .pDynamicState = &dynamicStateCI,
        .layout = ctx->pipelineLayout,
        .renderPass = NULL,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = -1,
        .subpass = 0,
    };

    if(vkCreateGraphicsPipelines(ctx->device, NULL, 1, &pipelineCI, NULL, &ctx->graphicsPipeline) != VK_SUCCESS) {
        SDL_Log("Failed to create graphics pipeline!");
        vkDestroyShaderModule(ctx->device, shaderModule, NULL);
        return -1;
    }

    vkDestroyShaderModule(ctx->device, shaderModule, NULL);

    return 0;
}

int createCommandPool(GraphicsContext* ctx) {
    VkCommandPoolCreateInfo commandPoolCI = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = NULL,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = ctx->graphicsQueueFamilyIdx,
    };

    if(vkCreateCommandPool(ctx->device, &commandPoolCI, NULL, &ctx->commandPool) != VK_SUCCESS) {
        SDL_Log("Failed create command pool!");
        return -1;
    }

    return 0;
}

int createCommandBuffers(GraphicsContext* ctx) {
    VkCommandBufferAllocateInfo commandBufferAI = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = NULL,
        .commandPool = ctx->commandPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = MAX_FRAMES_IN_FLIGHT,
    };

    if(vkAllocateCommandBuffers(ctx->device, &commandBufferAI, ctx->commandBuffer) != VK_SUCCESS) {
        SDL_Log("Failed allocate command buffer!");
        return -1;
    }
    
    return 0;
}

int transitionImageLayout(
    GraphicsContext* ctx,
    uint32_t imageIndex,
    VkImageLayout oldLayout,
    VkImageLayout newLayout,
    VkAccessFlags2 srcAccessMask,
    VkAccessFlags2 dstAccessMask,
    VkPipelineStageFlags2 srcStageMask,
    VkPipelineStageFlags2 dstStageMask
) {
    VkImageMemoryBarrier2 barrier = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .pNext = NULL,
        .srcStageMask = srcStageMask,
        .dstStageMask = dstStageMask,
        .srcAccessMask = srcAccessMask,
        .dstAccessMask = dstAccessMask,
        .oldLayout = oldLayout,
        .newLayout = newLayout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = ctx->swapChainImages[imageIndex],
        .subresourceRange = (VkImageSubresourceRange) {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        }
    };

    VkDependencyInfo depInfo = {
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .pNext = NULL,
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &barrier,
    };

    vkCmdPipelineBarrier2(ctx->commandBuffer[ctx->currentFrameIndex], &depInfo);

    return 0;
}

int recordCommandBuffer(GraphicsContext* ctx, uint32_t imageIndex) {
    VkCommandBuffer commandBuffer = ctx->commandBuffer[ctx->currentFrameIndex];

    VkCommandBufferBeginInfo commandBufferBeginInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    };

    if(vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo) != VK_SUCCESS) {
        SDL_Log("Failed begin!");
        return -1;
    }

    transitionImageLayout(
        ctx,
        imageIndex,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_ACCESS_2_NONE,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT
    );

    VkClearColorValue clearValue = {
        .float32 = {0.0f, 0.0f, 0.0f, 1.0f},
    };

    VkRenderingAttachmentInfo renderingAttachmentInfo = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = ctx->imageViews[imageIndex],
        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue = { clearValue },
    };

    VkRenderingInfo renderingInfo = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea = (VkRect2D) {
            .offset = (VkOffset2D){
                .x = 0,
                .y = 0
            },
            .extent = ctx->swapChainExtent
        },
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &renderingAttachmentInfo,
    };

    vkCmdBeginRendering(commandBuffer, &renderingInfo);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, ctx->graphicsPipeline);

    VkViewport viewport = {
        .x = 0,
        .y = 0,
        .width = (float)ctx->swapChainExtent.width,
        .height = (float)ctx->swapChainExtent.height,
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };

    vkCmdSetViewport(
        commandBuffer,
        0,
        1,
        &viewport
    );

    
    VkRect2D scissor = {
        .extent = ctx->swapChainExtent,
        .offset = (VkOffset2D){
            .x = 0, .y = 0
        }
    };

    vkCmdSetScissor(
        commandBuffer,
        0,
        1,
        &scissor
    );

    vkCmdDraw(commandBuffer, 3, 1, 0, 0);

    vkCmdEndRendering(commandBuffer);

    transitionImageLayout(
        ctx,
        imageIndex,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT
    );

    vkEndCommandBuffer(commandBuffer);

    return 0;
}

int createSyncObjects(GraphicsContext* ctx) {
    for (uint32_t i = 0; i < ctx->swapChainImagesCount; i++) 
    {
        VkSemaphoreCreateInfo semaphore_create_info = {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            .pNext = NULL,
            .flags = 0,
        };
        vkCreateSemaphore(ctx->device, &semaphore_create_info, NULL, &ctx->renderFinishedSemaphore[i]);
    }
    
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) 
    {
        VkSemaphoreCreateInfo semaphore_create_info = {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            .pNext = NULL,
            .flags = 0,
        };
        vkCreateSemaphore(ctx->device, &semaphore_create_info, NULL, &ctx->presentCompleteSemaphore[i]);

        VkFenceCreateInfo fence_create_info = {
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .flags = VK_FENCE_CREATE_SIGNALED_BIT,
            .pNext = NULL,
        };
        vkCreateFence(ctx->device, &fence_create_info, NULL, &ctx->drawFence[i]);
    }

    return 0;
}

void drawFrame(GraphicsContext* ctx) {
    while (VK_TIMEOUT == vkWaitForFences(ctx->device, 1, &ctx->drawFence[ctx->currentFrameIndex], VK_TRUE, UINT64_MAX))
        ;

    uint32_t imageIndex;
    VkResult acquireNextImageResult = vkAcquireNextImageKHR(ctx->device, ctx->swapChain, UINT64_MAX, ctx->presentCompleteSemaphore[ctx->currentFrameIndex], NULL, &imageIndex);
    if(acquireNextImageResult == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapChain(ctx);
        return;
    }else if(acquireNextImageResult != VK_SUCCESS && acquireNextImageResult != VK_SUBOPTIMAL_KHR) {
        SDL_Log("Failed aquire next image!");
        return;
    }

    vkResetFences(ctx->device, 1, &ctx->drawFence[ctx->currentFrameIndex]);

    VkCommandBufferResetFlags resetFlags = 0;
    vkResetCommandBuffer(ctx->commandBuffer[ctx->currentFrameIndex], resetFlags);
    recordCommandBuffer(ctx, imageIndex);

    VkPipelineStageFlags waitDstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pWaitSemaphores = &ctx->presentCompleteSemaphore[ctx->currentFrameIndex],
        .waitSemaphoreCount = 1,
        .pWaitDstStageMask = &waitDstStageMask,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &ctx->renderFinishedSemaphore[imageIndex],
        .pCommandBuffers = &ctx->commandBuffer[ctx->currentFrameIndex],
        .commandBufferCount = 1,
    };

    vkQueueSubmit(ctx->graphicsQueue, 1, &submit_info, ctx->drawFence[ctx->currentFrameIndex]);

    VkPresentInfoKHR presentInfo = {
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .pWaitSemaphores = &ctx->renderFinishedSemaphore[imageIndex],
        .waitSemaphoreCount = 1,
        .pSwapchains = &ctx->swapChain,
        .swapchainCount = 1,
        .pImageIndices = &imageIndex,
        .pResults = NULL,
    };

    VkResult queuePresentResult = vkQueuePresentKHR(ctx->graphicsQueue, &presentInfo);

    if(queuePresentResult == VK_ERROR_OUT_OF_DATE_KHR || ctx->framebufferResized) {
        ctx->framebufferResized = VK_FALSE;
        recreateSwapChain(ctx);
    } else if (queuePresentResult != VK_SUCCESS && queuePresentResult != VK_SUBOPTIMAL_KHR) {
        SDL_Log("Failed present queue!");
        return;
    }

    ctx->currentFrameIndex = (ctx->currentFrameIndex + 1) % MAX_FRAMES_IN_FLIGHT;
}

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity, 
    VkDebugUtilsMessageTypeFlagsEXT type, 
    const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
    void* _
) {
    if (1 || severity == VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT || severity == VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
    {
        SDL_Log("validation layer: type: %i msg: %s", type, pCallbackData->pMessage);
    }

    return VK_FALSE;
}

int initVulkan(GraphicsContext* ctx) {
	char* appName = "Hello Vulkan";
	char* appEngineName = "No engine";

    VkApplicationInfo appInfo = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pNext = NULL,
    	.pApplicationName = appName,
	    .applicationVersion = VK_MAKE_VERSION(0,0,1),
    	.pEngineName = appEngineName,
        .engineVersion = VK_MAKE_VERSION(0,0,1),
        .apiVersion = VK_API_VERSION_1_3,
    };

    VkInstanceCreateInfo instanceCI = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .pApplicationInfo = &appInfo,
        .enabledExtensionCount = ctx->instanceExtensionsCount,
        .ppEnabledExtensionNames = ctx->instanceExtensions,
        .enabledLayerCount = LAYER_COUNT,
        .ppEnabledLayerNames = pInstanceEnabledLayers,
    };

    uint32_t layerPropsCount;
    vkEnumerateInstanceLayerProperties(&layerPropsCount, NULL);

    VkLayerProperties* pLayerProperties = calloc(layerPropsCount, sizeof(VkLayerProperties));
    vkEnumerateInstanceLayerProperties(&layerPropsCount, pLayerProperties);

    for(uint32_t i = 0; i < layerPropsCount; i++) {
        SDL_Log("Has Layer: %s", pLayerProperties[i].layerName);
    }

    if (vkCreateInstance(&instanceCI, NULL, &ctx->instance) != VK_SUCCESS) {
        SDL_Log("Failed to create Vulkan instance!");
    } else {
        SDL_Log("Vulkan instance created!");
    }

    VkDebugUtilsMessengerCreateInfoEXT debugCI = {
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT,
        .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
        .pfnUserCallback = debugCallback,
    };

    PFN_vkCreateDebugUtilsMessengerEXT vkCreateDebugUtilsMessengerEXT = 
    (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
        ctx->instance, "vkCreateDebugUtilsMessengerEXT");


    vkCreateDebugUtilsMessengerEXT(ctx->instance, &debugCI, NULL, &ctx->debugMessanger);

    if(createSurface(ctx)) {
        return -1;
    }

    uint32_t deviceCount = 0;
	vkEnumeratePhysicalDevices(ctx->instance,&deviceCount,NULL);

    SDL_Log("Available devices count: %i", deviceCount);

    VkPhysicalDevice pPhysicalDevices[16];
	vkEnumeratePhysicalDevices(ctx->instance,&deviceCount, pPhysicalDevices);
    VkPhysicalDevice* physicalDevice = NULL;

    for (uint32_t i = 0; i < deviceCount; i++) {
        if (isDeviceSuitable(ctx, pPhysicalDevices[i])) {
            physicalDevice = &pPhysicalDevices[i];
            break;
        }
    }

    if (physicalDevice == NULL) {
        SDL_Log("Couldn't find suitable device!");
        return -1;
    }

    ctx->physicalDevice = *physicalDevice;

    VkPhysicalDeviceProperties physicalDeviceProperties;
    vkGetPhysicalDeviceProperties(*physicalDevice, &physicalDeviceProperties);

    SDL_Log("Picked device: %s", physicalDeviceProperties.deviceName);

    if(createLogicalDevice(ctx, *physicalDevice)){
        SDL_Log("Failed to create logical device!");
        return -1;
    }

    if(createSwapChain(ctx, *physicalDevice)) {
        SDL_Log("Failed to create swapChain!");
        return -1;
    }

    SDL_Log("Swapchain created!");

    if(createImageViews(ctx)){
        SDL_Log("Failed to create image views!");
        return -1;
    }

    SDL_Log("Image views created!");

    if(createGraphicsPipeline(ctx)){
        SDL_Log("Failed create graphics pipeline!");
        return -1;
    }

    SDL_Log("Graphics pipeline created!");

    if(createCommandPool(ctx)) {
        return -1;
    }

    SDL_Log("Command pool created!");

    if(createCommandBuffers(ctx)) {
        return -1;
    }

    SDL_Log("Allocated command buffer!");

    if(createSyncObjects(ctx)) {
        return -1;
    }

    SDL_Log("Sync objects created!");

    return 0;
}

void cleanup(GraphicsContext* ctx) {
    vkDeviceWaitIdle(ctx->device);

    vkFreeCommandBuffers(ctx->device, ctx->commandPool, sizeof(ctx->commandBuffer) / sizeof(VkCommandBuffer), ctx->commandBuffer);

    for (uint32_t i = 0; i < sizeof(ctx->renderFinishedSemaphore) / sizeof(VkSemaphore); i++) 
    {
        vkDestroySemaphore(ctx->device, ctx->renderFinishedSemaphore[i], NULL);
    }

    for (uint32_t i = 0; i < sizeof(ctx->presentCompleteSemaphore) / sizeof(VkSemaphore); i++) 
    {
        vkDestroySemaphore(ctx->device, ctx->presentCompleteSemaphore[i], NULL);
    }

    for (uint32_t i = 0; i < sizeof(ctx->drawFence) / sizeof(VkFence); i++) 
    {
        vkDestroyFence(ctx->device, ctx->drawFence[i], NULL);
    }

    vkDestroyCommandPool(ctx->device, ctx->commandPool, NULL);
    vkDestroyPipeline(ctx->device, ctx->graphicsPipeline, NULL);
    vkDestroyPipelineLayout(ctx->device, ctx->pipelineLayout, NULL);

    for (uint32_t i = 0; i < ctx->imageViewCount; i++)
    {
        vkDestroyImageView(ctx->device, ctx->imageViews[i], NULL);
    }
    
    vkDestroySwapchainKHR(ctx->device, ctx->swapChain, NULL);

    vkDestroyDevice(ctx->device, NULL);

    SDL_Vulkan_DestroySurface(ctx->instance, ctx->surface, NULL);

    PFN_vkDestroyDebugUtilsMessengerEXT vkDestroyDebugUtilsMessengerEXT = 
    (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
        ctx->instance, "vkDestroyDebugUtilsMessengerEXT");

    vkDestroyDebugUtilsMessengerEXT(ctx->instance, ctx->debugMessanger, NULL);

    vkDestroyInstance(ctx->instance, NULL);

    SDL_DestroyWindow(ctx->window);
}
