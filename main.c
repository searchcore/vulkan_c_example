#include <stdint.h>
#include <stdio.h>

#include <vulkan/vulkan.h>

#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <SDL3/SDL_vulkan.h>


static const int MAX_FRAMES_IN_FLIGHT = 2;
static const int MAX_IMAGES_COUNT = 8;

typedef struct GraphicsContext {
    SDL_Window* window;

    VkInstance inst;
    VkApplicationInfo app_info;
    VkDevice device;
    VkPhysicalDevice physicalDevice;

    uint32_t queues_count;
    float* queue_priorities;

    VkQueue queue;
    VkQueue present_queue;
    VkSurfaceKHR surface;

    char** device_extensions;
    uint32_t device_extensions_count;

    char** instance_extensions;
    uint32_t instance_extensions_count;

    uint32_t q_present_family_idx;
    uint32_t q_graphics_family_idx;

    VkSwapchainKHR swapchain;
    uint32_t swapchain_images_count;
    VkImage* swapchain_images;
    VkFormat swapchain_image_format;
    VkExtent2D swapchain_extent;

    uint32_t image_view_count;
    VkImageView* image_views;

    VkPipelineLayout pipeline_layout;
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


int init_vulkan(GraphicsContext* g_ctx);
void get_vk_device_extensions(char*** ext_out, uint32_t* extensions_count);
void get_vk_instance_extensions(char*** ext_out, uint32_t* extensions_count);
void cleanup(GraphicsContext* g_ctx);
void drawFrame(GraphicsContext* g_ctx);

int main(int argc, char **argv) {
    GraphicsContext g_ctx;

    g_ctx.framebufferResized = VK_FALSE;
    g_ctx.currentFrameIndex = 0;
    g_ctx.queues_count = 1;
    float queue_priorities[1] = { 0.5 };
    g_ctx.queue_priorities = queue_priorities;

    if (!SDL_Init(SDL_INIT_VIDEO)) {
        SDL_Log("Couldn't initialize SDL: %s", SDL_GetError());
        return -1;
    }

    SDL_Vulkan_LoadLibrary(NULL);

    g_ctx.window = SDL_CreateWindow(
        "Hello Vulkan",
        640, 480,
        SDL_WINDOW_RESIZABLE | SDL_WINDOW_VULKAN
    );

    if (g_ctx.window == NULL) {
        SDL_Log("Couldn't create window: %s", SDL_GetError());
        return -1;
    }

    get_vk_device_extensions(&g_ctx.device_extensions, &g_ctx.device_extensions_count);
    get_vk_instance_extensions(&g_ctx.instance_extensions, &g_ctx.instance_extensions_count);

    if (init_vulkan(&g_ctx) != 0) {
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
                g_ctx.framebufferResized = VK_TRUE;
                break;
            }
        }

        drawFrame(&g_ctx);
    }

    SDL_Log("Program end!");

    cleanup(&g_ctx);

    return 0;
}

void get_vk_device_extensions(char*** ext_out, uint32_t* extensions_count) {
    char* base_extensions[] = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_KHR_SPIRV_1_4_EXTENSION_NAME,
        VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
        VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME,
    };

    #define EXTENSIONS_AMOUNT sizeof(base_extensions) / sizeof(base_extensions[0])
    char** extensions = SDL_malloc(EXTENSIONS_AMOUNT * sizeof(const char *));
    memcpy(&extensions[0], base_extensions, EXTENSIONS_AMOUNT * sizeof(base_extensions[0]));

    *extensions_count = EXTENSIONS_AMOUNT;
    #undef EXTENSIONS_AMOUNT

    *ext_out = extensions;
}

void get_vk_instance_extensions(char*** ext_out, uint32_t* extensions_count) {
    uint32_t count_instance_extensions;
    const char * const *instance_extensions = SDL_Vulkan_GetInstanceExtensions(&count_instance_extensions);

    char* base_extensions[] = {
        VK_EXT_DEBUG_UTILS_EXTENSION_NAME
    };

    #define EXTENSIONS_AMOUNT sizeof(base_extensions) / sizeof(base_extensions[0])
    int count_extensions = count_instance_extensions + EXTENSIONS_AMOUNT;
    SDL_Log("count_extensions: total: %i instance: %i base: %i", count_extensions, count_instance_extensions, EXTENSIONS_AMOUNT);
    char** extensions = SDL_malloc(count_extensions * sizeof(const char *));
    memcpy(&extensions[0], instance_extensions, count_instance_extensions * sizeof(instance_extensions[0]));
    memcpy(&extensions[count_instance_extensions], base_extensions, EXTENSIONS_AMOUNT * sizeof(base_extensions[0]));
    #undef EXTENSIONS_AMOUNT

    SDL_Log("Need extensions:");
    for(int i = 0; i < count_extensions; i++) {
        SDL_Log("%s", extensions[i]);
    }

    *extensions_count = count_extensions;
    *ext_out = extensions;
}

int is_device_suitable(GraphicsContext* g_ctx, VkPhysicalDevice device) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(device, &props);

    VkPhysicalDeviceFeatures features;
    vkGetPhysicalDeviceFeatures(device, &features);

    SDL_Log("Checking device vulkan version...");

    if (props.apiVersion < VK_API_VERSION_1_3) return 0;

    uint32_t q_props_count;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &q_props_count, NULL);
    VkQueueFamilyProperties* q_props = calloc(q_props_count, sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties(device, &q_props_count, q_props);

    SDL_Log("Checking device queues...");

    int has_graphics_queue = 0;
    for(int i = 0; i < q_props_count; i++) {
        VkBool32 support_surface;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, g_ctx->surface, &support_surface);

        if (support_surface == VK_FALSE) break;

        if(q_props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            has_graphics_queue = 1;
            SDL_Log("Has graphics queue!");
            break;
        }
    }

    free(q_props);

    if (!has_graphics_queue) return 0;

    uint32_t ext_props_count;
    vkEnumerateDeviceExtensionProperties(device, NULL, &ext_props_count, NULL);
    VkExtensionProperties* ext_props = calloc(ext_props_count, sizeof(VkExtensionProperties));
    vkEnumerateDeviceExtensionProperties(device, NULL, &ext_props_count, ext_props);

    SDL_Log("Checking device extensions...");

    int found_ext = 0;
    for(int i = 0; i < ext_props_count; i++) {
        //SDL_Log("Device extension: %s", ext_props[i].extensionName);
        for(int j = 0; j < g_ctx->device_extensions_count; j++) {
            if(strcmp(ext_props[i].extensionName, g_ctx->device_extensions[j]) == 0) {
                found_ext += 1;
                break;
            }
        }
    }

    free(ext_props);

    if (found_ext != g_ctx->device_extensions_count) return 0;

    SDL_Log("Extensions OK!");

    #undef EXTENSIONS_AMOUNT

    return 1;
}

uint32_t find_queue_index(GraphicsContext* g_ctx, VkPhysicalDevice device) {
    uint32_t q_props_count;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &q_props_count, NULL);
    VkQueueFamilyProperties* q_props = calloc(q_props_count, sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties(device, &q_props_count, q_props);

    for(int i = 0; i < q_props_count; i++) {
        VkBool32 support_surface;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, g_ctx->surface, &support_surface);

        if((q_props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && (support_surface == VK_TRUE)) {
            free(q_props);
            return i;
        }
    }

    free(q_props);

    return UINT32_MAX;
}

int create_logical_device(GraphicsContext* g_ctx, VkPhysicalDevice device) {
    uint32_t queue_family_idx = find_queue_index(g_ctx, device);

    if (queue_family_idx == UINT32_MAX) {
        return -1;
    }

    g_ctx->q_present_family_idx = queue_family_idx;
    g_ctx->q_graphics_family_idx = queue_family_idx;

    VkDeviceQueueCreateInfo device_q_create_info;
    device_q_create_info.queueFamilyIndex = queue_family_idx;
    device_q_create_info.pQueuePriorities = g_ctx->queue_priorities;
    device_q_create_info.queueCount = g_ctx->queues_count;
    device_q_create_info.pNext = NULL;
    device_q_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    device_q_create_info.flags = 0;

    VkDeviceCreateInfo device_create_info;
    device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_create_info.pNext = NULL;
    device_create_info.flags = 0;
    device_create_info.queueCreateInfoCount = 1;
    device_create_info.pQueueCreateInfos = &device_q_create_info;
    device_create_info.enabledLayerCount = 0;
    device_create_info.ppEnabledLayerNames = NULL;
    device_create_info.pEnabledFeatures = NULL;

    device_create_info.ppEnabledExtensionNames = g_ctx->device_extensions;
    device_create_info.enabledExtensionCount = g_ctx->device_extensions_count;
    
    VkPhysicalDeviceFeatures2 physical_features_2 = { 
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = NULL,
    };
    VkPhysicalDeviceVulkan13Features phy_dev_vulkan_13_feat = { 
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .pNext = NULL,
        .synchronization2 = VK_TRUE,
        .dynamicRendering = VK_TRUE,
    };
    VkPhysicalDeviceExtendedDynamicStateFeaturesEXT feat_ext_dyn = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT,
        .pNext = NULL,
        .extendedDynamicState = 1,
    };

    device_create_info.pNext = &physical_features_2;
    physical_features_2.pNext = &phy_dev_vulkan_13_feat;
    phy_dev_vulkan_13_feat.pNext = &feat_ext_dyn;

    if (vkCreateDevice(device, &device_create_info, NULL, &g_ctx->device) != VK_SUCCESS) {
        return -1;
    }

    SDL_Log("Logical device created!");

    vkGetDeviceQueue(g_ctx->device, queue_family_idx, 0, &g_ctx->queue);
    vkGetDeviceQueue(g_ctx->device, g_ctx->q_present_family_idx, 0, &g_ctx->present_queue);

    return 0;
}


VkSurfaceFormatKHR choose_swap_surface_format(VkSurfaceFormatKHR* surfaceFormats, uint32_t surfaceFormatsCount) {
    for(int i = 0; i < surfaceFormatsCount; i++) {
        if(surfaceFormats[i].format == VK_FORMAT_B8G8R8A8_SRGB && surfaceFormats[i].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return surfaceFormats[i];
        }
    }

    return surfaceFormats[0];
}

VkPresentModeKHR choose_swap_present_mode(VkPresentModeKHR* presentModes, uint32_t presentModesCount) {
    for(int i = 0; i < presentModesCount; i++) {
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

VkExtent2D choose_swap_extent(GraphicsContext* g_ctx, VkSurfaceCapabilitiesKHR* caps) {
    if (caps->currentExtent.height != UINT32_MAX) {
        return caps->currentExtent;
    }

    int width, height;
    SDL_GetWindowSizeInPixels(g_ctx->window, &width, &height);

    return (struct VkExtent2D) {
        .width = clamp(width, caps->minImageExtent.width, caps->maxImageExtent.width),
        .height = clamp(height, caps->minImageExtent.height, caps->maxImageExtent.height)
    };
}

int create_swap_chain(GraphicsContext* g_ctx, VkPhysicalDevice physical_device) {
    VkSurfaceCapabilitiesKHR surfaceCaps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, g_ctx->surface, &surfaceCaps);

    uint32_t surfaceFormatsCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, g_ctx->surface, &surfaceFormatsCount, NULL);
    VkSurfaceFormatKHR* surfaceFormats = calloc(surfaceFormatsCount, sizeof(VkSurfaceFormatKHR));
    vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, g_ctx->surface, &surfaceFormatsCount, surfaceFormats);

    uint32_t presentModesCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, g_ctx->surface, &presentModesCount, NULL);
    VkPresentModeKHR* presentModes = calloc(surfaceFormatsCount, sizeof(VkSurfaceFormatKHR));
    vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, g_ctx->surface, &presentModesCount, presentModes);

    VkSurfaceFormatKHR swapSurfaceFormat = choose_swap_surface_format(surfaceFormats, surfaceFormatsCount);
    VkPresentModeKHR presentMode = choose_swap_present_mode(presentModes, presentModesCount);
    VkExtent2D swapExtent = choose_swap_extent(g_ctx, &surfaceCaps);

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
        .surface = g_ctx->surface,
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

    uint32_t q_indicies[2] = {g_ctx->q_present_family_idx, g_ctx->q_graphics_family_idx};
    if(g_ctx->q_graphics_family_idx == g_ctx->q_present_family_idx) {
        swapchain_create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        swapchain_create_info.queueFamilyIndexCount = 2;
        swapchain_create_info.pQueueFamilyIndices = q_indicies;
    } else {
        swapchain_create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        swapchain_create_info.queueFamilyIndexCount = 0;
        swapchain_create_info.pQueueFamilyIndices = VK_NULL_HANDLE;
    }

    VkSwapchainKHR swapchain;
    vkCreateSwapchainKHR(
        g_ctx->device,
        &swapchain_create_info,
        VK_NULL_HANDLE,
        &swapchain
    );

    uint32_t images_count;
    if(vkGetSwapchainImagesKHR(g_ctx->device, swapchain, &images_count, VK_NULL_HANDLE) != VK_SUCCESS) {
        return -1;
    }

    VkImage* images = calloc(images_count, sizeof(VkImage));
    if(vkGetSwapchainImagesKHR(g_ctx->device, swapchain, &images_count, images) != VK_SUCCESS) {
        return -1;
    }

    g_ctx->swapchain = swapchain;
    g_ctx->swapchain_images = images;
    g_ctx->swapchain_images_count = images_count;
    g_ctx->swapchain_image_format = swapSurfaceFormat.format;
    g_ctx->swapchain_extent = swapExtent;

    return 0;
}

int cleanupSwapChain(GraphicsContext* g_ctx) {
    for(int i = 0; i < g_ctx->image_view_count; i++){
        vkDestroyImageView(g_ctx->device, g_ctx->image_views[i], NULL);
    }
    g_ctx->image_views = NULL;
    g_ctx->image_view_count = 0;

    vkDestroySwapchainKHR(g_ctx->device, g_ctx->swapchain, NULL);
    g_ctx->swapchain = NULL;
}

int create_surface(GraphicsContext* g_ctx) {
    if(!SDL_Vulkan_CreateSurface(g_ctx->window, g_ctx->inst, NULL, &g_ctx->surface)) {
        SDL_Log("Failed to create surface: %s", SDL_GetError());
        return -1;
    }

    SDL_Log("Created surface!");

    return 0;
}

int create_image_views(GraphicsContext* g_ctx) {
    VkImageViewCreateInfo image_view_create_info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .flags = 0,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = g_ctx->swapchain_image_format,
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

    g_ctx->image_view_count = g_ctx->swapchain_images_count;
    g_ctx->image_views = calloc(g_ctx->swapchain_images_count, sizeof(VkImageView));

    for (int i = 0; i < g_ctx->swapchain_images_count; i++) {
        image_view_create_info.image = (g_ctx->swapchain_images)[i];

        if(vkCreateImageView(g_ctx->device, &image_view_create_info, VK_NULL_HANDLE, g_ctx->image_views + i) != VK_SUCCESS) {
            return -1;
        }
    }
    
    return 0;
}

int recreateSwapChain(GraphicsContext* g_ctx) {
    vkDeviceWaitIdle(g_ctx->device);

    cleanupSwapChain(g_ctx);

    create_swap_chain(g_ctx, g_ctx->physicalDevice);
    create_image_views(g_ctx);
}

int readShader(char* path, uint32_t* shader_size_out, char** shader_code){
    FILE* shaderFile = fopen("./slang.spv", "rb+");

    if (shaderFile == NULL) {
        return -1;
    }

    fseek(shaderFile, 0, SEEK_END);
    uint32_t shader_size = ftell(shaderFile);

    char* shaderCode = NULL;
    shaderCode = calloc(shader_size, sizeof(char));

    rewind(shaderFile);
    fread(shaderCode, 1, shader_size, shaderFile);
    fclose(shaderFile);

    *shader_size_out = shader_size;
    *shader_code = shaderCode;

    return 0;
}

int create_shader_module(GraphicsContext* g_ctx, char* path, VkShaderModule* shaderModule) {
    uint32_t shader_size;
    char* shaderCode = NULL;

    if (readShader("shader.spv", &shader_size, &shaderCode)) {
        SDL_Log("Failed to read shader!");
        return -1;
    }

    SDL_Log("Readed shader! %i, %i", shader_size, shaderCode);
 
    VkShaderModuleCreateInfo shader_create_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = shader_size,
        .pCode = (const uint32_t*)shaderCode,
        .pNext = NULL,
        .flags = 0
    };

    if(vkCreateShaderModule(g_ctx->device, &shader_create_info, VK_NULL_HANDLE, shaderModule) != VK_SUCCESS) {
        SDL_Log("Failed create shader module!");
        return -1;
    }

    return 0;
}

int create_graphics_pipeline(GraphicsContext* g_ctx) {
    VkShaderModule shaderModule;
    
    if(create_shader_module(g_ctx, "shader.spv", &shaderModule)) {
        SDL_Log("Error creating shader!");
        return -1;
    }

    SDL_Log("Shader created!");

    VkPipelineShaderStageCreateInfo vert_stage_create_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT,
        .module = shaderModule,
        .pName = "vertMain" 
    };

    VkPipelineShaderStageCreateInfo frag_stage_create_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
        .module = shaderModule,
        .pName = "fragMain" 
    };

    VkPipelineShaderStageCreateInfo shaderStages[] = {vert_stage_create_info, frag_stage_create_info};

    VkDynamicState dyn_states[] = {VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_VIEWPORT};

    VkPipelineDynamicStateCreateInfo dyn_state_create_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = 2,
        .pDynamicStates = dyn_states,
    };

    VkPipelineVertexInputStateCreateInfo vertex_in_create_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .vertexBindingDescriptionCount = 0,
        .pVertexBindingDescriptions = NULL,
        .vertexAttributeDescriptionCount = 0,
        .pVertexAttributeDescriptions = NULL
    };

    VkPipelineInputAssemblyStateCreateInfo input_assembly_state_create_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    };

    VkViewport viewport = {
        .x = 0,
        .y = 0,
        .width = (float)g_ctx->swapchain_extent.width,
        .height = (float)g_ctx->swapchain_extent.height,
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };

    VkRect2D scissor = {
        .extent = g_ctx->swapchain_extent,
        .offset = (VkOffset2D){
            .x = 0, .y = 0
        }
    };

    VkPipelineViewportStateCreateInfo viewport_create_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .flags = 0,
        .pNext = NULL,
        .pViewports = &viewport,
        .pScissors = &scissor,
        .viewportCount = 1,
        .scissorCount = 1
    };

    VkPipelineRasterizationStateCreateInfo rasterizer = {
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

    VkPipelineMultisampleStateCreateInfo multisampling = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        .sampleShadingEnable = VK_FALSE,
        .minSampleShading = 1.0f,
        .pSampleMask = NULL,
        .alphaToCoverageEnable = VK_FALSE,
        .alphaToOneEnable = VK_FALSE,
    };

    VkPipelineColorBlendAttachmentState color_blend_attachment = {
        .blendEnable = VK_FALSE,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
        .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
        .colorBlendOp = VK_BLEND_OP_ADD,
        .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
        .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
        .alphaBlendOp = VK_BLEND_OP_ADD,
    };

    VkPipelineColorBlendStateCreateInfo color_blend_state = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .logicOpEnable = VK_FALSE,
        .logicOp = VK_LOGIC_OP_COPY,
        .attachmentCount = 1,
        .pAttachments = &color_blend_attachment,
        .blendConstants = {0.0f, 0.0f, 0.0f, 0.0f},
        .flags = 0,
        .pNext = NULL,
    };

    VkPipelineLayoutCreateInfo layout_create_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .setLayoutCount = 0,
        .pushConstantRangeCount = 0,
        .pSetLayouts = NULL,
        .pPushConstantRanges = NULL,
    };

    if(vkCreatePipelineLayout(g_ctx->device, &layout_create_info, VK_NULL_HANDLE, &g_ctx->pipeline_layout) != VK_SUCCESS) {
        SDL_Log("Failed to create pipeline layout!");
        return -1;
    }

    VkPipelineRenderingCreateInfo pipeline_rendering_create_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .pNext = NULL,
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &g_ctx->swapchain_image_format,
        .depthAttachmentFormat = VK_FORMAT_UNDEFINED,
        .stencilAttachmentFormat = VK_FORMAT_UNDEFINED,
    };

    VkGraphicsPipelineCreateInfo pipeline_create_info = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &pipeline_rendering_create_info,
        .flags = 0,
        .stageCount = 2,
        .pStages = shaderStages,
        .pVertexInputState = &vertex_in_create_info,
        .pInputAssemblyState = &input_assembly_state_create_info,
        .pViewportState = &viewport_create_info,
        .pRasterizationState = &rasterizer,
        .pMultisampleState = &multisampling,
        .pColorBlendState = &color_blend_state,
        .pDynamicState = &dyn_state_create_info,
        .layout = g_ctx->pipeline_layout,
        .renderPass = NULL,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = -1,
        .subpass = 0,
    };

    if(vkCreateGraphicsPipelines(g_ctx->device, NULL, 1, &pipeline_create_info, NULL, &g_ctx->graphicsPipeline) != VK_SUCCESS) {
        SDL_Log("Failed to create graphics pipeline!");
        return -1;
    }

    return 0;
}

int createCommandPool(GraphicsContext* g_ctx) {
    VkCommandPoolCreateInfo command_pool_create = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = NULL,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = g_ctx->q_graphics_family_idx,
    };

    if(vkCreateCommandPool(g_ctx->device, &command_pool_create, NULL, &g_ctx->commandPool) != VK_SUCCESS) {
        SDL_Log("Failed create command pool!");
        return -1;
    }

    return 0;
}

int createCommandBuffers(GraphicsContext* g_ctx) {
    VkCommandBufferAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = NULL,
        .commandPool = g_ctx->commandPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = MAX_FRAMES_IN_FLIGHT,
    };

    if(vkAllocateCommandBuffers(g_ctx->device, &alloc_info, &g_ctx->commandBuffer) != VK_SUCCESS) {
        SDL_Log("Failed allocate command buffer!");
        return -1;
    }
    
    return 0;
}

int transition_image_layout(
    GraphicsContext* g_ctx,
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
        .image = g_ctx->swapchain_images[imageIndex],
        .subresourceRange = (VkImageSubresourceRange) {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        }
    };

    VkDependencyInfo dep_info = {
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .pNext = NULL,
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &barrier,
    };

    vkCmdPipelineBarrier2(g_ctx->commandBuffer[g_ctx->currentFrameIndex], &dep_info);

    return 0;
}

int recordCommandBuffer(GraphicsContext* g_ctx, uint32_t imageIndex) {
    VkCommandBuffer commandBuffer = g_ctx->commandBuffer[g_ctx->currentFrameIndex];

    VkCommandBufferBeginInfo buffer_begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    };

    if(vkBeginCommandBuffer(commandBuffer, &buffer_begin_info) != VK_SUCCESS) {
        SDL_Log("Failed begin!");
        return -1;
    }

    transition_image_layout(
        g_ctx,
        imageIndex,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_ACCESS_2_NONE,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT
    );

    VkClearColorValue clear_value = {
        .float32 = {0.0f, 0.0f, 0.0f, 1.0f},
    };

    VkRenderingAttachmentInfo attachment_info = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = g_ctx->image_views[imageIndex],
        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue = clear_value,
    };

    VkRenderingInfo rendering_info = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea = (VkRect2D) {
            .offset = (VkOffset2D){
                .x = 0,
                .y = 0
            },
            .extent = g_ctx->swapchain_extent
        },
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &attachment_info,
    };

    vkCmdBeginRendering(commandBuffer, &rendering_info);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_ctx->graphicsPipeline);

    VkViewport viewport = {
        .x = 0,
        .y = 0,
        .width = (float)g_ctx->swapchain_extent.width,
        .height = (float)g_ctx->swapchain_extent.height,
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
        .extent = g_ctx->swapchain_extent,
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

    transition_image_layout(
        g_ctx,
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

int createSyncObjects(GraphicsContext* g_ctx) {
    for (int i = 0; i < g_ctx->swapchain_images_count; i++) 
    {
        VkSemaphoreCreateInfo semaphore_create_info = {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            .pNext = NULL,
            .flags = 0,
        };
        vkCreateSemaphore(g_ctx->device, &semaphore_create_info, NULL, &g_ctx->renderFinishedSemaphore[i]);
    }
    
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) 
    {
        VkSemaphoreCreateInfo semaphore_create_info = {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            .pNext = NULL,
            .flags = 0,
        };
        vkCreateSemaphore(g_ctx->device, &semaphore_create_info, NULL, &g_ctx->presentCompleteSemaphore[i]);

        VkFenceCreateInfo fence_create_info = {
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .flags = VK_FENCE_CREATE_SIGNALED_BIT,
            .pNext = NULL,
        };
        vkCreateFence(g_ctx->device, &fence_create_info, NULL, &g_ctx->drawFence[i]);
    }

    return 0;
}

void drawFrame(GraphicsContext* g_ctx) {
    while (VK_TIMEOUT == vkWaitForFences(g_ctx->device, 1, &g_ctx->drawFence[g_ctx->currentFrameIndex], VK_TRUE, UINT64_MAX))
        ;

    uint32_t imageIndex;
    VkResult r = vkAcquireNextImageKHR(g_ctx->device, g_ctx->swapchain, UINT64_MAX, g_ctx->presentCompleteSemaphore[g_ctx->currentFrameIndex], NULL, &imageIndex);
    if(r == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapChain(g_ctx);
        return;
    }else if(r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) {
        SDL_Log("Failed aquire next image!");
        return;
    }

    vkResetFences(g_ctx->device, 1, &g_ctx->drawFence[g_ctx->currentFrameIndex]);

    VkCommandBufferResetFlags reset_flags = 0;
    vkResetCommandBuffer(g_ctx->commandBuffer[g_ctx->currentFrameIndex], reset_flags);
    recordCommandBuffer(g_ctx, imageIndex);

    VkPipelineStageFlags wait_dst_stage_mask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pWaitSemaphores = &g_ctx->presentCompleteSemaphore[g_ctx->currentFrameIndex],
        .waitSemaphoreCount = 1,
        .pWaitDstStageMask = &wait_dst_stage_mask,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &g_ctx->renderFinishedSemaphore[imageIndex],
        .pCommandBuffers = &g_ctx->commandBuffer[g_ctx->currentFrameIndex],
        .commandBufferCount = 1,
    };

    vkQueueSubmit(g_ctx->queue, 1, &submit_info, g_ctx->drawFence[g_ctx->currentFrameIndex]);

    VkPresentInfoKHR present_info = {
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .pWaitSemaphores = &g_ctx->renderFinishedSemaphore[imageIndex],
        .waitSemaphoreCount = 1,
        .pSwapchains = &g_ctx->swapchain,
        .swapchainCount = 1,
        .pImageIndices = &imageIndex,
        .pResults = NULL,
    };

    VkResult present_result = vkQueuePresentKHR(g_ctx->queue, &present_info);

    if(present_result == VK_ERROR_OUT_OF_DATE_KHR || g_ctx->framebufferResized) {
        g_ctx->framebufferResized = VK_FALSE;
        recreateSwapChain(g_ctx);
    } else if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) {
        SDL_Log("Failed aquire next image!");
        return;
    }

    g_ctx->currentFrameIndex = (g_ctx->currentFrameIndex + 1) % MAX_FRAMES_IN_FLIGHT;
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

int init_vulkan(GraphicsContext* g_ctx) {
    VkApplicationInfo app_info;

    app_info.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO;
	app_info.pNext=NULL;
	char* app_name = "Hello Vulkan";
	app_info.pApplicationName=app_name;
	app_info.applicationVersion=VK_MAKE_VERSION(0,0,1);
	char* app_engine_name = "No engine";
	app_info.pEngineName=app_engine_name;
	app_info.engineVersion=VK_MAKE_VERSION(0,0,1);
	app_info.apiVersion=VK_API_VERSION_1_3;

    VkInstanceCreateInfo instance_create_info;

    SDL_Log("Trying to create info!");

	instance_create_info.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	instance_create_info.pNext=NULL;
	instance_create_info.flags=0;
	instance_create_info.pApplicationInfo=&app_info;
	instance_create_info.enabledExtensionCount=g_ctx->instance_extensions_count;
    instance_create_info.ppEnabledExtensionNames=g_ctx->instance_extensions;

    #define LAYER_COUNT 1
    
    instance_create_info.enabledLayerCount=LAYER_COUNT;
    const char* pp_inst_layers[LAYER_COUNT] = {
        "VK_LAYER_KHRONOS_validation"
    };
    instance_create_info.ppEnabledLayerNames = pp_inst_layers;

    #undef LAYER_COUNT

    uint32_t layer_props_count;
    vkEnumerateInstanceLayerProperties(&layer_props_count, NULL);

    VkLayerProperties* props = calloc(layer_props_count, sizeof(VkLayerProperties));
    vkEnumerateInstanceLayerProperties(&layer_props_count, props);

    for(int i = 0; i < layer_props_count; i++) {
        SDL_Log("Has Layer: %s", props[i].layerName);
    }

    if (vkCreateInstance(&instance_create_info, NULL, &g_ctx->inst) != VK_SUCCESS) {
        SDL_Log("Failed to create Vulkan instance!");
    } else {
        SDL_Log("Vulkan instance created!");
    }

    VkDebugUtilsMessengerCreateInfoEXT debug_create_info = {
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT,
        .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
        .pfnUserCallback = debugCallback,
    };

    PFN_vkCreateDebugUtilsMessengerEXT vkCreateDebugUtilsMessengerEXT = 
    (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
        g_ctx->inst, "vkCreateDebugUtilsMessengerEXT");

    vkCreateDebugUtilsMessengerEXT(g_ctx->inst, &debug_create_info, NULL, &g_ctx->debugMessanger);

    if(create_surface(g_ctx)) {
        return -1;
    }

    uint32_t device_count = 0;
	vkEnumeratePhysicalDevices(g_ctx->inst,&device_count,NULL);

    SDL_Log("Available devices count: %i", device_count);

    VkPhysicalDevice vk_devices[16];
	vkEnumeratePhysicalDevices(g_ctx->inst,&device_count,vk_devices);
    VkPhysicalDevice* device = NULL;

    for (int i = 0; i < device_count; i++) {
        if (is_device_suitable(g_ctx, vk_devices[i])) {
            device = &vk_devices[i];
            break;
        }
    }

    if (device == NULL) {
        SDL_Log("Couldn't find suitable device!");
        return -1;
    }

    g_ctx->physicalDevice = *device;

    VkPhysicalDeviceProperties device_props;
    vkGetPhysicalDeviceProperties(*device, &device_props);

    SDL_Log("Picked device: %s", device_props.deviceName);

    if(create_logical_device(g_ctx, *device)){
        SDL_Log("Failed to create logical device!");
        return -1;
    }

    if(create_swap_chain(g_ctx, *device)) {
        SDL_Log("Failed to create swapchain!");
        return -1;
    }

    SDL_Log("Swapchain created!");

    if(create_image_views(g_ctx)){
        SDL_Log("Failed to create image views!");
        return -1;
    }

    SDL_Log("Image views created!");

    if(create_graphics_pipeline(g_ctx)){
        SDL_Log("Failed create graphics pipeline!");
        return -1;
    }

    SDL_Log("Graphics pipeline created!");

    if(createCommandPool(g_ctx)) {
        return -1;
    }

    SDL_Log("Command pool created!");

    if(createCommandBuffers(g_ctx)) {
        return -1;
    }

    SDL_Log("Allocated command buffer!");

    if(createSyncObjects(g_ctx)) {
        return -1;
    }

    SDL_Log("Sync objects created!");

    return 0;
}

void cleanup(GraphicsContext* g_ctx) {
    vkDeviceWaitIdle(g_ctx->device);
}
