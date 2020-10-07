#define _CRT_SECURE_NO_WARNINGS

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>


#include <imgui.h>
#include "examples/imgui_impl_glfw.h"
#include "examples/imgui_impl_vulkan.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <functional>
#include <cstdlib>
#include <cstdint> // UINT32_MAX

#include <algorithm>
#include <array>
#include <vector>
#include <optional>
#include <set>
#include <tuple>
#include <thread>

#include <filesystem> // hot reloading

#include "camera.hpp"

const int MAX_FRAMES_IN_FLIGHT = 2;

struct DrawData
{
    uint32_t materialIndex; // Index into material buffer
    uint32_t transformIndex; // Index into transform buffer
    uint32_t vertexOffset; // used to lookup attributes in vertex storage buffer
    uint32_t unused; // vec4 padding

    // Gameplay data?
};

struct MaterialData
{
    glm::vec4 albedoTint;

    float tilingX;
    float tilingY;
    float reflectance;
    float unused0; // pad to vec4

    uint32_t albedoTexture;
    uint32_t normalTexture;
    uint32_t roughnessTexture;
    uint32_t unused01; // pad to vec4
};

struct TransformData
{
    glm::mat4 model;
};

struct Vertex 
{
    glm::vec3 pos;
    float unused0;

    glm::vec3 normal;
    float unused01;

    glm::vec3 color;
    float unused02;

    glm::vec2 texCoord;
    float unused03;
    float unused04;

    glm::vec3 tangent;
    float unused05;

    bool operator==(const Vertex& other) const {
        return pos == other.pos && color == other.color && texCoord == other.texCoord;
    }

    static VkVertexInputBindingDescription getBindingDescription() 
    {
        VkVertexInputBindingDescription bindingDescription = {};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 4> getAttributeDescriptions()
    {
        std::array<VkVertexInputAttributeDescription, 4> attributeDescriptions = {};
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT; // vec3
        attributeDescriptions[0].offset = offsetof(Vertex, pos);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT; // vec3
        attributeDescriptions[1].offset = offsetof(Vertex, normal);
        
        attributeDescriptions[2].binding = 0;
        attributeDescriptions[2].location = 2;
        attributeDescriptions[2].format = VK_FORMAT_R32G32B32_SFLOAT; // vec3
        attributeDescriptions[2].offset = offsetof(Vertex, color);

        attributeDescriptions[3].binding = 0;
        attributeDescriptions[3].location = 3;
        attributeDescriptions[3].format = VK_FORMAT_R32G32_SFLOAT; // vec2
        attributeDescriptions[3].offset = offsetof(Vertex, texCoord);
        
        return attributeDescriptions;
    }

};

const uint32_t DrawDataCount = 2048, MatDataCount = 2048, TransformDataCount = 2048;

const std::vector<Vertex> triVertex = {
    { glm::vec3(-1.0f, -1.0f, 0.0), 0.0f, glm::vec3(), 0.0f,  glm::vec3(1.0f, 0.0f, 0.0f), 0.0f, glm::vec2(),  0.0f, 0.0f },
    { glm::vec3( 1.0f, -1.0f, 0.0), 0.0f, glm::vec3(), 0.0f,  glm::vec3(1.0f, 0.0f, 0.0f), 0.0f, glm::vec2(),  0.0f, 0.0f },
    { glm::vec3(-1.0f,  1.0f, 0.0), 0.0f, glm::vec3(), 0.0f,  glm::vec3(1.0f, 0.0f, 0.0f), 0.0f, glm::vec2(),  0.0f, 0.0f },
};

namespace std {
    template<> struct hash<Vertex> {
        size_t operator()(Vertex const& vertex) const {
            return ((hash<glm::vec3>()(vertex.pos) ^
                (hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^
                (hash<glm::vec2>()(vertex.texCoord) << 1);
        }
    };
}
struct UniformBufferObject {
    glm::mat4 view;
    glm::mat4 proj;

    float time;
    glm::vec3 unused0;

    glm::vec3 cameraPosition;
    float unused01;

    glm::vec3 lightPos;
    float unused02;
};

std::vector<Vertex> vertices;
std::vector<uint32_t> indices;
std::vector<MaterialData> matData;
std::vector<TransformData> transformData;
std::vector<DrawData> drawData;

std::vector<void*> vertexDataPointers, materialDataPointers, transformDataPointers, drawDataPointers;

std::vector<VkDrawIndexedIndirectCommand> indirectCommands;
std::vector<VkBuffer> indirectCommandsBuffer;
std::vector<VkDeviceMemory> indirectCommandsBufferMemory;

struct Texture
{
    VkImage image;
    VkDeviceMemory imageMemory;
    VkImageView imageView;
};

std::vector<Texture> textures;

VkBuffer vertexBuffer;
VkDeviceMemory vertexBufferMemory;

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete()
    {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

struct SwapChainSupportDetails
{
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

static std::vector<char> readFile(const std::string& filename)
{
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open())
    {
        throw std::runtime_error("failed to open file!");
    }
    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();

    return buffer;
}

class HelloTriangleApplication {
public:
    void run() 
    {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    const int WIDTH = 1920;//800;
    const int HEIGHT = 1080;//600;
    const std::string MODEL_PATH ="models/sponza/sponza.obj"; //"models/viking_room.obj";
    const std::string TEXTURE_PATH = "textures/viking_room.png";
    const std::string SHADER_PATH = "shaders/";
    std::unordered_map<std::string, std::filesystem::file_time_type> shaderPaths;

    GLFWwindow* window;
    VkInstance instance;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;

    const std::vector<const char*> validationLayers = {
        "VK_LAYER_KHRONOS_validation",
        //"VK_LAYER_RENDERDOC_Capture"
    };

    const std::vector<const char*> deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_KHR_DRAW_INDIRECT_COUNT_EXTENSION_NAME,
        VK_KHR_SHADER_DRAW_PARAMETERS_EXTENSION_NAME,
        VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME
    };

#ifdef NDEBUG
    const bool enableValidationLayers = false;
#else
    const bool enableValidationLayers = true;
#endif
    

    VkDevice device;
    QueueFamilyIndices  queueFamilyIndices;
    VkQueue graphicsQueue, presentQueue;
    VkSurfaceKHR surface;
    VkSwapchainKHR swapChain;

    std::vector<VkImage> swapChainImages;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;

    VkQueryPool queryPoolTimestamp;
    float timestampPeriod;
    VkRenderPass renderPass;

    VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;

    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> descriptorSets;
    // second pool for IMGUI
    VkDescriptorPool pool;

    VkPipelineLayout pipelineLayout;
    
    VkPipeline graphicsPipeline;

    std::vector<VkImageView> swapChainImageViews;

    std::vector<VkFramebuffer> swapChainFramebuffers;

    std::vector<VkCommandPool> commandPools;
    std::vector<VkCommandBuffer> commandBuffers;

    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    std::vector<VkFence> imagesInFlight;

    size_t currentFrame = 0;

    bool framebufferResized = false;

    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    VkBuffer indexBuffer;
    VkDeviceMemory indexBufferMemory;

    VkImage textureImage;
    VkDeviceMemory textureImageMemory;

    VkImageView textureImageView;
    VkSampler textureSampler;

    VkImage depthImage;
    VkDeviceMemory depthImageMemory;
    VkImageView depthImageView;

    // msaa
    VkImage colorImage;
    VkDeviceMemory colorImageMemory;
    VkImageView colorImageView;

    // One per swap chain image
    std::vector<VkBuffer> uniformBuffers, matBuffers, transformBuffers, drawBuffers, vertexBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory, matBuffersMemory, transformBuffersMemory, drawBuffersMemory, vertexBuffersMemory;

    Camera camera;
    glm::mat4 view;

    std::chrono::high_resolution_clock::time_point previousTime = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::time_point lastTimeStamp = previousTime;
    double prevx, prevy;

    ImDrawData* imguiDrawData;

    glm::vec3 lightPos, lightSpeed;
    struct CameraDebugData
    {
        glm::vec3 forward;
        glm::vec3 right;
        glm::vec3 top;
    };

    std::array<float, 100> frameTimes;
    int frameIndex = 0;
    float averageFrameTime = 0.0f;

    int frames = 0;
    float culmativeDelta = 0.0f;
    double fps = 0.0;

    double frameGpuAvg = 0.0;
    double frameCpuAvg = 0.0;

    void initWindow() 
    {
        glfwInit();
        
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        
        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwGetCursorPos(window, &prevx, &prevy);
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
        glfwSetScrollCallback(window, scroll_callback);

    }

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height)
    {
        auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }
    
    static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
    {
        auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
        app->camera.rotate(glm::vec3(0.f, (float)yoffset, 0.f));
    }
    
    void initVulkan() 
    {
        createInstance();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createCommandPool();
        createQueryPool();
        loadModel();
        createImageViews();
        createRenderPass();
        createDescriptorSetLayout();
        createGraphicsPipeline();
        createColorResources();
        createDepthResources();
        createFramebuffers();
        createTextImageView();
        createTextureSampler();
        createVertexBuffer();
        createIndexBuffer();
        createUniformBuffers();
        createBindlessBuffers();
        updateIndirectBuffer();
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffers();
        createSyncObjects();
        //createIMGUI();
        initGame();
    }

    void recreateSwapChain()
    {
        // Handle minimized window, which has a framebuffer size of 0
        // we just pause until window in the foreground again
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0)
        {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }
        vkDeviceWaitIdle(device); // don't touch resources that may still be in use
        cleanupSwapChain();
        
        createSwapChain();
        createImageViews();
        createRenderPass();
        createGraphicsPipeline();
        createDepthResources();
        createFramebuffers();
        createUniformBuffers();
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffers();
    }

    void mainLoop() 
    {
        while (!glfwWindowShouldClose(window))
        {
            double frameCpuBegin = glfwGetTime() * 1000;
            glfwPollEvents();
            update();
            drawFrame();
            // update title with performance
            double frameCpuEnd = glfwGetTime() * 1000;
            frameCpuAvg = frameCpuAvg * 0.95 + (frameCpuEnd - frameCpuBegin) * 0.05;
            char title[256];

            sprintf(title, "DanVulkan cpu %.2f ms; gpu: %.2f ms", frameCpuAvg, frameGpuAvg);
            glfwSetWindowTitle(window, title);

        }

        vkDeviceWaitIdle(device);
    }

    void cleanupSwapChain()
    {
        vkDestroyImageView(device, colorImageView, nullptr);
        vkDestroyImage(device, colorImage, nullptr);
        vkFreeMemory(device, colorImageMemory, nullptr);

        vkDestroyImageView(device, depthImageView, nullptr);
        vkDestroyImage(device, depthImage, nullptr);
        vkFreeMemory(device, depthImageMemory, nullptr);

        for (auto framebuffer : swapChainFramebuffers)
        {
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }
        for (size_t i = 0; i <= MAX_FRAMES_IN_FLIGHT; i++)
        {
            vkFreeCommandBuffers(device, commandPools[i], 1, &commandBuffers[i]);
        }
        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyRenderPass(device, renderPass, nullptr);
        for (auto imageView : swapChainImageViews)
        {
            vkDestroyImageView(device, imageView, nullptr);
        }

        vkDestroySwapchainKHR(device, swapChain, nullptr);

        for (size_t i = 0; i < swapChainImages.size(); i++)
        {
            vkDestroyBuffer(device, matBuffers[i], nullptr);
            vkDestroyBuffer(device, drawBuffers[i], nullptr);
            vkDestroyBuffer(device, transformBuffers[i], nullptr);
            vkDestroyBuffer(device, vertexBuffers[i], nullptr);
            vkDestroyBuffer(device, uniformBuffers[i], nullptr);
            vkDestroyBuffer(device, indirectCommandsBuffer[i], nullptr);

            vkFreeMemory(device, matBuffersMemory[i], nullptr);
            vkFreeMemory(device, drawBuffersMemory[i], nullptr);
            vkFreeMemory(device, transformBuffersMemory[i], nullptr);
            vkFreeMemory(device, vertexBuffersMemory[i], nullptr);
            vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
            vkFreeMemory(device, indirectCommandsBufferMemory[i], nullptr);


        }
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    }
    void cleanup() 
    {
        vkDestroyQueryPool(device, queryPoolTimestamp, nullptr);
        vkDestroyDescriptorPool(device, pool, nullptr); //imgui
        /*
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();*/
        cleanupSwapChain();

        vkDestroySampler(device, textureSampler, nullptr);
        for (Texture t : textures)
        {
            vkDestroyImage(device, t.image, nullptr);
            vkFreeMemory(device, t.imageMemory, nullptr);
            vkDestroyImageView(device, t.imageView, nullptr);
        }


        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        vkDestroyBuffer(device, indexBuffer, nullptr);
        vkFreeMemory(device, indexBufferMemory, nullptr);
        
        
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
            vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
            vkDestroyFence(device, inFlightFences[i], nullptr);
            vkDestroyCommandPool(device, commandPools[i], nullptr);
        }
        // CLEANUP: for some reason I'm allocating 3 commandPools, ImageIndex = 2 sometimes whereas its always i < 2 (max frames in flight)
        vkDestroyCommandPool(device, commandPools[MAX_FRAMES_IN_FLIGHT], nullptr);
       
      
        vkDestroyDevice(device, nullptr);
        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);

        glfwDestroyWindow(window);
        glfwTerminate();
    }
    
    void recreateGraphicsPipeline()
    {
        vkDeviceWaitIdle(device); // don't touch resources that may still be in use
        for (size_t i = 0; i <= MAX_FRAMES_IN_FLIGHT; i++)
        {
            vkFreeCommandBuffers(device, commandPools[i], 1, &commandBuffers[i]);
        }        
        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);

        createGraphicsPipeline();
        createColorResources();
        createDepthResources();
        createCommandBuffers();

    }

    void createInstance()
    {
        if (enableValidationLayers && !checkValidationLayerSupport()) {
            throw std::runtime_error("validation layers requested but not avaiable!");
        }
        VkApplicationInfo appInfo = {};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Hello Triangle";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

        VkInstanceCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        createInfo.enabledExtensionCount = glfwExtensionCount;
        createInfo.ppEnabledExtensionNames = glfwExtensions;

        if (enableValidationLayers)
        {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }
        else
        {
            createInfo.enabledLayerCount = 0;
        }

        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("failed to create vulkan instance");
        }
    }

    bool checkValidationLayerSupport()
    {
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        for (const char* layerName : validationLayers)
        {
            bool layerFound = false;

            for (const auto& layerProperties : availableLayers)
            {
                if (strcmp(layerName, layerProperties.layerName) == 0)
                {
                    layerFound = true;
                    break;
                }
            }
            if (!layerFound)
                return false;
        }
        return true;
    }

    void pickPhysicalDevice()
    {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

        if (deviceCount == 0)
        {
            throw std::runtime_error("failed to find GPUs with vulkan support!");
        }
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        for (const auto& device : devices)
        {
            if (isDeviceSuitable(device))
            {
                physicalDevice = device;
                msaaSamples = getMaxUsableSampleCount();
                break;
            }
        }
        if (physicalDevice == VK_NULL_HANDLE)
        {
            throw std::runtime_error("failed to find a suitable GPU!");
        }
        
        // check push constant size, we can only assume 128 on every device though
        VkPhysicalDeviceProperties pdp;
        vkGetPhysicalDeviceProperties(physicalDevice, &pdp);
        timestampPeriod = pdp.limits.timestampPeriod;
        //std::cout << "Push constant max size: " << pdp.limits.maxPushConstantsSize << std::endl;
    }

    bool isDeviceSuitable(VkPhysicalDevice device)
    {
        QueueFamilyIndices indices = findQueueFamilies(device);
        
        bool extensionsSupported = checkDeviceExtensionSupport(device);

        bool swapChainAdequate = false;
        if (extensionsSupported)
        {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
            swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }

        VkPhysicalDeviceFeatures supportedFeatures;
        vkGetPhysicalDeviceFeatures(device, &supportedFeatures);
        return indices.graphicsFamily.has_value() && extensionsSupported && swapChainAdequate &&
            supportedFeatures.samplerAnisotropy;
    }

    bool checkDeviceExtensionSupport(VkPhysicalDevice device)
    {
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> availbleExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availbleExtensions.data());

        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

        for (const auto& extension : availbleExtensions)
        {
            requiredExtensions.erase(extension.extensionName);
        }
        return requiredExtensions.empty();
    }
    
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device)
    {
        QueueFamilyIndices indices;

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

        int i = 0;
        for (const auto& queueFamily : queueFamilies)
        {

            if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)
            {
                indices.graphicsFamily = i;
            }

            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

            if (presentSupport)
            {
                indices.presentFamily = i;
            }
            if (indices.isComplete())
            {
                break;
            }

            i++;
        }
        return indices;
    }
    
    VkSampleCountFlagBits getMaxUsableSampleCount() {
        VkPhysicalDeviceProperties physicalDeviceProperties;
        vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);

        VkSampleCountFlags counts = physicalDeviceProperties.limits.framebufferColorSampleCounts & physicalDeviceProperties.limits.framebufferDepthSampleCounts;
        if (counts & VK_SAMPLE_COUNT_64_BIT) { return VK_SAMPLE_COUNT_64_BIT; }
        if (counts & VK_SAMPLE_COUNT_32_BIT) { return VK_SAMPLE_COUNT_32_BIT; }
        if (counts & VK_SAMPLE_COUNT_16_BIT) { return VK_SAMPLE_COUNT_16_BIT; }
        if (counts & VK_SAMPLE_COUNT_8_BIT) { return VK_SAMPLE_COUNT_8_BIT; }
        if (counts & VK_SAMPLE_COUNT_4_BIT) { return VK_SAMPLE_COUNT_4_BIT; }
        if (counts & VK_SAMPLE_COUNT_2_BIT) { return VK_SAMPLE_COUNT_2_BIT; }

        return VK_SAMPLE_COUNT_1_BIT;
    }

    void createLogicalDevice()
    {
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() };
        
        float queuePriority = 1.0f;
        for (uint32_t queueFamily : uniqueQueueFamilies)
        {
            VkDeviceQueueCreateInfo queueCreateInfo = {};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }
       
        VkPhysicalDeviceFeatures deviceFeatures = {};
        deviceFeatures.samplerAnisotropy = VK_TRUE;
        deviceFeatures.sampleRateShading = VK_TRUE;
        deviceFeatures.multiDrawIndirect = VK_TRUE;

        VkPhysicalDeviceDescriptorIndexingFeaturesEXT pdIndexing = {};
        pdIndexing.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES_EXT;
        pdIndexing.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
        pdIndexing.runtimeDescriptorArray = VK_TRUE;
        pdIndexing.descriptorBindingVariableDescriptorCount = VK_TRUE;

        VkDeviceCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();

        createInfo.pEnabledFeatures = &deviceFeatures;
        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();
        createInfo.enabledLayerCount = 0;
        createInfo.pNext = &pdIndexing;

        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create logical device");
        }
        vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
        queueFamilyIndices = indices;
    }

    void createSurface()
    {
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create window surface");
        }
    }

    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device)
    {
        SwapChainSupportDetails details;

        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

        if (formatCount != 0)
        {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
        }

        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

        if (presentModeCount != 0)
        {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &formatCount, details.presentModes.data());
        }

        return details;
    }

    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats)
    {
        for (const auto& availableFormat : availableFormats)
        {
            if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            {
                return availableFormat;
            }
        }
        // just fallback to first available if preffered one isn't there
        return availableFormats[0];
    }

    VkPresentModeKHR chooseSwapPresentmode(const std::vector<VkPresentModeKHR>& availablePresentModes)
    {
        // no vysnc
        return VK_PRESENT_MODE_IMMEDIATE_KHR;
        for (const auto& availablePresentMode : availablePresentModes)
        {
            // Look for triple buffering
            // NOTE: this can be inefficent as CPU + GPU are always working
            if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR)
            {
                return availablePresentMode;
            }
        }
        // FIFO guranteed to be available
        return VK_PRESENT_MODE_FIFO_KHR;
    }

    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities)
    {
        if (capabilities.currentExtent.width != UINT32_MAX)
        {
            return capabilities.currentExtent;
        }
        else 
        {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);
           
            VkExtent2D actualExtent = { 
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height)
            };

            actualExtent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actualExtent.width));
            actualExtent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actualExtent.height));
            
            return actualExtent;
        }
    }

    void createSwapChain()
    {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        VkPresentModeKHR presentMode = chooseSwapPresentmode(swapChainSupport.presentModes);
        VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

        // recommended to request at least one more image then min
        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
        if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)
        {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }

        VkSwapchainCreateInfoKHR createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };
        if (indices.graphicsFamily != indices.presentFamily)
        {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        }
        else
        {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
            createInfo.queueFamilyIndexCount = 0; // optional ??
            createInfo.pQueueFamilyIndices = nullptr; // optional ??
        }
        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;
        createInfo.oldSwapchain = VK_NULL_HANDLE;

        if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create swap chain");
        }

        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;
    }
#pragma region stuff
    VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags)
    {
        VkImageViewCreateInfo createInfo = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        createInfo.image = image;
        createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        createInfo.format = format;

        createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

        createInfo.subresourceRange.aspectMask = aspectFlags;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = 1;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = 1;

        VkImageView imageView;
        if (vkCreateImageView(device, &createInfo, nullptr, &imageView) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create image views");
        }
        return imageView;
    }

    void createImageViews()
    {
        swapChainImageViews.resize(swapChainImages.size());

        for (size_t i = 0; i < swapChainImages.size(); i++)
        {
            swapChainImageViews[i] = createImageView(swapChainImages[i], swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT);
        }

    }
    
  
    
    void createGraphicsPipeline()
    {
        auto [vertCreateInfo, vertShader] = loadShader("shaders/vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
        auto  [fragCreateInfo, fragShader] = loadShader("shaders/frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
        VkPipelineShaderStageCreateInfo shaderStages[] = { vertCreateInfo, fragCreateInfo };

        VkPipelineVertexInputStateCreateInfo vertexInputInfo = { VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
        vertexInputInfo.vertexBindingDescriptionCount = 0; // 1
        vertexInputInfo.pVertexBindingDescriptions = nullptr;// &bindingDescription;
        vertexInputInfo.vertexAttributeDescriptionCount = 0;//   static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexAttributeDescriptions = nullptr; // attributeDescriptions.data();

        VkPipelineInputAssemblyStateCreateInfo inputAssembly = { VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        VkViewport viewport = {};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float)swapChainExtent.width;
        viewport.height = (float)swapChainExtent.height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        VkRect2D scissor = {};
        scissor.offset = { 0, 0 };
        scissor.extent = swapChainExtent;

        VkPipelineViewportStateCreateInfo viewportState = { VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
        viewportState.viewportCount = 1;
        viewportState.pViewports = &viewport;
        viewportState.scissorCount = 1;
        viewportState.pScissors = &scissor;

        VkPipelineRasterizationStateCreateInfo rasterizer = { VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_NONE;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;
        rasterizer.depthBiasConstantFactor = 0.0f;
        rasterizer.depthBiasClamp = 0.0f;
        rasterizer.depthBiasSlopeFactor = 0.0f;

        VkPipelineMultisampleStateCreateInfo multisampling = { VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
        multisampling.sampleShadingEnable = VK_TRUE;
        multisampling.rasterizationSamples = msaaSamples;
        multisampling.minSampleShading = 0.2f;
        multisampling.pSampleMask = nullptr;
        multisampling.alphaToCoverageEnable = VK_FALSE;
        multisampling.alphaToOneEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
            VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

        VkPipelineColorBlendStateCreateInfo colorBlending = { VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f;
        colorBlending.blendConstants[1] = 0.0f;
        colorBlending.blendConstants[2] = 0.0f;
        colorBlending.blendConstants[3] = 0.0f;

        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
        depthStencil.depthBoundsTestEnable = VK_FALSE;
        depthStencil.minDepthBounds = 0.0f;
        depthStencil.maxDepthBounds = 1.0f;
        depthStencil.stencilTestEnable = VK_FALSE;
        // Dynamic State goes hereeeeeee

        VkPipelineLayoutCreateInfo pipelineLayoutInfo = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 0;
        pipelineLayoutInfo.pPushConstantRanges = nullptr;

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create pipeline layout");
        }

        VkGraphicsPipelineCreateInfo pipelineInfo = { VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = nullptr; // optional
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = nullptr; // optional
        pipelineInfo.layout = pipelineLayout;
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0; // index of the sub pass where this pipeline will be used
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // optional
        pipelineInfo.basePipelineIndex = -1; // optional
        pipelineInfo.pDepthStencilState = &depthStencil;

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create graphics pipeline");
        }

        vkDestroyShaderModule(device, fragShader, nullptr);
        vkDestroyShaderModule(device, vertShader, nullptr);
    }

    VkShaderModule createShaderModule(const std::vector<char>& code)
    {
        // We don't delete shaderModule after creating pipeline, this could be bad?
        VkShaderModuleCreateInfo createInfo = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create shader module");
        }

        return shaderModule;
    }

    std::tuple<VkPipelineShaderStageCreateInfo, VkShaderModule> loadShader(const std::string& filename, VkShaderStageFlagBits stage)
    {
        auto code = readFile(filename);
        VkShaderModule shaderModule = createShaderModule(code);

        VkPipelineShaderStageCreateInfo shaderStageCreateInfo = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
        shaderStageCreateInfo.stage = stage;
        shaderStageCreateInfo.module = shaderModule;
        shaderStageCreateInfo.pName = "main";

        return { shaderStageCreateInfo, shaderModule };
    }

    void createRenderPass()
    {
        VkAttachmentDescription colorAttachment = {};
        colorAttachment.format = swapChainImageFormat;
        colorAttachment.samples = msaaSamples;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentDescription depthAttachment{};
        depthAttachment.format = findDepthFormat();
        depthAttachment.samples = msaaSamples;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentDescription colorAttachmentResolve{};
        colorAttachmentResolve.format = swapChainImageFormat;
        colorAttachmentResolve.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachmentResolve.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachmentResolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachmentResolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachmentResolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachmentResolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachmentResolve.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference colorAttachmentRef = {};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentReference depthAttachmentRef{};
        depthAttachmentRef.attachment = 1;
        depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference colorAttachmentResolveRef = {};
        colorAttachmentResolveRef.attachment = 2;
        colorAttachmentResolveRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass = {};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1; // Same as our Fragment shader: layout outColor
        subpass.pColorAttachments = &colorAttachmentRef;
        subpass.pDepthStencilAttachment = &depthAttachmentRef;
        subpass.pResolveAttachments = &colorAttachmentResolveRef;

        VkSubpassDependency dependency = {};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;


        std::array<VkAttachmentDescription, 3> attachments = { colorAttachment, depthAttachment, colorAttachmentResolve };
        VkRenderPassCreateInfo renderPassInfo = { VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO };
        renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        renderPassInfo.pAttachments = attachments.data();
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create render pass!");
        }

    }

    void createFramebuffers()
    {
        swapChainFramebuffers.resize(swapChainImageViews.size());

        for (size_t i = 0; i < swapChainImageViews.size(); i++)
        {
            std::array<VkImageView, 3 > attachments = {
                colorImageView,
                depthImageView,
                swapChainImageViews[i]
            };

            VkFramebufferCreateInfo framebufferInfo = { VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO };
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
            framebufferInfo.pAttachments = attachments.data();
            framebufferInfo.width = swapChainExtent.width;
            framebufferInfo.height = swapChainExtent.height;
            framebufferInfo.layers = 1;

            if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS)
            {
                throw std::runtime_error("failed to create framebuffer");
            }

        }
    }

    void createCommandPool()
    {
        commandPools.resize(MAX_FRAMES_IN_FLIGHT + 1);
        QueueFamilyIndices queueFamilyIndicies = findQueueFamilies(physicalDevice);
        VkCommandPoolCreateInfo poolInfo = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
        poolInfo.queueFamilyIndex = queueFamilyIndicies.graphicsFamily.value();
        poolInfo.flags = 0; // optional
        for (size_t i = 0; i <= MAX_FRAMES_IN_FLIGHT; i++)
        {
            if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPools[i]) != VK_SUCCESS)
            {
                throw std::runtime_error("failed to create command pool");
            }
        }
    }

    void createQueryPool()
    {
        VkQueryPoolCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        createInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        createInfo.queryCount = 2;

        if (vkCreateQueryPool(device, &createInfo, nullptr, &queryPoolTimestamp) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create Query Pool Timestamps");
        }
    }
    
    VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features)
    {
        for (VkFormat format : candidates)
        {
            VkFormatProperties props;
            vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);

            if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features)
            {
                return format;
            }
            else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features)
            {
                return format;
            }
        }
        throw std::runtime_error("failed to find supported format!");
    }

    VkFormat findDepthFormat()
    {
        return findSupportedFormat(
            { VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
            VK_IMAGE_TILING_OPTIMAL,
            VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
        );
    }

    bool hasStencilComponent(VkFormat format)
    {
        return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
    }

    void createColorResources()
    {
        VkFormat colorFormat = swapChainImageFormat;

        createImage(swapChainExtent.width, swapChainExtent.height, msaaSamples, colorFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, colorImage, colorImageMemory);
        colorImageView = createImageView(colorImage, colorFormat, VK_IMAGE_ASPECT_COLOR_BIT);
    }

    void createDepthResources()
    {
        VkFormat depthFormat = findDepthFormat();
        
        createImage(swapChainExtent.width, swapChainExtent.height, msaaSamples, depthFormat, VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage, depthImageMemory);
        
        depthImageView = createImageView(depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);

        transitionImageLayout(depthImage, depthFormat, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    }

    void createTextureImage(const char *filename)
    {
        Texture texture;
        int texWidth, texHeight, texChannels;
        stbi_uc* pixels = stbi_load(filename, &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        VkDeviceSize imageSize = texWidth * texHeight * 4;

        if (!pixels)
        {
            std::cout << stbi_failure_reason() << std::endl;
            throw std::runtime_error("failed to load texture image!");
        }

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
            memcpy(data, pixels, static_cast<size_t>(imageSize));
        vkUnmapMemory(device, stagingBufferMemory);

        stbi_image_free(pixels);

        createImage(texWidth, texHeight, VK_SAMPLE_COUNT_1_BIT, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, texture.image, texture.imageMemory);

        transitionImageLayout(texture.image, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        copyBufferToImage(stagingBuffer, texture.image, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
        transitionImageLayout(texture.image, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
        
        textures.push_back(texture);

    }
    
    void createImage(uint32_t width, uint32_t height, VkSampleCountFlagBits numSamples, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, 
        VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory)
    {
        VkImageCreateInfo imageInfo = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent.width = width;
        imageInfo.extent.height = height;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.format = format;
        imageInfo.tiling = tiling;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = usage;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.samples = numSamples;
        imageInfo.flags = 0; // optional

        if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create image!");
        }

        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(device, image, &memRequirements);

        VkMemoryAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to allocate image memory!");
        }

        vkBindImageMemory(device, image, imageMemory, 0);
    }

    void createTextImageView()
    {
        for (uint32_t i = 0;  i  < textures.size(); i++)
        {
            textures[i].imageView = createImageView(textures[i].image, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT);
        }
    }
    
    void createTextureSampler() 
    {
        VkSamplerCreateInfo samplerInfo = { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;

        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;

        samplerInfo.anisotropyEnable = VK_TRUE;
        samplerInfo.maxAnisotropy = 16;

        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;

        samplerInfo.unnormalizedCoordinates = VK_FALSE;

        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;

        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerInfo.mipLodBias = 0.0f;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = 0.0f;

        if (vkCreateSampler(device, &samplerInfo, nullptr, &textureSampler) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create texture sampler!");
        }

    }

    VkCommandBuffer beginSingleTimeCommands()
    {
        VkCommandBufferAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = commandPools[currentFrame];
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        return commandBuffer;
    }

    void endSingleTimeCommands(VkCommandBuffer commandBuffer)
    {
        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue);

        vkFreeCommandBuffers(device, commandPools[currentFrame], 1, &commandBuffer);
    }

    void createCommandBuffers()
    {
        commandBuffers.resize(MAX_FRAMES_IN_FLIGHT + 1);
        for (size_t i = 0; i <= MAX_FRAMES_IN_FLIGHT; i++)
        {
            VkCommandBufferAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
            allocInfo.commandPool = commandPools[i];
            allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocInfo.commandBufferCount = 1; // TODO this is just a guess

            if (vkAllocateCommandBuffers(device, &allocInfo, &commandBuffers[i]) != VK_SUCCESS)
            {
                throw std::runtime_error("failed to allocate command buffers");
            }
        }

    }

    void updateCommandBuffer(uint32_t imageIndex)
    {
        VkResult result = vkResetCommandPool(device, commandPools[imageIndex], 0);
        if (result != VK_SUCCESS)
        {
            throw std::runtime_error("failed to reset command pool");
        }

        VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        beginInfo.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        if (vkBeginCommandBuffer(commandBuffers[imageIndex], &beginInfo) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to begin recording command buffer");
        }
        
        vkCmdResetQueryPool(commandBuffers[imageIndex], queryPoolTimestamp, 0, 2);
        vkCmdWriteTimestamp(commandBuffers[imageIndex], VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPoolTimestamp, 0);

        VkRenderPassBeginInfo renderPassInfo = { VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];
        renderPassInfo.renderArea.offset = { 0, 0 };
        renderPassInfo.renderArea.extent = swapChainExtent;

        std::array<VkClearValue, 2> clearValues{};
        clearValues[0].color = { 0.0f, 0.0f, 0.0f, 1.0f };
        clearValues[1].depthStencil = { 1.0f, 0 };

        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

        vkCmdBeginRenderPass(commandBuffers[imageIndex], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(commandBuffers[imageIndex], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
        VkBuffer vertexBuffers[] = { vertexBuffer };
        VkDeviceSize offsets[] = { 0 };

        vkCmdBindIndexBuffer(commandBuffers[imageIndex], indexBuffer, 0, VK_INDEX_TYPE_UINT32);

        vkCmdBindDescriptorSets(commandBuffers[imageIndex], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1,
            &descriptorSets[imageIndex], 0, nullptr);

        vkCmdDrawIndexedIndirect(commandBuffers[imageIndex], indirectCommandsBuffer[0], 0, (uint32_t)indirectCommands.size(), sizeof(VkDrawIndexedIndirectCommand));
        //vkCmdDrawIndexed(commandBuffers[imageIndex], static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

        //ImGui_ImplVulkan_RenderDrawData(imguiDrawData, commandBuffers[imageIndex]);

        vkCmdEndRenderPass(commandBuffers[imageIndex]);
        vkCmdWriteTimestamp(commandBuffers[imageIndex], VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPoolTimestamp, 1);

        if (vkEndCommandBuffer(commandBuffers[imageIndex]) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to record command buffer!");
        }
    }

    void createSyncObjects()
    {
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
        imagesInFlight.resize(swapChainImages.size(), VK_NULL_HANDLE);

        VkSemaphoreCreateInfo semaphoreInfo = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        VkFenceCreateInfo fenceInfo = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };

        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
                vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
                vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS)
            {
                throw std::runtime_error("failed to create synchronization objects for a frame");
            }
        }
    }

    void drawFrame()
    {
        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

        uint32_t imageIndex;
        VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);
       
        // swapchain is incompoatible with surface and can't be used for rendering, usually after window resize
        if (result == VK_ERROR_OUT_OF_DATE_KHR)
        {
            recreateSwapChain();
            return;
        }
        else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
        {
            throw std::runtime_error("failed to acquire swap chain image!");
        }

        // Check if a previous frame is using this image
        if (imagesInFlight[imageIndex] != VK_NULL_HANDLE)
        {
            vkWaitForFences(device, 1, &imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX);
        }
        // Mark the image as now being in use by this frame
        imagesInFlight[imageIndex] = inFlightFences[currentFrame];

        updateUniformBuffer(imageIndex);
        //drawIMGUI();
         // Update command buffers?
        updateCommandBuffer(imageIndex);

        VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
        
        VkSemaphore waitSemphores[] = { imageAvailableSemaphores[currentFrame] };
        VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemphores;
        submitInfo.pWaitDstStageMask = waitStages;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[imageIndex];

        VkSemaphore signalSemphores[] = { renderFinishedSemaphores[currentFrame] };
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemphores;
       
        vkResetFences(device, 1, &inFlightFences[currentFrame]);
        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to submit draw command buffer");
        }

        VkPresentInfoKHR presentInfo = { VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemphores;

        VkSwapchainKHR swapChains[] = { swapChain };
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex;
        presentInfo.pResults = nullptr; // optional

        result = vkQueuePresentKHR(presentQueue, &presentInfo);
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized)
        {
            framebufferResized = false;
            recreateSwapChain();
        }
        else if (result != VK_SUCCESS)
        {
            throw std::runtime_error("failed to present swap chain image");
        }
        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
        // GPU timing
        uint64_t timestampResults[2] = {};
        vkGetQueryPoolResults(device, queryPoolTimestamp, 0, 2, sizeof(timestampResults), timestampResults, sizeof(timestampResults[0]), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
        double frameGpuBegin = double(timestampResults[0]) * timestampPeriod * 1e-6;
        double frameGpuEnd = double(timestampResults[1]) * timestampPeriod * 1e-6;


        frameGpuAvg = frameGpuAvg * 0.95 + (frameGpuEnd - frameGpuBegin) * 0.05;

    }
    
    void update()
    {
        auto currentTime = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double, std::milli>(currentTime - previousTime).count(); // Convert to miliseconds
        
        float fpsTimer = (float)std::chrono::duration<double, std::milli>(previousTime - lastTimeStamp).count();

        frames++;
        int frameHistorySize = (int)frameTimes.size();
        frameTimes[frameIndex++] = dt ;
        if (frameIndex >= frameTimes.size())
        {
            float total = 0.0f;
            for (float frameTime : frameTimes)
            {
                total += frameTime;
            }
            averageFrameTime = total / frameHistorySize;
            frameIndex = 0;
        }
   
        /*
        ImGuiIO& io = ImGui::GetIO();



        // only Process keyboard stuff if imgui doesn't want it
        if (!io.WantCaptureKeyboard)
        {*/
            camera.keys.down = glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS;
            camera.keys.up = glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS;
            camera.keys.left = glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS;
            camera.keys.right = glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS;
            camera.update((float)dt);

            if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            {
                glfwSetWindowShouldClose(window, GLFW_TRUE);
            }
            /*
        }

        // Same with mouse capture
        if (!io.WantCaptureMouse)
        {*/
            double xpos, ypos;
            glfwGetCursorPos(window, &xpos, &ypos);
            float xdelta = (float)-(xpos - prevx);
            float ydelta = (float)-(ypos - prevy);

            // we're only going to rotate if the mouse is pressed
            //if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
                camera.rotate(glm::vec3(ydelta * 0.05, xdelta * 0.05f, 0.f));

            prevx = xpos;
            prevy = ypos;
      //  }

        if (fpsTimer > 1000.0f)
        {
            fps = static_cast<uint32_t>((float)frames * (1000.0f / fpsTimer));
            lastTimeStamp = previousTime;
            frames = 0;
        }
        
        previousTime = currentTime;
        checkFilesChanged();
    }
#pragma endregion
    void updateUniformBuffer(uint32_t currentImage)
    {

        static auto startTime = std::chrono::high_resolution_clock::now();

        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();
        
        /*
        ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.view = glm::lookAt(glm::vec3(1.0f, 1.0f, 0.5f), glm::vec3(0.0f, 0.0f, 0.0f), glm::6vec3(0.0f, 0.0f, 1.0f));
        ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float)swapChainExtent.height, 0.1f, 10.0f);
        ubo.proj[1][1] *= -1;
        */
        glm::mat4 model = glm::mat4(1.0f);//glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f)); // The model we use is rotated, so we put it upright here
        glm::mat4 testModel = glm::translate(glm::mat4(1.0f), glm::vec3(200.0, 0.0, 0.0));
        
        UniformBufferObject ubo = {};
        ubo.view = camera.matrices.view; 
        ubo.proj = camera.matrices.perspective; 
        ubo.time = time;
        ubo.cameraPosition = camera.position;
        ubo.lightPos = glm::vec3(lightPos);

        if (lightPos.x > 1100.0 || lightPos.x < 100.0)
            lightSpeed.x *= -1;
        //lightPos += lightSpeed;

        // Update to GPU
        // UBO
        void* data;
        vkMapMemory(device, uniformBuffersMemory[currentImage], 0, sizeof(ubo), 0, &data);
            memcpy(data, &ubo, sizeof(ubo));
        vkUnmapMemory(device, uniformBuffersMemory[currentImage]);

        // Verticies
        memcpy(vertexDataPointers[currentImage], vertices.data(), sizeof(Vertex) * vertices.size());
        // Transform
        memcpy(transformDataPointers[currentImage], transformData.data(), sizeof(TransformData) * transformData.size());
        // Material Data
        memcpy(materialDataPointers[currentImage], matData.data(), sizeof(MaterialData) * matData.size());
        // Draw Data
        memcpy(drawDataPointers[currentImage], drawData.data(), sizeof(DrawData) * drawData.size());
    }

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory)
    {
        VkBufferCreateInfo bufferInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create vertex buffer");
        }

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to allocate vertex buffer memory");
        }
        vkBindBufferMemory(device, buffer, bufferMemory, 0);
    }

    void createDeviceLocalBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer& buffer, VkDeviceMemory& bufferMemory, void* bufferData)
    {
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);


        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, size, 0, &data);
        memcpy(data, bufferData, (size_t)size);
        vkUnmapMemory(device, stagingBufferMemory);

        createBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buffer, bufferMemory);

        copyBuffer(stagingBuffer, buffer, size);

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    void createVertexBuffer()
    {
        //VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();
        //createDeviceLocalBuffer(bufferSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertexBuffer, vertexBufferMemory, vertices.data());
    }

    void createIndexBuffer()
    {
        VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();
        createDeviceLocalBuffer(bufferSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, indexBuffer, indexBufferMemory, indices.data());
    }

    void createUniformBuffers()
    {
        VkDeviceSize bufferSize = sizeof(UniformBufferObject);
        std::cout << sizeof(UniformBufferObject) << std::endl;
        uniformBuffers.resize(swapChainImages.size());
        uniformBuffersMemory.resize(swapChainImages.size());

        for (size_t i = 0; i < swapChainImages.size(); i++)
        {
            createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                uniformBuffers[i], uniformBuffersMemory[i]);
        }
    }

    void createBindlessBuffers()
    {
        matBuffers.resize(swapChainImages.size());
        matBuffersMemory.resize(swapChainImages.size());
        materialDataPointers.resize(swapChainImages.size());

        transformBuffers.resize(swapChainImages.size());
        transformBuffersMemory.resize(swapChainImages.size());
        transformDataPointers.resize(swapChainImages.size());

        drawBuffers.resize(swapChainImages.size());
        drawBuffersMemory.resize(swapChainImages.size());
        drawDataPointers.resize(swapChainImages.size());


        vertexBuffers.resize(swapChainImages.size());
        vertexBuffersMemory.resize(swapChainImages.size());
        vertexDataPointers.resize(swapChainImages.size());

        indirectCommandsBuffer.resize(swapChainImages.size());
        indirectCommandsBufferMemory.resize(swapChainImages.size());

        // TODO: Pick a size for these, the size of the vector WILL change
        VkDeviceSize matBufferSize = sizeof(MaterialData) * matData.size();
        VkDeviceSize transformBufferSize = sizeof(TransformData) * TransformDataCount;
        VkDeviceSize drawBufferSize = sizeof(DrawData) * drawData.size();
        VkDeviceSize vertexBufferSize = sizeof(vertices[0]) * vertices.size();

        for (size_t i = 0; i < swapChainImages.size(); i++)
        {
            // Material Data
            createBuffer(matBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                matBuffers[i], matBuffersMemory[i]);

            // Transform
            createBuffer(transformBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                transformBuffers[i], transformBuffersMemory[i]);

            // DrawData
            createBuffer(drawBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                drawBuffers[i], drawBuffersMemory[i]);

            // Verticies
            createBuffer(vertexBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                vertexBuffers[i], vertexBuffersMemory[i]);

            vkMapMemory(device, matBuffersMemory[i], 0, sizeof(MaterialData) * matData.size(), 0, &materialDataPointers[i]);
            vkMapMemory(device, transformBuffersMemory[i], 0, sizeof(TransformData) * transformData.size(), 0, &transformDataPointers[i]);
            vkMapMemory(device, drawBuffersMemory[i], 0, sizeof(DrawData) * drawData.size(), 0, &drawDataPointers[i]);
            vkMapMemory(device, vertexBuffersMemory[i], 0, sizeof(Vertex) * vertices.size(), 0, &vertexDataPointers[i]);
        }

    }

    void updateIndirectBuffer()
    {
        VkDeviceSize bufferSize = sizeof(VkDrawIndexedIndirectCommand) * indirectCommands.size();
        createDeviceLocalBuffer(bufferSize, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, indirectCommandsBuffer[0], indirectCommandsBufferMemory[0], indirectCommands.data());
    }

    void createDescriptorSetLayout()
    {
        VkDescriptorSetLayoutBinding uboLayoutBinding = {};
        uboLayoutBinding.binding = 0;
        uboLayoutBinding.descriptorCount = 1;
        uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        VkDescriptorSetLayoutBinding materialLayoutBinding = {};
        materialLayoutBinding.binding = 1;
        materialLayoutBinding.descriptorCount = 1;
        materialLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        materialLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutBinding drawLayoutBinding = {};
        drawLayoutBinding.binding = 2;
        drawLayoutBinding.descriptorCount = 1;
        drawLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        drawLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        VkDescriptorSetLayoutBinding transformLayoutBinding = {};
        transformLayoutBinding.binding = 3;
        transformLayoutBinding.descriptorCount = 1;
        transformLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        transformLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        VkDescriptorSetLayoutBinding vertexLayoutBinding = {};
        vertexLayoutBinding.binding = 4;
        vertexLayoutBinding.descriptorCount = 1;
        vertexLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        vertexLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        VkDescriptorSetLayoutBinding samplerLayoutBinding = {};
        samplerLayoutBinding.binding = 5;
        samplerLayoutBinding.descriptorCount = (uint32_t)textures.size();
        samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutBindingFlagsCreateInfoEXT setLayoutBindingsFlags{};
        setLayoutBindingsFlags.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO_EXT;
        setLayoutBindingsFlags.bindingCount = 6;
        std::vector<VkDescriptorBindingFlagsEXT> descriptorBindingFlags = {
            0,
            0,
            0,
            0,
            0,
            VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT_EXT
        };
        setLayoutBindingsFlags.pBindingFlags = descriptorBindingFlags.data();

        std::array<VkDescriptorSetLayoutBinding, 6> bindings = { uboLayoutBinding,
            materialLayoutBinding,
            drawLayoutBinding,
            transformLayoutBinding,
            vertexLayoutBinding, 
            samplerLayoutBinding
        };
        VkDescriptorSetLayoutCreateInfo layoutInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();
        //layoutInfo.pNext = &setLayoutBindingsFlags;

        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor set layout!");
        }
    }
    void createDescriptorPool()
    {
        std::array<VkDescriptorPoolSize, 6> poolSizes = {};

        poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[0].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
      
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[1].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
      
        poolSizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[2].descriptorCount = static_cast<uint32_t>(swapChainImages.size());

        poolSizes[3].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[3].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
      
        poolSizes[4].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[4].descriptorCount = static_cast<uint32_t>(swapChainImages.size());

        poolSizes[5].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes[5].descriptorCount = (uint32_t)textures.size();

        VkDescriptorPoolCreateInfo poolInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = static_cast<uint32_t>(swapChainImages.size());

        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor pool!");
        }
    }

    void createDescriptorSets()
    {
        std::vector<VkDescriptorSetLayout> layouts(swapChainImages.size(), descriptorSetLayout);
        VkDescriptorSetAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(swapChainImages.size());
        allocInfo.pSetLayouts = layouts.data();

        descriptorSets.resize(swapChainImages.size());
        if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to allocate descriptor sets");
        }

        for (size_t i = 0; i < swapChainImages.size(); i++)
        {
            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = uniformBuffers[i];
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBufferObject);
            
            VkDescriptorBufferInfo matBufferInfo{};
            matBufferInfo.buffer = matBuffers[i];
            matBufferInfo.offset = 0;
            matBufferInfo.range = sizeof(MaterialData) * matData.size();

            VkDescriptorBufferInfo drawBufferInfo{};
            drawBufferInfo.buffer = drawBuffers[i];
            drawBufferInfo.offset = 0;
            drawBufferInfo.range = sizeof(DrawData) * drawData.size();

            VkDescriptorBufferInfo transformBufferInfo{};
            transformBufferInfo.buffer = transformBuffers[i];
            transformBufferInfo.offset = 0;
            transformBufferInfo.range = sizeof(TransformData) * transformData.size();

            VkDescriptorBufferInfo vertexBufferInfo{};
            vertexBufferInfo.buffer = vertexBuffers[i];
            vertexBufferInfo.offset = 0;
            vertexBufferInfo.range =  sizeof(vertices[0])* vertices.size();;


            std::vector<VkDescriptorImageInfo> imageInfo;
            imageInfo.resize(textures.size());
            for (uint32_t i = 0; i < textures.size(); i++)
            {
                imageInfo[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                imageInfo[i].imageView = textures[i].imageView;
                imageInfo[i].sampler = textureSampler;
            }


            std::array<VkWriteDescriptorSet, 6> descriptorWrites{};

            descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[0].dstSet = descriptorSets[i];
            descriptorWrites[0].dstBinding = 0;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].pBufferInfo = &bufferInfo;
  
            descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[1].dstSet = descriptorSets[i];
            descriptorWrites[1].dstBinding = 1;
            descriptorWrites[1].dstArrayElement = 0;
            descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[1].descriptorCount = 1;
            descriptorWrites[1].pBufferInfo = &matBufferInfo;
           
            descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[2].dstSet = descriptorSets[i];
            descriptorWrites[2].dstBinding = 2;
            descriptorWrites[2].dstArrayElement = 0;
            descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[2].descriptorCount = 1;
            descriptorWrites[2].pBufferInfo = &drawBufferInfo;

            descriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[3].dstSet = descriptorSets[i];
            descriptorWrites[3].dstBinding = 3;
            descriptorWrites[3].dstArrayElement = 0;
            descriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[3].descriptorCount = 1;
            descriptorWrites[3].pBufferInfo = &transformBufferInfo;
            
            descriptorWrites[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[4].dstSet = descriptorSets[i];
            descriptorWrites[4].dstBinding = 4;
            descriptorWrites[4].dstArrayElement = 0;
            descriptorWrites[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[4].descriptorCount = 1;
            descriptorWrites[4].pBufferInfo = &vertexBufferInfo;

            descriptorWrites[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[5].dstSet = descriptorSets[i];
            descriptorWrites[5].dstBinding = 5;
            descriptorWrites[5].dstArrayElement = 0;
            descriptorWrites[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[5].descriptorCount = (uint32_t)imageInfo.size();
            descriptorWrites[5].pImageInfo = imageInfo.data();

            vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
        }
    }

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
    {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
        {
            if ((typeFilter & (1 << i)) && // usable for the buffer
                (memProperties.memoryTypes[i].propertyFlags & properties) == properties) // and properties are all supported
            {
                return i;
            }
        }
        throw std::runtime_error("failed to find suitable memory type");
    }

    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size)
    {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkBufferCopy copyRegion = {};
        copyRegion.size = size;
        vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

        endSingleTimeCommands(commandBuffer);
    }

    void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout)
    {
        VkPipelineStageFlags sourceStage;
        VkPipelineStageFlags destinationStage;

        VkCommandBuffer commandBuffer = beginSingleTimeCommands();
        
        VkImageMemoryBarrier barrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

            if (hasStencilComponent(format)) {
                barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
            }
        }
        else {
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        }
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;

        if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
        {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        }
        else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }
        else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        }
        else
        {
            throw std::invalid_argument("unsupported layout transition");
        }


        vkCmdPipelineBarrier(
            commandBuffer,
            sourceStage, destinationStage,
            0,
            0, nullptr,
            0, nullptr,
            1, &barrier
        );
        
        endSingleTimeCommands(commandBuffer);
    }

    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height)
    {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();
        
        VkBufferImageCopy region = {};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;

        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;

        region.imageOffset = { 0, 0, 0 };
        region.imageExtent = {
            width,
            height,
            1
        };
        vkCmdCopyBufferToImage(
            commandBuffer,
            buffer,
            image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &region
        );
        
        endSingleTimeCommands(commandBuffer);
    }

    void loadModel()
    {
        drawData.clear();

        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn, err;
        std::string baseDir = MODEL_PATH.find_last_of("/\\") != std::string::npos ? MODEL_PATH.substr(0, MODEL_PATH.find_last_of("/\\")) : ".\\";
        //baseDir = "textures/\\";
        if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, MODEL_PATH.c_str(), baseDir.c_str()))
        {
            throw std::runtime_error(warn + err);
        }

        std::unordered_map<Vertex, uint32_t> uniqueVerticies{};
        //std::cout << warn << std::endl;
        TransformData td{ };
        td.model = glm::mat4(1.0f);
        transformData.push_back(td);
        // TODO: Load materials
        int i = 0;
        for (auto mat : materials)
        {
            MaterialData md{};
            md.reflectance = mat.shininess;
            if (mat.ambient_texname.length() > 0)
            {
                md.albedoTexture = i;
                createTextureImage(mat.ambient_texname.c_str());
                i++;
            }
            
            if (mat.bump_texname.length() > 0)
            {
                md.normalTexture = i;
                createTextureImage(mat.bump_texname.c_str());
                i++;
            }

            if (mat.specular_texname.length() > 0)
            {
                md.roughnessTexture = i;
                createTextureImage(mat.specular_texname.c_str());
                i++;
            }
            else {
                md.roughnessTexture = -1;
            }
            //std::cout << mat. << std::endl;
            matData.push_back(md);
        }
        uint32_t vertexOffset = 0;
        int indexCounter = 0;
        std::vector<Vertex> interested;
        for (const auto& shape : shapes)
        {
            // set this shape's material based off the first face's material
            int current_material_id = shape.mesh.material_ids[0];
            //std::cout << "SIZE " << shape.mesh.indices.size() << std::endl;
            DrawData dd;
            dd.materialIndex = current_material_id;
            dd.transformIndex =  0;
            dd.unused = 0;
            dd.vertexOffset = (uint32_t)vertices.size();
            drawData.push_back(dd);
            
            VkDrawIndexedIndirectCommand cmd;
            cmd.indexCount = uint32_t(shape.mesh.indices.size());
            cmd.instanceCount = 1;
            cmd.firstIndex = uint32_t(indices.size());
            cmd.vertexOffset = 0;
            cmd.firstInstance = 0;
            indirectCommands.push_back(cmd);

            int j = 0;
            for (const auto& index : shape.mesh.indices)
            {

                Vertex vertex{};

                vertex.pos = {
                    attrib.vertices[3 * (int)index.vertex_index + 0],
                    attrib.vertices[3 * (int)index.vertex_index + 1],
                    attrib.vertices[3 * (int)index.vertex_index + 2]
                };
                
                vertex.normal = {
                    attrib.normals[3 * (int)index.normal_index + 0],
                    attrib.normals[3 * (int)index.normal_index + 1],
                    attrib.normals[3 * (int)index.normal_index + 2],
                };
                
                vertex.texCoord = {
                    attrib.texcoords[2 * (int)index.texcoord_index + 0],
                    1.0f - attrib.texcoords[2 * (int)index.texcoord_index + 1]
                };


                vertex.color = { 1.0f, 1.0f, 1.0f };
                if (uniqueVerticies.count(vertex) == 0)
                {
                    uniqueVerticies[vertex] = static_cast<uint32_t>(vertices.size());
                    vertices.push_back(vertex);
                }
                indices.push_back(uniqueVerticies[vertex]);
                if (indexCounter == 1 && j == 0)
                {
                    //std::cout << glm::to_string(vertex.pos) << std::endl;
                    //std::cout << indices.size() << " " << vertices.size() << std::endl;
                }
                j++;
                if (indexCounter == 1)
                {
                    interested.push_back(vertex);
                }
            }
            indexCounter++;

        }
        for (int i = 0; i < vertices.size(); i+=3)
        {
            Vertex v1 = vertices[i];
            Vertex v2 = vertices[i + 1];
            Vertex v3 = vertices[i + 2];

            glm::vec3 edge1 = v2.pos - v1.pos;
            glm::vec3 edge2 = v3.pos - v1.pos;
            glm::vec2 deltaUV1 = v2.texCoord - v1.texCoord;
            glm::vec2 deltaUV2 = v3.texCoord - v1.texCoord;

            float f = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y);
            glm::vec3 tangent1;
            tangent1.x = f * (deltaUV2.y * edge1.x - deltaUV1.y * edge2.x);
            tangent1.y = f * (deltaUV2.y * edge1.y - deltaUV1.y * edge2.y);
            tangent1.z = f * (deltaUV2.y * edge1.z - deltaUV1.y * edge2.z);
            
            vertices[i].tangent = tangent1;

            vertices[i + 1].tangent = tangent1;

            vertices[i + 2].tangent = tangent1;

        }
    }

    void checkFilesChanged()
    {
        for (auto& file : std::filesystem::recursive_directory_iterator(SHADER_PATH)) 
        {
            if (file.path().extension().string() == ".spv") continue;
            auto last_write_time = std::filesystem::last_write_time(file);

            auto filePath = file.path().string();
            auto keyExists = shaderPaths.find(filePath) != shaderPaths.end();
            if (keyExists)
            {
                // File was written to since we last checked
                if (last_write_time != shaderPaths[filePath])
                {
                    std::cout << filePath << " was changed!" << std::endl;
                    shaderPaths[filePath] = last_write_time;
                    shaderFileChanged(file.path());
                }
            }
            else 
            {
                shaderPaths[filePath] = last_write_time;
            }
        }
    }

    void shaderFileChanged(std::filesystem::path shaderSourceFile)
    {
        std::string filePath = shaderSourceFile.string();
        std::string compiledPath = shaderSourceFile.replace_extension(".spv").string();
        std::array<char, 128> buffer;
        std::string result;
        std::string cmd = "glslc.exe " + filePath + " -o " +  compiledPath;
        std::unique_ptr<FILE, decltype(&_pclose)> pipe(_popen(cmd.c_str(), "r"), _pclose);
        if (!pipe) {
            throw std::runtime_error("failed to send shell command for shader reload");
        }
        while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
            result += buffer.data();
        }
        std::cout << result << std::endl;
        recreateGraphicsPipeline();
    }
    void createIMGUI()
    {
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO(); (void)io;

        ImGui::StyleColorsDark();
        
        {
            VkDescriptorPoolSize pool_sizes[] =
            {
                { VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
                { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
                { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
                { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
                { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
                { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
                { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
                { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
                { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
                { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
                { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
            };
            VkDescriptorPoolCreateInfo pool_info = {};
            pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
            pool_info.maxSets = 1000 * IM_ARRAYSIZE(pool_sizes);
            pool_info.poolSizeCount = (uint32_t)IM_ARRAYSIZE(pool_sizes);
            pool_info.pPoolSizes = pool_sizes;
            VkResult err = vkCreateDescriptorPool(device, &pool_info, nullptr, &pool);
        }

        ImGui_ImplGlfw_InitForVulkan(window, true);
        ImGui_ImplVulkan_InitInfo init_info = {};
        init_info.Instance = instance;
        init_info.PhysicalDevice = physicalDevice;
        init_info.Device = device;
        init_info.QueueFamily = queueFamilyIndices.graphicsFamily.value(); // ???
        init_info.Queue = graphicsQueue; // ???
        init_info.PipelineCache = VK_NULL_HANDLE; // TODO
        init_info.DescriptorPool = pool;
        init_info.Allocator = NULL; // hope this canb e null
        init_info.MinImageCount = (uint32_t)swapChainFramebuffers.size();
        init_info.ImageCount = (uint32_t)swapChainFramebuffers.size();
        init_info.CheckVkResultFn = NULL; // same here
        ImGui_ImplVulkan_Init(&init_info, renderPass);
        
         VkCommandBuffer cb = beginSingleTimeCommands();
         ImGui_ImplVulkan_CreateFontsTexture(cb);
         endSingleTimeCommands(cb);     
         ImGui_ImplVulkan_DestroyFontUploadObjects(); 
    }

    void drawIMGUI()
    {
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();

        ImGui::NewFrame();
        ImGui::Begin("Performance");
        ImGui::Columns(2);

        ImGui::Text("POS");
        ImGui::Text("ROT");
        ImGui::NextColumn();
        //ImGui::Text(std::to_string(fps).c_str());
        ImGui::Text(glm::to_string(camera.position).c_str());
        ImGui::Text(glm::to_string(camera.rotation).c_str());

        
        //ImGui::Text(std::to_string(averageFrameTime).c_str());

        //auto size = ImGui::GetWindowSize();
        //std::cout << size.x << ", " << size.y << std::endl;
        ImGui::End();
        /*
        float cameraWindowWidth = 250;
        float cameraWindowHeight = 400;
        //ImGui::SetNextWindowPos(ImVec2(WIDTH - cameraWindowWidth, 0), ImGuiCond_Once, ImVec2(0, 0));
        //ImGui::SetNextWindowSize(ImVec2(cameraWindowWidth, cameraWindowHeight), ImGuiCond_Once);
        ImGui::Begin("Camera");
        ImGui::DragFloat3("Position", (float*)&camera.position);
        ImGui::End();
        */

        ImGui::Render();
        imguiDrawData = ImGui::GetDrawData();
    }

    void initGame()
    {
        frameTimes.fill(0.0f);

        camera.setPerspective(60.0f, swapChainExtent.width / (float)swapChainExtent.height, 1.0f, 10000.0f);

        glm::vec3 pos = glm::vec3(0.f, -150.f, 65.f);
        camera.setPosition(pos);
        camera.setRotation(glm::vec3(180.0f, -88.8f, 90.f));
        camera.setMovementSpeed(0.5f);
        camera.flipY = false;

        lightPos = glm::vec3(0.f, -250.0f, 65.f);
        lightSpeed = glm::vec3(1.5, 0, 0);
    }
};

int main() {
    HelloTriangleApplication app;

    try {
        app.run();
    }
    catch (const std::exception & e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}