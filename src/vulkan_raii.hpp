#pragma once

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <utility>

namespace danvulkan::vk
{
class Instance
{
public:
    Instance() = default;
    ~Instance() { reset(); }

    Instance(const Instance&) = delete;
    Instance& operator=(const Instance&) = delete;

    Instance(Instance&& other) noexcept : handle_(std::exchange(other.handle_, VK_NULL_HANDLE)) {}
    Instance& operator=(Instance&& other) noexcept
    {
        if (this != &other)
        {
            reset();
            handle_ = std::exchange(other.handle_, VK_NULL_HANDLE);
        }
        return *this;
    }

    [[nodiscard]] VkInstance get() const noexcept { return handle_; }
    [[nodiscard]] VkInstance* put() noexcept
    {
        reset();
        return &handle_;
    }
    explicit operator bool() const noexcept { return handle_ != VK_NULL_HANDLE; }
    operator VkInstance() const noexcept { return handle_; }

    void reset(VkInstance handle = VK_NULL_HANDLE) noexcept
    {
        if (handle_ != VK_NULL_HANDLE)
        {
            vkDestroyInstance(handle_, nullptr);
        }
        handle_ = handle;
    }

private:
    VkInstance handle_ = VK_NULL_HANDLE;
};

class Surface
{
public:
    Surface() = default;
    ~Surface() { reset(); }

    Surface(const Surface&) = delete;
    Surface& operator=(const Surface&) = delete;

    Surface(Surface&& other) noexcept
        : instance_(std::exchange(other.instance_, VK_NULL_HANDLE)),
          handle_(std::exchange(other.handle_, VK_NULL_HANDLE))
    {
    }
    Surface& operator=(Surface&& other) noexcept
    {
        if (this != &other)
        {
            reset();
            instance_ = std::exchange(other.instance_, VK_NULL_HANDLE);
            handle_ = std::exchange(other.handle_, VK_NULL_HANDLE);
        }
        return *this;
    }

    [[nodiscard]] VkSurfaceKHR get() const noexcept { return handle_; }
    [[nodiscard]] VkSurfaceKHR* put(VkInstance instance) noexcept
    {
        reset();
        instance_ = instance;
        return &handle_;
    }
    operator VkSurfaceKHR() const noexcept { return handle_; }

    void reset() noexcept
    {
        if (handle_ != VK_NULL_HANDLE)
        {
            vkDestroySurfaceKHR(instance_, handle_, nullptr);
        }
        handle_ = VK_NULL_HANDLE;
        instance_ = VK_NULL_HANDLE;
    }

private:
    VkInstance instance_ = VK_NULL_HANDLE;
    VkSurfaceKHR handle_ = VK_NULL_HANDLE;
};

class DebugMessenger
{
public:
    DebugMessenger() = default;
    ~DebugMessenger() { reset(); }

    DebugMessenger(const DebugMessenger&) = delete;
    DebugMessenger& operator=(const DebugMessenger&) = delete;

    [[nodiscard]] VkDebugUtilsMessengerEXT* put(VkInstance instance) noexcept
    {
        reset();
        instance_ = instance;
        return &handle_;
    }

    void reset() noexcept
    {
        if (handle_ != VK_NULL_HANDLE)
        {
            const auto destroy = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT"));
            if (destroy != nullptr)
            {
                destroy(instance_, handle_, nullptr);
            }
        }
        handle_ = VK_NULL_HANDLE;
        instance_ = VK_NULL_HANDLE;
    }

private:
    VkInstance instance_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT handle_ = VK_NULL_HANDLE;
};

class Device
{
public:
    Device() = default;
    ~Device() { reset(); }

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    Device(Device&& other) noexcept : handle_(std::exchange(other.handle_, VK_NULL_HANDLE)) {}
    Device& operator=(Device&& other) noexcept
    {
        if (this != &other)
        {
            reset();
            handle_ = std::exchange(other.handle_, VK_NULL_HANDLE);
        }
        return *this;
    }

    [[nodiscard]] VkDevice get() const noexcept { return handle_; }
    [[nodiscard]] VkDevice* put() noexcept
    {
        reset();
        return &handle_;
    }
    explicit operator bool() const noexcept { return handle_ != VK_NULL_HANDLE; }
    operator VkDevice() const noexcept { return handle_; }

    void reset(VkDevice handle = VK_NULL_HANDLE) noexcept
    {
        if (handle_ != VK_NULL_HANDLE)
        {
            vkDestroyDevice(handle_, nullptr);
        }
        handle_ = handle;
    }

private:
    VkDevice handle_ = VK_NULL_HANDLE;
};

class Allocator
{
public:
    Allocator() = default;
    ~Allocator() { reset(); }

    Allocator(const Allocator&) = delete;
    Allocator& operator=(const Allocator&) = delete;

    Allocator(Allocator&& other) noexcept
        : handle_(std::exchange(other.handle_, VK_NULL_HANDLE))
    {
    }
    Allocator& operator=(Allocator&& other) noexcept
    {
        if (this != &other)
        {
            reset();
            handle_ = std::exchange(other.handle_, VK_NULL_HANDLE);
        }
        return *this;
    }

    [[nodiscard]] VmaAllocator get() const noexcept { return handle_; }
    [[nodiscard]] VmaAllocator* put() noexcept
    {
        reset();
        return &handle_;
    }
    explicit operator bool() const noexcept { return handle_ != VK_NULL_HANDLE; }
    operator VmaAllocator() const noexcept { return handle_; }

    void reset(VmaAllocator handle = VK_NULL_HANDLE) noexcept
    {
        if (handle_ != VK_NULL_HANDLE)
        {
            vmaDestroyAllocator(handle_);
        }
        handle_ = handle;
    }

private:
    VmaAllocator handle_ = VK_NULL_HANDLE;
};

class Swapchain
{
public:
    Swapchain() = default;
    ~Swapchain() { reset(); }

    Swapchain(const Swapchain&) = delete;
    Swapchain& operator=(const Swapchain&) = delete;
    Swapchain(Swapchain&& other) noexcept
        : device_(std::exchange(other.device_, VK_NULL_HANDLE)),
          handle_(std::exchange(other.handle_, VK_NULL_HANDLE))
    {
    }
    Swapchain& operator=(Swapchain&& other) noexcept
    {
        if (this != &other)
        {
            reset();
            device_ = std::exchange(other.device_, VK_NULL_HANDLE);
            handle_ = std::exchange(other.handle_, VK_NULL_HANDLE);
        }
        return *this;
    }

    [[nodiscard]] VkSwapchainKHR get() const noexcept { return handle_; }
    [[nodiscard]] VkSwapchainKHR* put(VkDevice device) noexcept
    {
        reset();
        device_ = device;
        return &handle_;
    }
    operator VkSwapchainKHR() const noexcept { return handle_; }

    void reset() noexcept
    {
        if (handle_ != VK_NULL_HANDLE)
        {
            vkDestroySwapchainKHR(device_, handle_, nullptr);
        }
        handle_ = VK_NULL_HANDLE;
        device_ = VK_NULL_HANDLE;
    }

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkSwapchainKHR handle_ = VK_NULL_HANDLE;
};

class Buffer
{
public:
    Buffer() = default;
    Buffer(VmaAllocator allocator, VkBuffer buffer, VmaAllocation allocation,
        void* mapped, VkDeviceSize size) noexcept
        : allocator_(allocator), buffer_(buffer), allocation_(allocation), mapped_(mapped), size_(size)
    {
    }
    ~Buffer() { reset(); }

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    Buffer(Buffer&& other) noexcept
        : allocator_(std::exchange(other.allocator_, VK_NULL_HANDLE)),
          buffer_(std::exchange(other.buffer_, VK_NULL_HANDLE)),
          allocation_(std::exchange(other.allocation_, VK_NULL_HANDLE)),
          mapped_(std::exchange(other.mapped_, nullptr)),
          size_(std::exchange(other.size_, 0))
    {
    }
    Buffer& operator=(Buffer&& other) noexcept
    {
        if (this != &other)
        {
            reset();
            allocator_ = std::exchange(other.allocator_, VK_NULL_HANDLE);
            buffer_ = std::exchange(other.buffer_, VK_NULL_HANDLE);
            allocation_ = std::exchange(other.allocation_, VK_NULL_HANDLE);
            mapped_ = std::exchange(other.mapped_, nullptr);
            size_ = std::exchange(other.size_, 0);
        }
        return *this;
    }

    [[nodiscard]] VkBuffer get() const noexcept { return buffer_; }
    [[nodiscard]] VmaAllocation allocation() const noexcept { return allocation_; }
    [[nodiscard]] void* mapped() const noexcept { return mapped_; }
    [[nodiscard]] VkDeviceSize size() const noexcept { return size_; }
    operator VkBuffer() const noexcept { return buffer_; }

    [[nodiscard]] VkResult flush(VkDeviceSize offset = 0, VkDeviceSize size = VK_WHOLE_SIZE) const noexcept
    {
        return vmaFlushAllocation(allocator_, allocation_, offset, size);
    }

    void reset() noexcept
    {
        if (buffer_ != VK_NULL_HANDLE && allocation_ != VK_NULL_HANDLE)
        {
            vmaDestroyBuffer(allocator_, buffer_, allocation_);
        }
        mapped_ = nullptr;
        allocation_ = VK_NULL_HANDLE;
        buffer_ = VK_NULL_HANDLE;
        allocator_ = VK_NULL_HANDLE;
        size_ = 0;
    }

private:
    VmaAllocator allocator_ = VK_NULL_HANDLE;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = VK_NULL_HANDLE;
    void* mapped_ = nullptr;
    VkDeviceSize size_ = 0;
};

class Image
{
public:
    Image() = default;
    Image(VkDevice device, VmaAllocator allocator, VkImage image, VmaAllocation allocation) noexcept
        : device_(device), allocator_(allocator), image_(image), allocation_(allocation)
    {
    }
    ~Image() { reset(); }

    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;

    Image(Image&& other) noexcept
        : device_(std::exchange(other.device_, VK_NULL_HANDLE)),
          allocator_(std::exchange(other.allocator_, VK_NULL_HANDLE)),
          image_(std::exchange(other.image_, VK_NULL_HANDLE)),
          allocation_(std::exchange(other.allocation_, VK_NULL_HANDLE)),
          view_(std::exchange(other.view_, VK_NULL_HANDLE))
    {
    }
    Image& operator=(Image&& other) noexcept
    {
        if (this != &other)
        {
            reset();
            device_ = std::exchange(other.device_, VK_NULL_HANDLE);
            allocator_ = std::exchange(other.allocator_, VK_NULL_HANDLE);
            image_ = std::exchange(other.image_, VK_NULL_HANDLE);
            allocation_ = std::exchange(other.allocation_, VK_NULL_HANDLE);
            view_ = std::exchange(other.view_, VK_NULL_HANDLE);
        }
        return *this;
    }

    [[nodiscard]] VkImage get() const noexcept { return image_; }
    [[nodiscard]] VkImageView view() const noexcept { return view_; }
    operator VkImage() const noexcept { return image_; }

    void setView(VkImageView view) noexcept
    {
        if (view_ != VK_NULL_HANDLE)
        {
            vkDestroyImageView(device_, view_, nullptr);
        }
        view_ = view;
    }

    void reset() noexcept
    {
        if (view_ != VK_NULL_HANDLE)
        {
            vkDestroyImageView(device_, view_, nullptr);
        }
        if (image_ != VK_NULL_HANDLE && allocation_ != VK_NULL_HANDLE)
        {
            vmaDestroyImage(allocator_, image_, allocation_);
        }
        view_ = VK_NULL_HANDLE;
        allocation_ = VK_NULL_HANDLE;
        image_ = VK_NULL_HANDLE;
        allocator_ = VK_NULL_HANDLE;
        device_ = VK_NULL_HANDLE;
    }

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VmaAllocator allocator_ = VK_NULL_HANDLE;
    VkImage image_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = VK_NULL_HANDLE;
    VkImageView view_ = VK_NULL_HANDLE;
};
} // namespace danvulkan::vk
