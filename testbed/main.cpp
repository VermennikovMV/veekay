#include <algorithm>
#include <cstdint>
#include <climits>
#include <cstring>
#include <limits>
#include <array>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <string>

#include <veekay/input.hpp>
#include <veekay/veekay.hpp>

#include <vulkan/vulkan_core.h>
#include <imgui.h>
#include <lodepng.h>

namespace {

constexpr uint32_t max_models = 1024;
constexpr uint32_t max_point_lights = 4;
constexpr uint32_t max_materials = 4;

enum class LightMode : uint32_t {
        Diffuse = 0,
        Directional = 1,
        Point = 2,
};

struct Vertex {
        veekay::vec3 position;
        veekay::vec3 normal;
        veekay::vec2 uv;
        // NOTE: You can add more attributes
};

struct Material {
        veekay::vec3 ambient_color;
        veekay::vec3 diffuse_color;
        veekay::vec3 specular_color;
        float shininess = 32.0f;

        VkSampler sampler = VK_NULL_HANDLE;
        veekay::graphics::Texture* texture = nullptr;
        VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
};

struct DirectionalLight {
        veekay::vec4 direction_intensity;
        veekay::vec4 color;
};

struct PointLight {
        veekay::vec4 position_intensity;
        veekay::vec4 color;
};

struct SceneUniforms {
        veekay::mat4 view_projection;
        veekay::vec4 camera_position;
        veekay::vec4 ambient_color;
        veekay::vec4 diffuse_color;
        veekay::vec4 light_mode;
        DirectionalLight directional_light;
        veekay::vec4 point_light_count;
        PointLight point_lights[max_point_lights];
};

struct ModelUniforms {
        veekay::mat4 model;
        veekay::vec4 ambient_color;
        veekay::vec4 diffuse_color;
        veekay::vec4 specular_color_shininess;
};

struct Mesh {
	veekay::graphics::Buffer* vertex_buffer;
	veekay::graphics::Buffer* index_buffer;
	uint32_t indices;
};

struct Transform {
	veekay::vec3 position = {};
	veekay::vec3 scale = {1.0f, 1.0f, 1.0f};
	veekay::vec3 rotation = {};

	// NOTE: Model matrix (translation, rotation and scaling)
	veekay::mat4 matrix() const;
};

struct Model {
        Mesh mesh;
        Transform transform;
        float angular_speed = 1.0f;
        veekay::vec3 rotation_axis = {0.0f, 1.0f, 0.0f};
        Material* material = nullptr;
        veekay::mat4 override_model = veekay::mat4::identity();
        bool use_override = false;
};

struct Camera {
	constexpr static float default_fov = 60.0f;
	constexpr static float default_near_plane = 0.01f;
	constexpr static float default_far_plane = 100.0f;

	veekay::vec3 position = {};
	veekay::vec3 rotation = {};

	float fov = default_fov;
	float near_plane = default_near_plane;
	float far_plane = default_far_plane;

	// NOTE: View matrix of camera (inverse of a transform)
	veekay::mat4 view() const;

	// NOTE: View and projection composition
	veekay::mat4 view_projection(float aspect_ratio) const;
};

// NOTE: Scene objects
inline namespace {
        constexpr float ground_height = -0.5f;
        Camera camera{
                .position = {0.0f, -0.5f, -3.0f}
        };

        std::vector<Model> models;
        std::vector<Material> materials;

        struct ShadowInstance {
                size_t source_index;
                size_t shadow_index;
        };

        std::vector<ShadowInstance> shadow_instances;

        struct LightingState {
                veekay::vec4 ambient_color;
                veekay::vec4 diffuse_color;
                DirectionalLight directional_light;
                std::array<PointLight, max_point_lights> point_lights;
                uint32_t point_light_count;
        };

        LightingState lighting = [] {
                LightingState state{};
                const veekay::vec3 dir = veekay::vec3::normalized(veekay::vec3{-0.3f, -1.0f, -0.2f});
                state.ambient_color = veekay::vec4{0.08f, 0.08f, 0.1f, 1.0f};
                state.diffuse_color = veekay::vec4{1.0f, 0.95f, 0.9f, 0.8f};
                state.directional_light = DirectionalLight{
                        .direction_intensity = veekay::vec4{dir.x, dir.y, dir.z, 1.0f},
                        .color = veekay::vec4{1.0f, 0.95f, 0.9f, 1.0f},
                };
                state.point_lights = {
                        PointLight{
                                .position_intensity = veekay::vec4{-1.5f, 0.0f, -1.0f, 8.0f},
                                .color = veekay::vec4{1.0f, 0.9f, 0.7f, 1.0f},
                        },
                        PointLight{},
                        PointLight{},
                        PointLight{}
                };
                state.point_light_count = 1u;
                return state;
        }();

        LightMode light_mode = LightMode::Directional;

        bool mouse_captured = false;
        float camera_move_speed = 3.5f;
        float camera_sensitivity = 0.08f;
}

float toRadians(float degrees) {
        return degrees * float(M_PI) / 180.0f;
}

veekay::vec3 forwardFromRotation(const veekay::vec3& rotation) {
        const float pitch = toRadians(rotation.x);
        const float yaw = toRadians(rotation.y);

        veekay::vec3 forward{
                std::cos(pitch) * std::sin(yaw),
                std::sin(pitch),
                std::cos(pitch) * std::cos(yaw)
        };

        float length = veekay::vec3::length(forward);
        if (length > std::numeric_limits<float>::epsilon()) {
                forward = forward / length;
        }

        return forward;
}

veekay::vec3 rightFromRotation(const veekay::vec3& rotation) {
        const veekay::vec3 forward = forwardFromRotation(rotation);
        const veekay::vec3 up{0.0f, 1.0f, 0.0f};

        veekay::vec3 right = veekay::vec3::cross(up, forward);
        float length = veekay::vec3::length(right);
        if (length > std::numeric_limits<float>::epsilon()) {
                right = right / length;
        }

        return right;
}

veekay::mat4 directionalShadowProjection(const veekay::vec3& light_direction, float plane_height) {
        veekay::vec3 n{0.0f, 1.0f, 0.0f};
        veekay::vec3 l = veekay::vec3::normalized(light_direction);

        const float d = -plane_height;
        float nl = veekay::vec3::dot(n, l);
        const float epsilon = 1e-4f;
        if (std::fabs(nl) < epsilon) {
                nl = (nl < 0.0f ? -epsilon : epsilon);
        }

        veekay::mat4 projection{};

        projection[0][0] = nl - n.x * l.x;
        projection[0][1] = -n.x * l.y;
        projection[0][2] = -n.x * l.z;
        projection[0][3] = -n.x * d;

        projection[1][0] = -n.y * l.x;
        projection[1][1] = nl - n.y * l.y;
        projection[1][2] = -n.y * l.z;
        projection[1][3] = -n.y * d;

        projection[2][0] = -n.z * l.x;
        projection[2][1] = -n.z * l.y;
        projection[2][2] = nl - n.z * l.z;
        projection[2][3] = -n.z * d;

        projection[3][0] = -l.x;
        projection[3][1] = -l.y;
        projection[3][2] = -l.z;
        projection[3][3] = nl;

        return projection;
}

veekay::mat4 shadowMatrixForModel(const Model& model, const veekay::vec3& light_direction) {
        const veekay::mat4 model_matrix = model.transform.matrix();
        veekay::vec3 normalized_dir = light_direction;
        float len = veekay::vec3::length(normalized_dir);
        if (len > std::numeric_limits<float>::epsilon()) {
                normalized_dir = normalized_dir / len;
        }

        veekay::mat4 projection = directionalShadowProjection(normalized_dir, ground_height);

        // NOTE: Keep the projected geometry locked to the ground plane so the shadow never falls
        // under or above the surface when the light direction changes.
        veekay::mat4 clamp_to_ground = veekay::mat4::translation({0.0f, ground_height, 0.0f}) *
                                       veekay::mat4::scaling({1.0f, 0.0f, 1.0f});

        return clamp_to_ground * projection * model_matrix;
}

// NOTE: Vulkan objects
inline namespace {
	VkShaderModule vertex_shader_module;
        VkShaderModule fragment_shader_module;

        VkDescriptorPool descriptor_pool;
        VkDescriptorSetLayout descriptor_set_layout;

        VkPipelineLayout pipeline_layout;
        VkPipeline pipeline;

        veekay::graphics::Buffer* scene_uniforms_buffer;
        veekay::graphics::Buffer* model_uniforms_buffer;

        Mesh cone_mesh;
        Mesh plane_mesh;

        veekay::graphics::Texture* missing_texture;
        VkSampler missing_texture_sampler;

        veekay::graphics::Texture* texture;
        VkSampler texture_sampler;

        veekay::graphics::Texture* white_texture;
        VkSampler flat_sampler;
}

veekay::mat4 Transform::matrix() const {
        auto t = veekay::mat4::translation(position);
        auto s = veekay::mat4::scaling(scale);

        auto rx = veekay::mat4::rotation({1.0f, 0.0f, 0.0f}, toRadians(rotation.x));
        auto ry = veekay::mat4::rotation({0.0f, 1.0f, 0.0f}, toRadians(rotation.y));
        auto rz = veekay::mat4::rotation({0.0f, 0.0f, 1.0f}, toRadians(rotation.z));

        return t * rz * ry * rx * s;
}

veekay::mat4 Camera::view() const {
        auto t = veekay::mat4::translation(-position);

        auto rx = veekay::mat4::rotation({1.0f, 0.0f, 0.0f}, toRadians(-rotation.x));
        auto ry = veekay::mat4::rotation({0.0f, 1.0f, 0.0f}, toRadians(-rotation.y));
        auto rz = veekay::mat4::rotation({0.0f, 0.0f, 1.0f}, toRadians(-rotation.z));

        auto flip = veekay::mat4::scaling({1.0f, -1.0f, 1.0f});

        return flip * rz * ry * rx * t;
}

veekay::mat4 Camera::view_projection(float aspect_ratio) const {
	auto projection = veekay::mat4::projection(fov, aspect_ratio, near_plane, far_plane);

	return view() * projection;
}

// NOTE: Loads shader byte code from file
// NOTE: Your shaders are compiled via CMake with this code too, look it up
VkShaderModule loadShaderModule(const char* path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        size_t size = file.tellg();
        std::vector<uint32_t> buffer(size / sizeof(uint32_t));
        file.seekg(0);
	file.read(reinterpret_cast<char*>(buffer.data()), size);
	file.close();

	VkShaderModuleCreateInfo info{
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = size,
		.pCode = buffer.data(),
	};

	VkShaderModule result;
	if (vkCreateShaderModule(veekay::app.vk_device, &
	                         info, nullptr, &result) != VK_SUCCESS) {
		return nullptr;
	}

        return result;
}

veekay::graphics::Texture* loadTextureFromFile(VkCommandBuffer cmd, const char* path) {
        std::vector<uint8_t> image;
        unsigned width = 0;
        unsigned height = 0;

        unsigned error = lodepng::decode(image, width, height, path);
        if (error != 0 || image.empty()) {
                std::cerr << "Failed to load texture from " << path << " error " << error << "\n";
                return nullptr;
        }

        return new veekay::graphics::Texture(cmd, width, height, VK_FORMAT_R8G8B8A8_UNORM, image.data());
}

void initialize(VkCommandBuffer cmd) {
	VkDevice& device = veekay::app.vk_device;
	VkPhysicalDevice& physical_device = veekay::app.vk_physical_device;

	{ // NOTE: Build graphics pipeline
		vertex_shader_module = loadShaderModule("./shaders/shader.vert.spv");
		if (!vertex_shader_module) {
			std::cerr << "Failed to load Vulkan vertex shader from file\n";
			veekay::app.running = false;
			return;
		}

		fragment_shader_module = loadShaderModule("./shaders/shader.frag.spv");
		if (!fragment_shader_module) {
			std::cerr << "Failed to load Vulkan fragment shader from file\n";
			veekay::app.running = false;
			return;
		}

		VkPipelineShaderStageCreateInfo stage_infos[2];

		// NOTE: Vertex shader stage
		stage_infos[0] = VkPipelineShaderStageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertex_shader_module,
			.pName = "main",
		};

		// NOTE: Fragment shader stage
		stage_infos[1] = VkPipelineShaderStageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragment_shader_module,
			.pName = "main",
		};

		// NOTE: How many bytes does a vertex take?
		VkVertexInputBindingDescription buffer_binding{
			.binding = 0,
			.stride = sizeof(Vertex),
			.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
		};

		// NOTE: Declare vertex attributes
		VkVertexInputAttributeDescription attributes[] = {
			{
				.location = 0, // NOTE: First attribute
				.binding = 0, // NOTE: First vertex buffer
				.format = VK_FORMAT_R32G32B32_SFLOAT, // NOTE: 3-component vector of floats
				.offset = offsetof(Vertex, position), // NOTE: Offset of "position" field in a Vertex struct
			},
			{
				.location = 1,
				.binding = 0,
				.format = VK_FORMAT_R32G32B32_SFLOAT,
				.offset = offsetof(Vertex, normal),
			},
			{
				.location = 2,
				.binding = 0,
				.format = VK_FORMAT_R32G32_SFLOAT,
				.offset = offsetof(Vertex, uv),
			},
		};

		// NOTE: Describe inputs
		VkPipelineVertexInputStateCreateInfo input_state_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
			.vertexBindingDescriptionCount = 1,
			.pVertexBindingDescriptions = &buffer_binding,
			.vertexAttributeDescriptionCount = sizeof(attributes) / sizeof(attributes[0]),
			.pVertexAttributeDescriptions = attributes,
		};

		// NOTE: Every three vertices make up a triangle,
		//       so our vertex buffer contains a "list of triangles"
		VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		};

		// NOTE: Declare clockwise triangle order as front-facing
		//       Discard triangles that are facing away
		//       Fill triangles, don't draw lines instaed
                VkPipelineRasterizationStateCreateInfo raster_info{
                        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                        .polygonMode = VK_POLYGON_MODE_FILL,
                        .cullMode = VK_CULL_MODE_NONE,
                        .frontFace = VK_FRONT_FACE_CLOCKWISE,
                        .lineWidth = 1.0f,
                };

		// NOTE: Use 1 sample per pixel
		VkPipelineMultisampleStateCreateInfo sample_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
			.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
			.sampleShadingEnable = false,
			.minSampleShading = 1.0f,
		};

		VkViewport viewport{
			.x = 0.0f,
			.y = 0.0f,
			.width = static_cast<float>(veekay::app.window_width),
			.height = static_cast<float>(veekay::app.window_height),
			.minDepth = 0.0f,
			.maxDepth = 1.0f,
		};

		VkRect2D scissor{
			.offset = {0, 0},
			.extent = {veekay::app.window_width, veekay::app.window_height},
		};

		// NOTE: Let rasterizer draw on the entire window
		VkPipelineViewportStateCreateInfo viewport_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,

			.viewportCount = 1,
			.pViewports = &viewport,

			.scissorCount = 1,
			.pScissors = &scissor,
		};

		// NOTE: Let rasterizer perform depth-testing and overwrite depth values on condition pass
		VkPipelineDepthStencilStateCreateInfo depth_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
			.depthTestEnable = true,
			.depthWriteEnable = true,
			.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
		};

		// NOTE: Let fragment shader write all the color channels
		VkPipelineColorBlendAttachmentState attachment_info{
			.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
			                  VK_COLOR_COMPONENT_G_BIT |
			                  VK_COLOR_COMPONENT_B_BIT |
			                  VK_COLOR_COMPONENT_A_BIT,
		};

		// NOTE: Let rasterizer just copy resulting pixels onto a buffer, don't blend
		VkPipelineColorBlendStateCreateInfo blend_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,

			.logicOpEnable = false,
			.logicOp = VK_LOGIC_OP_COPY,

			.attachmentCount = 1,
			.pAttachments = &attachment_info
		};

		{
			VkDescriptorPoolSize pools[] = {
                                {
                                        .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                        .descriptorCount = max_materials,
                                },
                                {
                                        .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                                        .descriptorCount = max_materials,
                                },
                                {
                                        .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                        .descriptorCount = max_materials,
                                }
                        };

                        VkDescriptorPoolCreateInfo info{
                                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                .maxSets = max_materials,
                                .poolSizeCount = sizeof(pools) / sizeof(pools[0]),
                                .pPoolSizes = pools,
                        };

			if (vkCreateDescriptorPool(device, &info, nullptr,
			                           &descriptor_pool) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan descriptor pool\n";
				veekay::app.running = false;
				return;
			}
		}

		// NOTE: Descriptor set layout specification
		{
			VkDescriptorSetLayoutBinding bindings[] = {
				{
					.binding = 0,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
				},
                                {
                                        .binding = 1,
                                        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                                        .descriptorCount = 1,
                                        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                },
                                {
                                        .binding = 2,
                                        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                        .descriptorCount = 1,
                                        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
                                },
                        };

			VkDescriptorSetLayoutCreateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
				.bindingCount = sizeof(bindings) / sizeof(bindings[0]),
				.pBindings = bindings,
			};

			if (vkCreateDescriptorSetLayout(device, &info, nullptr,
			                                &descriptor_set_layout) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan descriptor set layout\n";
				veekay::app.running = false;
				return;
			}
		}

                // NOTE: Declare external data sources, only push constants this time
                VkPipelineLayoutCreateInfo layout_info{
                        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                        .setLayoutCount = 1,
			.pSetLayouts = &descriptor_set_layout,
		};

		// NOTE: Create pipeline layout
		if (vkCreatePipelineLayout(device, &layout_info,
		                           nullptr, &pipeline_layout) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan pipeline layout\n";
			veekay::app.running = false;
			return;
		}
		
		VkGraphicsPipelineCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
			.stageCount = 2,
			.pStages = stage_infos,
			.pVertexInputState = &input_state_info,
			.pInputAssemblyState = &assembly_state_info,
			.pViewportState = &viewport_info,
			.pRasterizationState = &raster_info,
			.pMultisampleState = &sample_info,
			.pDepthStencilState = &depth_info,
			.pColorBlendState = &blend_info,
			.layout = pipeline_layout,
			.renderPass = veekay::app.vk_render_pass,
		};

		// NOTE: Create graphics pipeline
		if (vkCreateGraphicsPipelines(device, nullptr,
		                              1, &info, nullptr, &pipeline) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan pipeline\n";
			veekay::app.running = false;
			return;
		}
	}

	scene_uniforms_buffer = new veekay::graphics::Buffer(
		sizeof(SceneUniforms),
		nullptr,
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

        model_uniforms_buffer = new veekay::graphics::Buffer(
                max_models * veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms)),
                nullptr,
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

        // NOTE: This texture and sampler is used when texture could not be loaded
        {
                VkSamplerCreateInfo info{
                        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                };

		if (vkCreateSampler(device, &info, nullptr, &missing_texture_sampler) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan texture sampler\n";
			veekay::app.running = false;
			return;
		}

		uint32_t pixels[] = {
			0xff000000, 0xffff00ff,
			0xffff00ff, 0xff000000,
		};

                missing_texture = new veekay::graphics::Texture(cmd, 2, 2,
                                                                VK_FORMAT_B8G8R8A8_UNORM,
                                                                pixels);
        }

        { // NOTE: Flat sampler and white texture for untextured surfaces
                VkSamplerCreateInfo info{
                        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                        .magFilter = VK_FILTER_LINEAR,
                        .minFilter = VK_FILTER_LINEAR,
                        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
                        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                };

                if (vkCreateSampler(device, &info, nullptr, &flat_sampler) != VK_SUCCESS) {
                        std::cerr << "Failed to create Vulkan flat sampler\n";
                        veekay::app.running = false;
                        return;
                }

                uint32_t pixel = 0xffffffff;
                white_texture = new veekay::graphics::Texture(cmd, 1, 1,
                                                               VK_FORMAT_B8G8R8A8_UNORM,
                                                               &pixel);
        }

        { // NOTE: Load image texture and create an additional sampler
                VkSamplerCreateInfo info{
                        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                        .magFilter = VK_FILTER_LINEAR,
                        .minFilter = VK_FILTER_LINEAR,
                        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
                        .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                        .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                        .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                };

                if (vkCreateSampler(device, &info, nullptr, &texture_sampler) != VK_SUCCESS) {
                        std::cerr << "Failed to create Vulkan texture sampler\n";
                        veekay::app.running = false;
                        return;
                }

                texture = loadTextureFromFile(cmd, "./assets/lenna.png");
                if (!texture) {
                        vkDestroySampler(device, texture_sampler, nullptr);
                        texture = missing_texture;
                        texture_sampler = missing_texture_sampler;
                }
        }

        materials.push_back(Material{
                .ambient_color = veekay::vec3{0.12f, 0.08f, 0.04f},
                .diffuse_color = veekay::vec3{1.0f, 0.6f, 0.2f},
                .specular_color = veekay::vec3{0.9f, 0.85f, 0.8f},
                .shininess = 24.0f,
                .sampler = texture_sampler,
                .texture = texture,
        });

        materials.push_back(Material{
                .ambient_color = veekay::vec3{0.08f, 0.12f, 0.2f},
                .diffuse_color = veekay::vec3{0.3f, 0.8f, 1.0f},
                .specular_color = veekay::vec3{0.95f, 0.95f, 0.95f},
                .shininess = 48.0f,
                .sampler = missing_texture_sampler,
                .texture = missing_texture,
        });

        materials.push_back(Material{
                .ambient_color = veekay::vec3{0.12f, 0.12f, 0.12f},
                .diffuse_color = veekay::vec3{0.4f, 0.4f, 0.4f},
                .specular_color = veekay::vec3{0.0f, 0.0f, 0.0f},
                .shininess = 8.0f,
                .sampler = flat_sampler,
                .texture = white_texture,
        });

        materials.push_back(Material{
                .ambient_color = veekay::vec3{0.0f, 0.0f, 0.0f},
                .diffuse_color = veekay::vec3{0.15f, 0.15f, 0.15f},
                .specular_color = veekay::vec3{0.0f, 0.0f, 0.0f},
                .shininess = 2.0f,
                .sampler = flat_sampler,
                .texture = white_texture,
        });

        if (materials.size() > max_materials) {
                std::cerr << "Too many materials for descriptor pool" << std::endl;
                veekay::app.running = false;
                return;
        }

        {
                std::vector<VkDescriptorSetLayout> layouts(materials.size(), descriptor_set_layout);
                VkDescriptorSetAllocateInfo info{
                        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                        .descriptorPool = descriptor_pool,
                        .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
                        .pSetLayouts = layouts.data(),
                };

                std::vector<VkDescriptorSet> descriptor_sets(materials.size());
                if (vkAllocateDescriptorSets(device, &info, descriptor_sets.data()) != VK_SUCCESS) {
                        std::cerr << "Failed to create Vulkan descriptor set\n";
                        veekay::app.running = false;
                        return;
                }

                VkDescriptorBufferInfo buffer_infos[] = {
                        {
                                .buffer = scene_uniforms_buffer->buffer,
                                .offset = 0,
                                .range = sizeof(SceneUniforms),
                        },
                        {
                                .buffer = model_uniforms_buffer->buffer,
                                .offset = 0,
                                .range = sizeof(ModelUniforms),
                        },
                };

                for (size_t i = 0; i < materials.size(); ++i) {
                        materials[i].descriptor_set = descriptor_sets[i];
                        VkDescriptorImageInfo image_info{
                                .sampler = materials[i].sampler,
                                .imageView = materials[i].texture->view,
                                .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        };

                        VkWriteDescriptorSet write_infos[] = {
                                {
                                        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                        .dstSet = materials[i].descriptor_set,
                                        .dstBinding = 0,
                                        .dstArrayElement = 0,
                                        .descriptorCount = 1,
                                        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                        .pBufferInfo = &buffer_infos[0],
                                },
                                {
                                        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                        .dstSet = materials[i].descriptor_set,
                                        .dstBinding = 1,
                                        .dstArrayElement = 0,
                                        .descriptorCount = 1,
                                        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                                        .pBufferInfo = &buffer_infos[1],
                                },
                                {
                                        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                        .dstSet = materials[i].descriptor_set,
                                        .dstBinding = 2,
                                        .dstArrayElement = 0,
                                        .descriptorCount = 1,
                                        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                        .pImageInfo = &image_info,
                                },
                        };

                        vkUpdateDescriptorSets(device,
                                               sizeof(write_infos) / sizeof(write_infos[0]),
                                               write_infos, 0, nullptr);
                }
        }

        // NOTE: Plane mesh initialization
        {
                std::vector<Vertex> vertices = {
                        {{-5.0f, 0.0f, -5.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
                        {{5.0f, 0.0f, -5.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
                        {{5.0f, 0.0f, 5.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
                        {{-5.0f, 0.0f, 5.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},
                };

                std::vector<uint32_t> indices = {
                        0, 1, 2,
                        2, 3, 0,
                };

                plane_mesh.vertex_buffer = new veekay::graphics::Buffer(
                        vertices.size() * sizeof(Vertex), vertices.data(),
                        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

                plane_mesh.index_buffer = new veekay::graphics::Buffer(
                        indices.size() * sizeof(uint32_t), indices.data(),
                        VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

                plane_mesh.indices = uint32_t(indices.size());
        }

        // NOTE: Cone mesh initialization
        {
                const float radius = 0.5f;
                const float height = 1.0f;
                const uint32_t segments = 32;

                std::vector<Vertex> vertices;
                std::vector<uint32_t> indices;

                // Base center
                vertices.push_back(Vertex{{0.0f, 0.0f, 0.0f}, {0.0f, -1.0f, 0.0f}, {0.5f, 0.5f}});

                // Base ring vertices
                for (uint32_t i = 0; i < segments; ++i) {
                        float angle = (float(i) / float(segments)) * 2.0f * float(M_PI);
                        float x = radius * cosf(angle);
                        float z = radius * sinf(angle);

                        vertices.push_back(Vertex{{x, 0.0f, z}, {0.0f, -1.0f, 0.0f}, {0.5f + x, 0.5f + z}});
                }

                // Base indices (triangle fan)
                for (uint32_t i = 0; i < segments; ++i) {
                        uint32_t next = (i + 1) % segments;
                        indices.push_back(0);
                        indices.push_back(1 + next);
                        indices.push_back(1 + i);
                }

                const veekay::vec3 apex{0.0f, height, 0.0f};

                for (uint32_t i = 0; i < segments; ++i) {
                        uint32_t next = (i + 1) % segments;

                        veekay::vec3 b0 = vertices[1 + i].position;
                        veekay::vec3 b1 = vertices[1 + next].position;

                        veekay::vec3 edge0 = b0 - apex;
                        veekay::vec3 edge1 = b1 - apex;
                        veekay::vec3 normal = veekay::vec3::normalized(veekay::vec3::cross(edge1, edge0));

                        uint32_t start = static_cast<uint32_t>(vertices.size());
                        vertices.push_back(Vertex{apex, normal, {0.5f, 1.0f}});
                        vertices.push_back(Vertex{b0, normal, {0.0f, 0.0f}});
                        vertices.push_back(Vertex{b1, normal, {1.0f, 0.0f}});

                        indices.push_back(start);
                        indices.push_back(start + 1);
                        indices.push_back(start + 2);
                }

                cone_mesh.vertex_buffer = new veekay::graphics::Buffer(
                        vertices.size() * sizeof(Vertex), vertices.data(),
                        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

                cone_mesh.index_buffer = new veekay::graphics::Buffer(
                        indices.size() * sizeof(uint32_t), indices.data(),
                        VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

                cone_mesh.indices = uint32_t(indices.size());
        }

        // NOTE: Add models to scene
        models.emplace_back(Model{
                .mesh = plane_mesh,
                .transform = Transform{
                        .position = {0.0f, ground_height, 0.0f},
                        .scale = {1.0f, 1.0f, 1.0f},
                },
                .angular_speed = 0.0f,
                .rotation_axis = {0.0f, 1.0f, 0.0f},
                .material = &materials[2],
        });

        const size_t first_cone_index = models.size();

        models.emplace_back(Model{
                .mesh = cone_mesh,
                .transform = Transform{
                        .position = {-1.5f, ground_height, -2.0f},
                        .scale = {0.8f, 0.8f, 0.8f},
                },
                .angular_speed = 0.0f,
                .rotation_axis = {0.0f, 1.0f, 0.0f},
                .material = &materials[0],
        });

        models.emplace_back(Model{
                .mesh = cone_mesh,
                .transform = Transform{
                        .position = {1.2f, ground_height, -0.5f},
                        .scale = {1.2f, 1.2f, 1.2f},
                },
                .angular_speed = 0.0f,
                .rotation_axis = {0.0f, 1.0f, 0.0f},
                .material = &materials[1],
        });

        models.emplace_back(Model{
                .mesh = cone_mesh,
                .transform = Transform{
                        .position = {0.0f, ground_height, 1.2f},
                        .scale = {0.6f, 0.6f, 0.6f},
                },
                .angular_speed = 0.0f,
                .rotation_axis = {0.0f, 1.0f, 0.0f},
                .material = &materials[0],
        });

        for (size_t i = 0; i < 3; ++i) {
                const size_t source_index = first_cone_index + i;
                Model shadow = models[source_index];
                shadow.material = &materials[3];
                shadow.angular_speed = 0.0f;
                shadow.use_override = true;
                shadow.override_model = shadowMatrixForModel(models[source_index],
                                                             veekay::vec3{-lighting.directional_light.direction_intensity.x,
                                                                          -lighting.directional_light.direction_intensity.y,
                                                                          -lighting.directional_light.direction_intensity.z});

                const size_t shadow_index = models.size();
                models.emplace_back(shadow);
                shadow_instances.push_back({source_index, shadow_index});
        }

        float aspect_ratio = float(veekay::app.window_width) / float(veekay::app.window_height);
        SceneUniforms scene_uniforms{};
        scene_uniforms.view_projection = camera.view_projection(aspect_ratio);
        scene_uniforms.camera_position = veekay::vec4{camera.position.x, camera.position.y, camera.position.z, 1.0f};
        scene_uniforms.ambient_color = lighting.ambient_color;
        scene_uniforms.diffuse_color = lighting.diffuse_color;

        int mode_index = static_cast<int>(light_mode);
        scene_uniforms.light_mode = veekay::vec4{static_cast<float>(mode_index), 0.0f, 0.0f, 0.0f};
        scene_uniforms.directional_light = lighting.directional_light;

        uint32_t active_point_lights = (light_mode == LightMode::Point) ? lighting.point_light_count : 0u;
        scene_uniforms.point_light_count = veekay::vec4{static_cast<float>(active_point_lights), 0.0f, 0.0f, 0.0f};
        for (uint32_t i = 0; i < max_point_lights; ++i) {
                scene_uniforms.point_lights[i] = lighting.point_lights[i];
        }

        *(SceneUniforms*)scene_uniforms_buffer->mapped_region = scene_uniforms;

        std::vector<ModelUniforms> model_uniforms(models.size());
        for (size_t i = 0, n = models.size(); i < n; ++i) {
                const Model& model = models[i];
                ModelUniforms& uniforms = model_uniforms[i];

                uniforms.model = model.transform.matrix();
                uniforms.ambient_color = veekay::vec4{
                        model.material->ambient_color.x,
                        model.material->ambient_color.y,
                        model.material->ambient_color.z,
                        1.0f
                };
                uniforms.diffuse_color = veekay::vec4{
                        model.material->diffuse_color.x,
                        model.material->diffuse_color.y,
                        model.material->diffuse_color.z,
                        1.0f
                };
                uniforms.specular_color_shininess = veekay::vec4{
                        model.material->specular_color.x,
                        model.material->specular_color.y,
                        model.material->specular_color.z,
                        model.material->shininess
                };
        }

        const size_t alignment =
                veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

        for (size_t i = 0, n = model_uniforms.size(); i < n; ++i) {
                const ModelUniforms& uniforms = model_uniforms[i];

                char* const pointer = static_cast<char*>(model_uniforms_buffer->mapped_region) + i * alignment;
                *reinterpret_cast<ModelUniforms*>(pointer) = uniforms;
        }
}

// NOTE: Destroy resources here, do not cause leaks in your program!
void shutdown() {
        VkDevice& device = veekay::app.vk_device;

        if (texture_sampler != missing_texture_sampler && texture_sampler != VK_NULL_HANDLE) {
                vkDestroySampler(device, texture_sampler, nullptr);
        }
        vkDestroySampler(device, missing_texture_sampler, nullptr);

        if (texture && texture != missing_texture) {
                delete texture;
        }
        delete white_texture;
        delete missing_texture;

        delete cone_mesh.index_buffer;
        delete cone_mesh.vertex_buffer;

        delete plane_mesh.index_buffer;
        delete plane_mesh.vertex_buffer;

        delete model_uniforms_buffer;
        delete scene_uniforms_buffer;

        vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
        vkDestroyDescriptorPool(device, descriptor_pool, nullptr);

        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
        vkDestroyShaderModule(device, fragment_shader_module, nullptr);
        vkDestroyShaderModule(device, vertex_shader_module, nullptr);

        vkDestroySampler(device, flat_sampler, nullptr);
}

void update(double time) {
        ImGui::Begin("Controls:");
        ImGui::TextUnformatted("Camera: hold RMB + move mouse, WASD to move, Q/E to go down/up");
        ImGui::SliderFloat("Camera move speed", &camera_move_speed, 0.5f, 15.0f, "%.1f u/s");
        ImGui::SliderFloat("Mouse sensitivity", &camera_sensitivity, 0.01f, 0.25f, "%.3f");
        ImGui::ColorEdit3("Ambient color", lighting.ambient_color.elements);

        int mode_index = static_cast<int>(light_mode);
        const char* light_modes[] = {"Diffuse", "Directional", "Point"};
        if (ImGui::Combo("Light type", &mode_index, light_modes, IM_ARRAYSIZE(light_modes))) {
                light_mode = static_cast<LightMode>(mode_index);
        }

        if (light_mode == LightMode::Diffuse) {
                ImGui::ColorEdit3("Diffuse color", lighting.diffuse_color.elements);
                ImGui::SliderFloat("Diffuse intensity", &lighting.diffuse_color.w, 0.0f, 2.5f, "%.2f");
        } else if (light_mode == LightMode::Directional) {
                ImGui::SliderFloat("Directional intensity", &lighting.directional_light.direction_intensity.w, 0.0f, 3.0f, "%.2f");

                veekay::vec3 directional_dir{
                        lighting.directional_light.direction_intensity.x,
                        lighting.directional_light.direction_intensity.y,
                        lighting.directional_light.direction_intensity.z
                };
                if (ImGui::SliderFloat3("Directional direction", directional_dir.elements, -1.0f, 1.0f)) {
                        float len = veekay::vec3::length(directional_dir);
                        if (len > std::numeric_limits<float>::epsilon()) {
                                directional_dir = directional_dir / len;
                        }

                        lighting.directional_light.direction_intensity.x = directional_dir.x;
                        lighting.directional_light.direction_intensity.y = directional_dir.y;
                        lighting.directional_light.direction_intensity.z = directional_dir.z;
                }
        } else if (light_mode == LightMode::Point) {
                ImGui::ColorEdit3("Point light color", lighting.point_lights[0].color.elements);
                ImGui::SliderFloat("Point light intensity", &lighting.point_lights[0].position_intensity.w, 0.0f, 40.0f, "%.1f");
        }

        ImGui::SeparatorText("Materials");
        for (size_t i = 0; i < materials.size(); ++i) {
                ImGui::PushID(static_cast<int>(i));
                std::string label = "Material " + std::to_string(i + 1);
                ImGui::SeparatorText(label.c_str());
                ImGui::ColorEdit3("Ambient", materials[i].ambient_color.elements);
                ImGui::ColorEdit3("Diffuse", materials[i].diffuse_color.elements);
                ImGui::ColorEdit3("Specular", materials[i].specular_color.elements);
                ImGui::SliderFloat("Shininess", &materials[i].shininess, 1.0f, 128.0f, "%.0f");
                ImGui::PopID();
        }
        ImGui::End();

        static double previous_time = time;
        float delta_time = static_cast<float>(time - previous_time);
        previous_time = time;

        const bool right_mouse_down = veekay::input::mouse::isButtonDown(veekay::input::mouse::Button::right);
        if (right_mouse_down && !mouse_captured) {
                veekay::input::mouse::setCaptured(true);
                mouse_captured = true;
                veekay::input::mouse::cursorDelta();
        } else if (!right_mouse_down && mouse_captured) {
                veekay::input::mouse::setCaptured(false);
                mouse_captured = false;
        }

        if (mouse_captured) {
                veekay::vec2 delta = veekay::input::mouse::cursorDelta();
                camera.rotation.x -= delta.y * camera_sensitivity;
                camera.rotation.y += delta.x * camera_sensitivity;
                camera.rotation.x = std::clamp(camera.rotation.x, -89.0f, 89.0f);
        }

        veekay::vec3 movement{};
        const veekay::vec3 forward = forwardFromRotation(camera.rotation);
        const veekay::vec3 right = rightFromRotation(camera.rotation);
        veekay::vec3 up = veekay::vec3::cross(forward, right);
        float up_length = veekay::vec3::length(up);
        if (up_length > std::numeric_limits<float>::epsilon()) {
                up = up / up_length;
        } else {
                up = {0.0f, 1.0f, 0.0f};
        }

        using veekay::input::keyboard::isKeyDown;
        using veekay::input::keyboard::Key;

        if (isKeyDown(Key::w)) movement += forward;
        if (isKeyDown(Key::s)) movement -= forward;
        if (isKeyDown(Key::d)) movement += right;
        if (isKeyDown(Key::a)) movement -= right;
        if (isKeyDown(Key::e)) movement += up;
        if (isKeyDown(Key::q)) movement -= up;

        float movement_length = veekay::vec3::length(movement);
        if (movement_length > std::numeric_limits<float>::epsilon()) {
                movement = movement / movement_length;
        }

        float move_speed = camera_move_speed;
        if (isKeyDown(Key::left_shift)) {
                move_speed *= 2.5f;
        }
        if (isKeyDown(Key::left_control)) {
                move_speed *= 0.5f;
        }

        camera.position += movement * (move_speed * delta_time);

        float aspect_ratio = float(veekay::app.window_width) / float(veekay::app.window_height);
        SceneUniforms scene_uniforms{};
        scene_uniforms.view_projection = camera.view_projection(aspect_ratio);
        scene_uniforms.camera_position = veekay::vec4{camera.position.x, camera.position.y, camera.position.z, 1.0f};
        scene_uniforms.ambient_color = lighting.ambient_color;
        scene_uniforms.diffuse_color = lighting.diffuse_color;
        scene_uniforms.light_mode = veekay::vec4{static_cast<float>(mode_index), 0.0f, 0.0f, 0.0f};
        scene_uniforms.directional_light = lighting.directional_light;

        uint32_t active_point_lights = (light_mode == LightMode::Point) ? lighting.point_light_count : 0u;
        scene_uniforms.point_light_count = veekay::vec4{static_cast<float>(active_point_lights), 0.0f, 0.0f, 0.0f};
        for (uint32_t i = 0; i < max_point_lights; ++i) {
                scene_uniforms.point_lights[i] = lighting.point_lights[i];
        }

        veekay::vec3 shadow_light_dir{
                -lighting.directional_light.direction_intensity.x,
                -lighting.directional_light.direction_intensity.y,
                -lighting.directional_light.direction_intensity.z,
        };

        for (const ShadowInstance& instance : shadow_instances) {
                const Model& source = models[instance.source_index];
                Model& shadow = models[instance.shadow_index];
                shadow.transform = source.transform;
                shadow.override_model = shadowMatrixForModel(source, shadow_light_dir);
        }

        std::vector<ModelUniforms> model_uniforms(models.size());
        for (size_t i = 0, n = models.size(); i < n; ++i) {
                const Model& model = models[i];
                ModelUniforms& uniforms = model_uniforms[i];

                uniforms.model = model.use_override ? model.override_model : model.transform.matrix();
                uniforms.ambient_color = veekay::vec4{
                        model.material->ambient_color.x,
                        model.material->ambient_color.y,
                        model.material->ambient_color.z,
                        1.0f
                };
                uniforms.diffuse_color = veekay::vec4{
                        model.material->diffuse_color.x,
                        model.material->diffuse_color.y,
                        model.material->diffuse_color.z,
                        1.0f
                };
                uniforms.specular_color_shininess = veekay::vec4{
                        model.material->specular_color.x,
                        model.material->specular_color.y,
                        model.material->specular_color.z,
                        model.material->shininess
                };
        }

        *(SceneUniforms*)scene_uniforms_buffer->mapped_region = scene_uniforms;

        const size_t alignment =
                veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

        for (size_t i = 0, n = model_uniforms.size(); i < n; ++i) {
                const ModelUniforms& uniforms = model_uniforms[i];

                char* const pointer = static_cast<char*>(model_uniforms_buffer->mapped_region) + i * alignment;
                *reinterpret_cast<ModelUniforms*>(pointer) = uniforms;
        }
}

void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
	vkResetCommandBuffer(cmd, 0);

	{ // NOTE: Start recording rendering commands
		VkCommandBufferBeginInfo info{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		};

		vkBeginCommandBuffer(cmd, &info);
	}

	{ // NOTE: Use current swapchain framebuffer and clear it
		VkClearValue clear_color{.color = {{0.1f, 0.1f, 0.1f, 1.0f}}};
		VkClearValue clear_depth{.depthStencil = {1.0f, 0}};

		VkClearValue clear_values[] = {clear_color, clear_depth};

		VkRenderPassBeginInfo info{
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass = veekay::app.vk_render_pass,
			.framebuffer = framebuffer,
			.renderArea = {
				.extent = {
					veekay::app.window_width,
					veekay::app.window_height
				},
			},
			.clearValueCount = 2,
			.pClearValues = clear_values,
		};

		vkCmdBeginRenderPass(cmd, &info, VK_SUBPASS_CONTENTS_INLINE);
	}

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
	VkDeviceSize zero_offset = 0;

	VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
	VkBuffer current_index_buffer = VK_NULL_HANDLE;

	const size_t model_uniorms_alignment =
		veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

	for (size_t i = 0, n = models.size(); i < n; ++i) {
		const Model& model = models[i];
		const Mesh& mesh = model.mesh;

		if (current_vertex_buffer != mesh.vertex_buffer->buffer) {
			current_vertex_buffer = mesh.vertex_buffer->buffer;
			vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex_buffer, &zero_offset);
		}

                if (current_index_buffer != mesh.index_buffer->buffer) {
                        current_index_buffer = mesh.index_buffer->buffer;
                        vkCmdBindIndexBuffer(cmd, current_index_buffer, zero_offset, VK_INDEX_TYPE_UINT32);
                }

                uint32_t offset = i * model_uniorms_alignment;
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
                                        0, 1, &model.material->descriptor_set, 1, &offset);

                vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
        }

	vkCmdEndRenderPass(cmd);
	vkEndCommandBuffer(cmd);
}

} // namespace

int main() {
	return veekay::run({
		.init = initialize,
		.shutdown = shutdown,
		.update = update,
		.render = render,
	});
}
