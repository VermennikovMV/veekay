#include <cstdint>
#include <climits>
#include <cstring>
#include <vector>
#include <array>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <string>
#include <limits>

#include <veekay/veekay.hpp>

#include <vulkan/vulkan_core.h>
#include <imgui.h>
#include <lodepng.h>

namespace {

constexpr uint32_t max_models = 1024;
constexpr uint32_t max_point_lights = 16;
constexpr uint32_t max_spot_lights = 8;
constexpr uint32_t max_shadow_spot_lights = 2;
constexpr veekay::vec3 world_up{0.0f, 1.0f, 0.0f};

struct Vertex {
  veekay::vec3 position;
  veekay::vec3 normal;
  veekay::vec2 uv;
  // NOTE: You can add more attributes
};

struct SceneUniforms {
  veekay::mat4 view_projection;
  veekay::mat4 shadow_view_projection;
  veekay::vec4 camera_position; // w unused
  veekay::vec4 ambient_color; // rgb * intensity
  veekay::vec4 directional_direction_intensity; // xyz dir, w intensity
  veekay::vec4 directional_color;
  veekay::vec4 light_counts; // x - points, y - spots
  veekay::mat4 spot_shadow_view_projections[max_shadow_spot_lights];
  veekay::vec4 spot_shadow_indices; // x/y store indices of spot lights casting shadows
};

struct ModelUniforms {
  veekay::mat4 model;
  veekay::vec4 albedo_shininess; // rgb + shininess
  veekay::vec4 specular_color; // rgb + padding
};

struct Mesh {
  veekay::graphics::Buffer* vertex_buffer;
  veekay::graphics::Buffer* index_buffer;
  uint32_t indices;
  veekay::vec3 bounds_min{};
  veekay::vec3 bounds_max{};
  bool has_bounds = false;
};

struct Transform {
  veekay::vec3 position = {};
  veekay::vec3 scale = {1.0f, 1.0f, 1.0f};
  veekay::vec3 rotation = {};

  // NOTE: Model matrix (translation, rotation and scaling)
  veekay::mat4 matrix() const;
};

struct Material {
  veekay::vec3 albedo_color{1.0f, 1.0f, 1.0f};
  float shininess = 32.0f;
  veekay::vec3 specular_color{1.0f, 1.0f, 1.0f};
  float _pad = 0.0f;
};

struct Model {
  Mesh mesh;
  Transform transform;
  Material material;
  VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
};

VkSampler texture_sampler;
std::vector<veekay::graphics::Texture*> textures;

veekay::graphics::Texture* load_texture(VkCommandBuffer cmd, const char* filename) {
  std::vector<unsigned char> image;
  unsigned width, height;
  unsigned error = lodepng::decode(image, width, height, filename);
  if (error) {
    std::cerr << "Lodepng error " << error << ": " << lodepng_error_text(error) << std::endl;
    // Create a fallback checkerboard texture
    width = 64;
    height = 64;
    image.resize(width * height * 4);
    for (unsigned y = 0; y < height; ++y) {
      for (unsigned x = 0; x < width; ++x) {
        bool white = ((x / 8) + (y / 8)) % 2 == 0;
        unsigned char c = white ? 255 : 0;
        size_t index = (y * width + x) * 4;
        image[index + 0] = c;
        image[index + 1] = c;
        image[index + 2] = c;
        image[index + 3] = 255;
      }
    }
  }
  return new veekay::graphics::Texture(cmd, width, height, VK_FORMAT_R8G8B8A8_UNORM, image.data());
}

struct AmbientLight {
  veekay::vec3 color{0.2f, 0.2f, 0.2f};
  float intensity = 0.25f;
};

struct DirectionalLight {
  veekay::vec3 direction{-0.2f, -1.0f, -0.3f};
  float intensity = 1.0f;
  veekay::vec3 color{1.0f, 1.0f, 1.0f};
  float _pad = 0.0f;
};

struct PointLight {
  veekay::vec3 position{};
  float intensity = 15.0f;
  veekay::vec3 color{1.0f, 0.95f, 0.8f};
  float radius = 10.0f;
};

struct PointLightGpu {
  veekay::vec4 position_intensity;
  veekay::vec4 color_radius;
};

struct SpotLight {
  veekay::vec3 position{};
  float intensity = 18.0f;
  veekay::vec3 direction{0.0f, -1.0f, 0.0f};
  float inner_angle = 20.0f;
  veekay::vec3 color{1.0f, 0.95f, 0.8f};
  float outer_angle = 32.0f;
  float range = 8.0f;
  float _pad = 0.0f;
};

struct SpotLightGpu {
  veekay::vec4 position_intensity;
  veekay::vec4 direction_inner_cos;
  veekay::vec4 color_range;
  veekay::vec4 params; // x stores cos_outer
};

struct ShadowPass {
  static constexpr uint32_t map_size = 2048;

  veekay::mat4 view_projection{};

VkFormat depth_format = VK_FORMAT_UNDEFINED;
  VkImage depth_image = VK_NULL_HANDLE;
  VkDeviceMemory depth_memory = VK_NULL_HANDLE;
  VkImageView depth_view = VK_NULL_HANDLE;
  VkImageLayout image_layout = VK_IMAGE_LAYOUT_UNDEFINED;

  VkSampler sampler = VK_NULL_HANDLE;

  VkShaderModule vertex_shader_module = VK_NULL_HANDLE;
  VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
  VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
  VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;

  veekay::graphics::Buffer* uniform_buffer = nullptr;
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

  struct Axes {
    veekay::vec3 front;
    veekay::vec3 right;
    veekay::vec3 up;
  };

  Axes axes() const;

  // NOTE: View matrix of camera (inverse of a transform)
  veekay::mat4 view() const;

  // NOTE: View and projection composition
  veekay::mat4 view_projection(float aspect_ratio) const;
};

enum class CameraMode : uint32_t {
  FreeFly = 0,
  Orbit = 1,
};

struct FreeFlyState {
  veekay::vec3 position{0.0f, 1.5f, -3.0f};
  veekay::vec3 rotation{};
};

struct OrbitState {
  veekay::vec3 target{0.0f, 0.0f, 0.0f};
  float distance = 5.0f;
  float yaw = 0.0f;
  float pitch = -15.0f;
};

void applyCameraState(CameraMode mode);
void saveCameraState(CameraMode mode);
void captureOrbitStateFromCamera(float desired_distance = -1.0f);
void captureOrbitStateFromCamera(const veekay::vec3& focus_point, float desired_distance = -1.0f);

// NOTE: Scene objects
inline namespace {
  CameraMode camera_mode = CameraMode::FreeFly;
  FreeFlyState free_fly_state{};
  OrbitState orbit_state{};
  Camera camera{
    .position = free_fly_state.position,
    .rotation = free_fly_state.rotation,
  };

  std::vector<Model> models;
  AmbientLight ambient_light{};
  DirectionalLight directional_light{};
  std::vector<PointLight> point_lights{
    // {{-1.5f, 1.5f, -0.5f}, 20.0f, {1.0f, 0.95f, 0.8f}, 8.0f},
    // {{2.0f, 0.5f, 1.5f}, 18.0f, {0.6f, 0.8f, 1.0f}, 7.0f},
    // {{0.5f, 2.5f, -2.5f}, 25.0f, {1.0f, 0.6f, 0.6f}, 10.0f},
  };
  std::vector<SpotLight> spot_lights{
    // {{0.0f, 2.5f, 2.5f}, 22.0f, {-0.2f, -1.0f, -0.3f}, 18.0f, {1.0f, 0.9f, 0.7f}, 30.0f, 9.0f},
  };
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
  veekay::graphics::Buffer* point_lights_buffer;
  veekay::graphics::Buffer* spot_lights_buffer;

  ShadowPass shadow;
  std::array<ShadowPass, max_shadow_spot_lights> spot_shadows;
  size_t active_spot_shadow_count = 0;
  std::array<int32_t, max_shadow_spot_lights> spot_shadow_light_indices{};

  Mesh plane_mesh;
  Mesh cube_mesh;
}

float toRadians(float degrees) {
  return degrees * float(M_PI) / 180.0f;
}

float toDegrees(float radians) {
  return radians * 180.0f / float(M_PI);
}

veekay::mat4 Transform::matrix() const {
  auto t = veekay::mat4::translation(position);
  auto s = veekay::mat4::scaling(scale);

  const auto rx = veekay::mat4::rotation({1.0f, 0.0f, 0.0f}, toRadians(rotation.x));
  const auto ry = veekay::mat4::rotation({0.0f, 1.0f, 0.0f}, toRadians(rotation.y));
  const auto rz = veekay::mat4::rotation({0.0f, 0.0f, 1.0f}, toRadians(rotation.z));
  const auto r = rz * ry * rx;

  return t * r * s;
}

veekay::vec3 transformPoint(const veekay::mat4& matrix, const veekay::vec3& point) {
  const float x = point.x;
  const float y = point.y;
  const float z = point.z;
  return {
    matrix[0][0] * x + matrix[1][0] * y + matrix[2][0] * z + matrix[3][0],
    matrix[0][1] * x + matrix[1][1] * y + matrix[2][1] * z + matrix[3][1],
    matrix[0][2] * x + matrix[1][2] * y + matrix[2][2] * z + matrix[3][2],
  };
}

void computeMeshBounds(Mesh& mesh, const std::vector<Vertex>& vertices) {
  if (vertices.empty()) {
    mesh.has_bounds = false;
    mesh.bounds_min = {};
    mesh.bounds_max = {};
    return;
  }

  veekay::vec3 min_bounds = vertices.front().position;
  veekay::vec3 max_bounds = vertices.front().position;
  for (const Vertex& vertex : vertices) {
    const veekay::vec3& pos = vertex.position;
    min_bounds.x = std::min(min_bounds.x, pos.x);
    min_bounds.y = std::min(min_bounds.y, pos.y);
    min_bounds.z = std::min(min_bounds.z, pos.z);
    max_bounds.x = std::max(max_bounds.x, pos.x);
    max_bounds.y = std::max(max_bounds.y, pos.y);
    max_bounds.z = std::max(max_bounds.z, pos.z);
  }

  mesh.bounds_min = min_bounds;
  mesh.bounds_max = max_bounds;
  mesh.has_bounds = true;
}

bool computeModelAabb(const Model& model, veekay::vec3& out_min, veekay::vec3& out_max) {
  if (!model.mesh.has_bounds)
    return false;

  const veekay::vec3 local_min = model.mesh.bounds_min;
  const veekay::vec3 local_max = model.mesh.bounds_max;
  const veekay::vec3 corners[] = {
    {local_min.x, local_min.y, local_min.z},
    {local_min.x, local_min.y, local_max.z},
    {local_min.x, local_max.y, local_min.z},
    {local_min.x, local_max.y, local_max.z},
    {local_max.x, local_min.y, local_min.z},
    {local_max.x, local_min.y, local_max.z},
    {local_max.x, local_max.y, local_min.z},
    {local_max.x, local_max.y, local_max.z},
  };

  const veekay::mat4 matrix = model.transform.matrix();

  veekay::vec3 min_bounds = transformPoint(matrix, corners[0]);
  veekay::vec3 max_bounds = min_bounds;
  for (size_t i = 1; i < sizeof(corners) / sizeof(corners[0]); ++i) {
    const veekay::vec3 world = transformPoint(matrix, corners[i]);
    min_bounds.x = std::min(min_bounds.x, world.x);
    min_bounds.y = std::min(min_bounds.y, world.y);
    min_bounds.z = std::min(min_bounds.z, world.z);
    max_bounds.x = std::max(max_bounds.x, world.x);
    max_bounds.y = std::max(max_bounds.y, world.y);
    max_bounds.z = std::max(max_bounds.z, world.z);
  }

  out_min = min_bounds;
  out_max = max_bounds;
  return true;
}

bool rayIntersectsAabb(const veekay::vec3& origin, const veekay::vec3& direction,
                       const veekay::vec3& aabb_min, const veekay::vec3& aabb_max, float& out_distance) {
  float t_min = 0.0f;
  float t_max = std::numeric_limits<float>::max();

  for (int axis = 0; axis < 3; ++axis) {
    const float origin_component = axis == 0 ? origin.x : (axis == 1 ? origin.y : origin.z);
    const float dir_component = axis == 0 ? direction.x : (axis == 1 ? direction.y : direction.z);
    const float min_component = axis == 0 ? aabb_min.x : (axis == 1 ? aabb_min.y : aabb_min.z);
    const float max_component = axis == 0 ? aabb_max.x : (axis == 1 ? aabb_max.y : aabb_max.z);

if (std::fabs(dir_component) < 1e-4f) {
      if (origin_component < min_component  origin_component > max_component)
        return false;
      continue;
    }

    float inv_d = 1.0f / dir_component;
    float t0 = (min_component - origin_component) * inv_d;
    float t1 = (max_component - origin_component) * inv_d;
    if (t0 > t1)
      std::swap(t0, t1);

    if (t0 > t_min)
      t_min = t0;
    if (t1 < t_max)
      t_max = t1;
    if (t_max < t_min)
      return false;
  }

  out_distance = t_min >= 0.0f ? t_min : t_max;
  return out_distance >= 0.0f;
}

bool findOrbitFocusPoint(const veekay::vec3& origin, const veekay::vec3& direction,
                         veekay::vec3& out_point, float& out_distance) {
  bool hit = false;
  float closest = std::numeric_limits<float>::max();

  constexpr float ground_height = 0.0f;
  if (std::fabs(direction.y) > 1e-4f) {
    const float t = (ground_height - origin.y) / direction.y;
    if (t > 0.0f) {
      closest = t;
      out_point = origin + direction * t;
      hit = true;
    }
  }

  for (const Model& model : models) {
    veekay::vec3 min_bounds{};
    veekay::vec3 max_bounds{};
    if (!computeModelAabb(model, min_bounds, max_bounds))
      continue;

    float t = 0.0f;
    if (rayIntersectsAabb(origin, direction, min_bounds, max_bounds, t)) {
      if (t > 0.0f && t < closest) {
        closest = t;
        out_point = origin + direction * t;
        hit = true;
      }
    }
  }

  if (hit)
    out_distance = closest;
  return hit;
}

veekay::vec3 safeNormalize(const veekay::vec3& vector,
                           const veekay::vec3& fallback = {0.0f, 0.0f, 1.0f}) {
  if (veekay::vec3::squaredLength(vector) < 1e-6f)
    return veekay::vec3::normalized(fallback);
  return veekay::vec3::normalized(vector);
}

Camera::Axes Camera::axes() const {
  const float pitch = toRadians(rotation.x);
  const float yaw = toRadians(rotation.y);

  veekay::vec3 front{
    std::cos(pitch) * std::sin(yaw),
    std::sin(pitch),
    std::cos(pitch) * std::cos(yaw)
  };
  front = veekay::vec3::normalized(front);

  veekay::vec3 right = veekay::vec3::cross(front, world_up);
  if (veekay::vec3::squaredLength(right) < 1e-4f)
    right = {1.0f, 0.0f, 0.0f};
  right = veekay::vec3::normalized(right);

  veekay::vec3 up = veekay::vec3::cross(right, front);
  up = veekay::vec3::normalized(up);

  return {front, right, up};
}

veekay::mat4 lookAt(const veekay::vec3& eye, const veekay::vec3& center, const veekay::vec3& up) {
  const veekay::vec3 f = veekay::vec3::normalized(center - eye);
  const veekay::vec3 s = veekay::vec3::normalized(veekay::vec3::cross(f, up));
  const veekay::vec3 u = veekay::vec3::cross(s, f);

  veekay::mat4 result = veekay::mat4::identity();
  result[0][0] = s.x;
  result[1][0] = s.y;
  result[2][0] = s.z;
  result[0][1] = u.x;
  result[1][1] = u.y;
  result[2][1] = u.z;
  result[0][2] = -f.x;
  result[1][2] = -f.y;
  result[2][2] = -f.z;
  result[3][0] = -veekay::vec3::dot(s, eye);
  result[3][1] = -veekay::vec3::dot(u, eye);
  result[3][2] = veekay::vec3::dot(f, eye);
  return result;
}

veekay::mat4 orthographic(float left, float right, float bottom, float top, float near_plane, float far_plane) {
  veekay::mat4 result{};

  result[0][0] = 2.0f / (right - left);
  result[1][1] = -2.0f / (top - bottom);
  result[2][2] = 1.0f / (near_plane - far_plane);
  result[3][0] = -(right + left) / (right - left);
  result[3][1] = -(top + bottom) / (top - bottom);
  result[3][2] = near_plane / (near_plane - far_plane);
  result[3][3] = 1.0f;

  return result;
}

veekay::mat4 Camera::view() const {
  const Axes camera_axes = axes();
  return lookAt(position, position + camera_axes.front, camera_axes.up);
}

veekay::mat4 Camera::view_projection(float aspect_ratio) const {
  auto projection = veekay::mat4::projection(fov, aspect_ratio, near_plane, far_plane);

  return view() * projection;
}

namespace {
  void clampOrbitState(OrbitState& state) {
    state.distance = std::clamp(state.distance, 0.5f, 100.0f);
    state.pitch = std::clamp(state.pitch, -89.0f, 89.0f);
    if (sta

te.yaw > 360.0f)
      state.yaw -= 360.0f;
    else if (state.yaw < -360.0f)
      state.yaw += 360.0f;
  }

  veekay::vec3 cameraViewForward(const Camera::Axes& axes) {
    return axes.front;
  }

  veekay::vec3 cameraViewRight(const Camera::Axes& axes) {
    return axes.right;
  }

  veekay::vec3 cameraViewUp(const Camera::Axes& axes) {
    return axes.up;
  }

  Camera::Axes orbitAxesFromState(const OrbitState& state) {
    const float pitch_rad = toRadians(state.pitch);
    const float yaw_rad = toRadians(state.yaw);
    veekay::vec3 front{
      std::cos(pitch_rad) * std::sin(yaw_rad),
      std::sin(pitch_rad),
      std::cos(pitch_rad) * std::cos(yaw_rad)
    };
    if (veekay::vec3::squaredLength(front) < 1e-4f)
      front = {0.0f, 0.0f, 1.0f};
    else
      front = veekay::vec3::normalized(front);

    veekay::vec3 right = veekay::vec3::cross(front, world_up);
    if (veekay::vec3::squaredLength(right) < 1e-4f)
      right = {1.0f, 0.0f, 0.0f};
    else
      right = veekay::vec3::normalized(right);

    veekay::vec3 up = veekay::vec3::cross(right, front);
    up = veekay::vec3::normalized(up);

    return {front, right, up};
  }

  void updateOrbitStateFromTarget(const veekay::vec3& target, veekay::vec3 front, float distance) {
    if (veekay::vec3::squaredLength(front) < 1e-4f)
      front = {0.0f, 0.0f, 1.0f};
    else
      front = veekay::vec3::normalized(front);

    float safe_distance = distance;
    if (!std::isfinite(safe_distance)  safe_distance < 0.5f)
      safe_distance = std::max(orbit_state.distance, 5.0f);

    const float clamped_y = std::clamp(front.y, -1.0f, 1.0f);
    const float pitch = toDegrees(std::asin(clamped_y));
    const float yaw = toDegrees(std::atan2(front.x, front.z));

    orbit_state.target = target;
    orbit_state.distance = std::clamp(safe_distance, 0.5f, 100.0f);
    orbit_state.pitch = std::clamp(pitch, -89.0f, 89.0f);
    orbit_state.yaw = yaw;
    clampOrbitState(orbit_state);
  }
}

void applyCameraState(CameraMode mode) {
  switch (mode) {
  case CameraMode::FreeFly:
    camera.position = free_fly_state.position;
    camera.rotation = free_fly_state.rotation;
    break;
  case CameraMode::Orbit: {
    clampOrbitState(orbit_state);
    Camera::Axes axes = orbitAxesFromState(orbit_state);
    const veekay::vec3 forward = cameraViewForward(axes);
    camera.rotation = {orbit_state.pitch, orbit_state.yaw, 0.0f};
    camera.position = orbit_state.target - forward * orbit_state.distance;
    break;
  }
  }
}

void saveCameraState(CameraMode mode) {
  switch (mode) {
  case CameraMode::FreeFly:
    free_fly_state.position = camera.position;
    free_fly_state.rotation = camera.rotation;
    break;
  case CameraMode::Orbit: {
    const Camera::Axes axes = camera.axes();
    const veekay::vec3 forward = cameraViewForward(axes);
    orbit_state.pitch = camera.rotation.x;
    orbit_state.yaw = camera.rotation.y;
    float distance = veekay::vec3::length(orbit_state.target - camera.position);
    if (distance < 0.001f)
      distance = std::max(orbit_state.distance, 0.5f);
    orbit_state.distance = distance;
    orbit_state.target = camera.position + forward * orbit_state.distance;
    clampOrbitState(orbit_state);
    break;
  }
  }
}

void captureOrbitStateFromCamera(float desired_distance) {
  const Camera::Axes axes = camera.axes();
  veekay::vec3 front = cameraViewForward(axes);
  if (veekay::vec3::squaredLength(front) < 1e-4f)
    front = {0.0f, 0.0f, 1.0f};
  else
    front = veekay::vec3::normalized(front);

  veekay::vec3 focus_point{};
  float focus_distance = 0.0f;
  const bool hit = findOrbitFocusPoint(camera.position, front, focus_point, focus_distance);
  if (!hit) {
    float fallback_distance = desired_distance > 0.0f ? desired_distance : orbit_state.distance;
    if (!std::isfinite(fallback_distance)  fallback_distance < 0.5f)
      fallback_distance = 5.0f;
    focus_distance = fallback_distance;
    focus_point = camera.position + front * focus_distance;
  }

  updateOrbitStateFromTarget(focus_point, front, focus_distance);
}

void captureOrbitStateFromCamera(const veekay::vec3& focus_point, float desired_distance) {
  veekay::vec3 front = focus_point - camera.position;
  if (veekay::vec3::squaredLength(front) < 1e-4f)
    front = cameraViewForward(camera.axes());
  updateOrbitStateFromTarget(
    focus_point,
    front,
    desired_distance > 0.0f ? desired_distance : veekay::vec3::length(focus_point - camera.position)
  );
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

VkFormat chooseShadowDepthFormat(VkPhysicalDevice physical_device) {
  const VkFormat candidates[] = {
    VK_FORMAT_D32_SFLOAT,
    VK_FORMAT_D32_SFLOAT_S8_UINT,
    VK_FORMAT_D24_UNORM_S8_UINT,
  };

  for (VkFormat format : candidates) {
    VkFormatProperties properties;
    vkGetPhysicalDeviceFormatProperties(physical_device, format, &properties);

    const VkFormatFeatureFlags required = VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT |
                                          VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
    if ((properties.optimalTilingFeatures & required) == required) {
      return format;
    }
  }

  return VK_FORMAT_UNDEFINED;
}

void destroyShadowPassResources(VkDevice device, ShadowPass& pass);

bool createShadowPassResources(VkDevice device, VkPhysicalDevice physical_device, ShadowPass& pass, VkFormat depth_format) {
  pass.depth_format = depth_format;
  pass.image_layout = VK_IMAGE_LAYOUT_UNDEFINED;

  VkImageCreateInfo image_info{
    .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
    .imageType = VK_IMAGE_TYPE_2D,
    .format = depth_format,
    .extent = {ShadowPass::map_size, ShadowPass::map_size, 1},
    .mipLevels = 1,
    .arrayLayers = 1,
    .samples = VK_SAMPLE_COUNT_1_BIT,
    .tiling = VK_IMAGE_TILING_OPTIMAL,
    .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
  };

  if (vkCreateImage(device, &image_info, nullptr, &pass.depth_image) != VK_SUCCESS) {
    std::cerr << "Failed to create Vulkan shadow depth image\n";
    destroyShadowPassResources(device, pass);
    return false;
  }

  VkMemoryRequirements requirements;
  vkGetImageMemoryRequirements(device, pass.depth_image, &requirements);

  VkPhysicalDeviceMemoryProperties properties;
  vkGetPhysicalDeviceMemoryProperties(physical_device, &properties);

  uint32_t index = UINT_MAX;
  for (uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
    const VkMemoryType& type = properties.memoryTypes[i];
    if ((requirements.memoryTypeBits & (1 << i)) &&
        (type.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
      index = i;
      break;
    }
  }

  if (index == UINT_MAX) {
    std::cerr << "Failed to find memory type for shadow map\n";
    destroyShadowPassResources(device, pass);
    return false;
  }

  VkMemoryAllocateInfo allocate_info{
    .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
    .allocationSize = requirements.size,
    .memoryTypeIndex = index,
  };

if (vkAllocateMemory(device, &allocate_info, nullptr, &pass.depth_memory) != VK_SUCCESS) {
    std::cerr << "Failed to allocate shadow map memory\n";
    destroyShadowPassResources(device, pass);
    return false;
  }

  if (vkBindImageMemory(device, pass.depth_image, pass.depth_memory, 0) != VK_SUCCESS) {
    std::cerr << "Failed to bind shadow map memory\n";
    destroyShadowPassResources(device, pass);
    return false;
  }

  VkImageViewCreateInfo view_info{
    .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
    .image = pass.depth_image,
    .viewType = VK_IMAGE_VIEW_TYPE_2D,
    .format = depth_format,
    .subresourceRange = {
      .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
      .baseMipLevel = 0,
      .levelCount = 1,
      .baseArrayLayer = 0,
      .layerCount = 1,
    },
  };

  if (vkCreateImageView(device, &view_info, nullptr, &pass.depth_view) != VK_SUCCESS) {
    std::cerr << "Failed to create shadow map image view\n";
    destroyShadowPassResources(device, pass);
    return false;
  }

  VkSamplerCreateInfo sampler_info{
    .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
    .magFilter = VK_FILTER_LINEAR,
    .minFilter = VK_FILTER_LINEAR,
    .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
    .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
    .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
    .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
    .compareEnable = VK_TRUE,
    .compareOp = VK_COMPARE_OP_LESS,
    .minLod = 0.0f,
    .maxLod = 1.0f,
    .borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
  };

  if (vkCreateSampler(device, &sampler_info, nullptr, &pass.sampler) != VK_SUCCESS) {
    std::cerr << "Failed to create shadow sampler\n";
    destroyShadowPassResources(device, pass);
    return false;
  }

  pass.uniform_buffer = new veekay::graphics::Buffer(
    sizeof(veekay::mat4),
    nullptr,
    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

  return true;
}

void destroyShadowPassResources(VkDevice device, ShadowPass& pass) {
  if (pass.uniform_buffer) {
    delete pass.uniform_buffer;
    pass.uniform_buffer = nullptr;
  }
  if (pass.sampler) {
    vkDestroySampler(device, pass.sampler, nullptr);
    pass.sampler = VK_NULL_HANDLE;
  }
  if (pass.depth_view) {
    vkDestroyImageView(device, pass.depth_view, nullptr);
    pass.depth_view = VK_NULL_HANDLE;
  }
  if (pass.depth_memory) {
    vkFreeMemory(device, pass.depth_memory, nullptr);
    pass.depth_memory = VK_NULL_HANDLE;
  }
  if (pass.depth_image) {
    vkDestroyImage(device, pass.depth_image, nullptr);
    pass.depth_image = VK_NULL_HANDLE;
  }
  pass.image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
}

void initialize(VkCommandBuffer cmd) {
  VkDevice& device = veekay::app.vk_device;
  VkPhysicalDevice& physical_device = veekay::app.vk_physical_device;
  applyCameraState(camera_mode);
  captureOrbitStateFromCamera();

  // Create Sampler
  VkSamplerCreateInfo samplerInfo{};
  samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  samplerInfo.magFilter = VK_FILTER_NEAREST;
  samplerInfo.minFilter = VK_FILTER_NEAREST;
  samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.anisotropyEnable = VK_FALSE; // Keep it simple for now
  samplerInfo.maxAnisotropy = 1.0f;
  samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
  samplerInfo.unnormalizedCoordinates = VK_FALSE;
  samplerInfo.compareEnable = VK_FALSE;
  samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
  samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

  if (vkCreateSampler(device, &samplerInfo, nullptr, &texture_sampler) != VK_SUCCESS) {
    std::cerr << "Failed to create texture sampler!" << std::endl;
  }

  // Load Textures
  textures.push_back(load_texture(cmd, "assets/lenna.png"));
  textures.push_back(load_texture(cmd, "assets/robi.png"));

  // Shadow map resources
  VkFormat shadow_depth_format = chooseShadowDepthFormat(physical_device);
  if (shadow_depth_format == VK_FORMAT_UNDEFINED) {
    std::cerr <

< "Failed to find supported depth format for shadow map\n";
    veekay::app.running = false;
    return;
  }
  if (!createShadowPassResources(device, physical_device, shadow, shadow_depth_format)) {
    veekay::app.running = false;
    return;
  }

  shadow.depth_format = shadow_depth_format;
  active_spot_shadow_count = 0;
  spot_shadow_light_indices.fill(-1);
  for (ShadowPass& pass : spot_shadows) {
    if (!createShadowPassResources(device, physical_device, pass, shadow_depth_format)) {
      veekay::app.running = false;
      return;
    }
  }

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
      .cullMode = VK_CULL_MODE_BACK_BIT,
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
          .descriptorCount = max_models * 2 + 2 + max_shadow_spot_lights * 2,
        },
        {
          .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
          .descriptorCount = max_models * 2 + 2 + max_shadow_spot_lights * 2,
        },
        {
          .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .descriptorCount = max_models * 4,
        },
        {
          .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          .descriptorCount = max_models * 5,
        }
      };

      VkDescriptorPoolCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = max_models + 8,
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
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .descriptorCount = 1,
          .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
        },
        {
          .binding = 3,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .descriptorCount = 1,
          .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
        },
        {
          .binding = 4,
          .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          .descriptorCount = 1,
          .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
        },
        {
          .binding = 5,
          .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          .descriptorCount = 1,
          .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
        },
        {
          .binding = 6,
          .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          .descriptorCount = max_shadow_spot_lights,
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

    /*
    {
      VkDescriptorSetAllocateInfo info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &descriptor_set_layout,
      };

      if (vkAllocateDescriptorSets(device, &info, &descriptor_set) != VK_SUCCESS) {
        std::cerr << "Failed to create Vulkan descriptor set\n";
        veekay::app.running = false;
        return;
      }
    }
    */

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

// NOTE: Shadow map pipeline (depth only, dynamic rendering)
    shadow.vertex_shader_module = loadShaderModule("./shaders/shadow.vert.spv");
    if (!shadow.vertex_shader_module) {
      std::cerr << "Failed to load Vulkan shadow vertex shader from file\n";
      veekay::app.running = false;
      return;
    }

    VkPipelineShaderStageCreateInfo shadow_stage{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_VERTEX_BIT,
      .module = shadow.vertex_shader_module,
      .pName = "main",
    };

    VkVertexInputAttributeDescription shadow_attributes[] = {
      {
        .location = 0,
        .binding = 0,
        .format = VK_FORMAT_R32G32B32_SFLOAT,
        .offset = offsetof(Vertex, position),
      },
    };

    VkPipelineVertexInputStateCreateInfo shadow_input_state{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .vertexBindingDescriptionCount = 1,
      .pVertexBindingDescriptions = &buffer_binding,
      .vertexAttributeDescriptionCount = sizeof(shadow_attributes) / sizeof(shadow_attributes[0]),
      .pVertexAttributeDescriptions = shadow_attributes,
    };

    VkPipelineRasterizationStateCreateInfo shadow_raster{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = VK_CULL_MODE_FRONT_BIT,
      .frontFace = VK_FRONT_FACE_CLOCKWISE,
      .depthBiasEnable = VK_TRUE,
      .lineWidth = 1.0f,
    };

    VkViewport shadow_viewport{
      .x = 0.0f,
      .y = 0.0f,
      .width = static_cast<float>(ShadowPass::map_size),
      .height = static_cast<float>(ShadowPass::map_size),
      .minDepth = 0.0f,
      .maxDepth = 1.0f,
    };

    VkRect2D shadow_scissor{
      .offset = {0, 0},
      .extent = {ShadowPass::map_size, ShadowPass::map_size},
    };

    VkPipelineViewportStateCreateInfo shadow_viewport_info{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1,
      .pViewports = &shadow_viewport,
      .scissorCount = 1,
      .pScissors = &shadow_scissor,
    };

    VkPipelineDepthStencilStateCreateInfo shadow_depth{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = true,
      .depthWriteEnable = true,
      .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
    };

    VkPipelineColorBlendStateCreateInfo shadow_blend{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 0,
      .pAttachments = nullptr,
    };

    VkDynamicState dyn_states[] = {
      VK_DYNAMIC_STATE_VIEWPORT,
      VK_DYNAMIC_STATE_SCISSOR,
      VK_DYNAMIC_STATE_DEPTH_BIAS,
    };

    VkPipelineDynamicStateCreateInfo dyn_state_info{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = sizeof(dyn_states) / sizeof(dyn_states[0]),
      .pDynamicStates = dyn_states,
    };

    {
      VkDescriptorSetLayoutBinding shadow_bindings[] = {
        {
          .binding = 0,
          .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          .descriptorCount = 1,
          .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        },
        {
          .binding = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
          .descriptorCount = 1,
          .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        },
      };

      VkDescriptorSetLayoutCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = sizeof(shadow_bindings) / sizeof(shadow_bindings[0]),
        .pBindings = shadow_bindings,
      };

      if (vkCreateDescriptorSetLayout(device, &info, nullptr, &shadow.descriptor_set_layout) != VK_SUCCESS) {
        std::cerr << "Failed to create shadow descriptor set layout\n";
        veekay::app.running = false;
        return;
      }
    }

{
      VkPipelineLayoutCreateInfo layout_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &shadow.descriptor_set_layout,
      };

      if (vkCreatePipelineLayout(device, &layout_info, nullptr, &shadow.pipeline_layout) != VK_SUCCESS) {
        std::cerr << "Failed to create shadow pipeline layout\n";
        veekay::app.running = false;
        return;
      }
    }

    VkPipelineRenderingCreateInfo rendering_info{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = 0,
      .pColorAttachmentFormats = nullptr,
      .depthAttachmentFormat = shadow.depth_format,
    };

    VkGraphicsPipelineCreateInfo shadow_info{
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &rendering_info,
      .stageCount = 1,
      .pStages = &shadow_stage,
      .pVertexInputState = &shadow_input_state,
      .pInputAssemblyState = &assembly_state_info,
      .pViewportState = &shadow_viewport_info,
      .pRasterizationState = &shadow_raster,
      .pMultisampleState = &sample_info,
      .pDepthStencilState = &shadow_depth,
      .pColorBlendState = &shadow_blend,
      .pDynamicState = &dyn_state_info,
      .layout = shadow.pipeline_layout,
      .renderPass = VK_NULL_HANDLE,
    };

    if (vkCreateGraphicsPipelines(device, nullptr, 1, &shadow_info, nullptr, &shadow.pipeline) != VK_SUCCESS) {
      std::cerr << "Failed to create shadow graphics pipeline\n";
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

  point_lights_buffer = new veekay::graphics::Buffer(
    max_point_lights * sizeof(PointLightGpu),
    nullptr,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

  spot_lights_buffer = new veekay::graphics::Buffer(
    max_spot_lights * sizeof(SpotLightGpu),
    nullptr,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

  {
    VkDescriptorSetAllocateInfo info{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = descriptor_pool,
      .descriptorSetCount = 1,
      .pSetLayouts = &shadow.descriptor_set_layout,
    };

    if (vkAllocateDescriptorSets(device, &info, &shadow.descriptor_set) != VK_SUCCESS) {
      std::cerr << "Failed to allocate shadow descriptor set\n";
      veekay::app.running = false;
      return;
    }

    VkDescriptorBufferInfo buffer_infos[] = {
      {
        .buffer = shadow.uniform_buffer->buffer,
        .offset = 0,
        .range = sizeof(veekay::mat4),
      },
      {
        .buffer = model_uniforms_buffer->buffer,
        .offset = 0,
        .range = sizeof(ModelUniforms),
      },
    };

    VkWriteDescriptorSet writes[] = {
      {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = shadow.descriptor_set,
        .dstBinding = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .pBufferInfo = &buffer_infos[0],
      },
      {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = shadow.descriptor_set,
        .dstBinding = 1,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
        .pBufferInfo = &buffer_infos[1],
      },
    };

    vkUpdateDescriptorSets(device, sizeof(writes) / sizeof(writes[0]), writes, 0, nullptr);
  }

  for (size_t i = 0; i < spot_shadows.size(); ++i) {
    VkDescriptorSetAllocateInfo info{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = descriptor_pool,
      .descriptorSetCount = 1,
      .pSetLayouts = &shadow.descriptor_set_layout,
    };

if (vkAllocateDescriptorSets(device, &info, &spot_shadows[i].descriptor_set) != VK_SUCCESS) {
      std::cerr << "Failed to allocate spot shadow descriptor set " << i << "\n";
      veekay::app.running = false;
      return;
    }

    VkDescriptorBufferInfo buffer_infos[] = {
      {
        .buffer = spot_shadows[i].uniform_buffer->buffer,
        .offset = 0,
        .range = sizeof(veekay::mat4),
      },
      {
        .buffer = model_uniforms_buffer->buffer,
        .offset = 0,
        .range = sizeof(ModelUniforms),
      },
    };

    VkWriteDescriptorSet writes[] = {
      {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = spot_shadows[i].descriptor_set,
        .dstBinding = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .pBufferInfo = &buffer_infos[0],
      },
      {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = spot_shadows[i].descriptor_set,
        .dstBinding = 1,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
        .pBufferInfo = &buffer_infos[1],
      },
    };

    vkUpdateDescriptorSets(device, sizeof(writes) / sizeof(writes[0]), writes, 0, nullptr);
  }

  // NOTE: Plane mesh initialization
  {
    // (v0)------(v1)
    //  |  \       |
    //  |   `--,   |
    //  |       \  |
    // (v3)------(v2)
    std::vector<Vertex> vertices = {
      {{-5.0f, 0.0f, 5.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
      {{5.0f, 0.0f, 5.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
      {{5.0f, 0.0f, -5.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
      {{-5.0f, 0.0f, -5.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},
    };

    std::vector<uint32_t> indices = {
      0, 2, 1, 2, 0, 3
    };

    computeMeshBounds(plane_mesh, vertices);

    plane_mesh.vertex_buffer = new veekay::graphics::Buffer(
      vertices.size() * sizeof(Vertex), vertices.data(),
      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

    plane_mesh.index_buffer = new veekay::graphics::Buffer(
      indices.size() * sizeof(uint32_t), indices.data(),
      VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

    plane_mesh.indices = uint32_t(indices.size());
  }

  // NOTE: Cube mesh initialization
  {
    std::vector<Vertex> vertices = {
      {{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f}},
      {{+0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 1.0f}},
      {{+0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f}},
      {{-0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f}},

      {{+0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},
      {{+0.5f, -0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
      {{+0.5f, +0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
      {{+0.5f, +0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},

      {{+0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
      {{-0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
      {{-0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}},
      {{+0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},

      {{-0.5f, -0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},
      {{-0.5f, -0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
      {{-0.5f, +0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
      {{-0.5f, +0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},

      {{-0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},
      {{+0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
      {{+0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
      {{-0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},

      {{-0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},
      {{+0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
      {{+0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
      {{-0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
    };

    std::vector<uint32_t> indices = {
      0, 1, 2, 2, 3, 0,
      4, 5, 6, 6, 7, 4,
      8, 9, 10, 10, 11, 8,
      12, 13, 14, 14, 15, 12,
      16, 17, 18, 18, 19, 16,
      20, 21, 22, 22, 23, 20,
    };

computeMeshBounds(cube_mesh, vertices);

    cube_mesh.vertex_buffer = new veekay::graphics::Buffer(
      vertices.size() * sizeof(Vertex), vertices.data(),
      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

    cube_mesh.index_buffer = new veekay::graphics::Buffer(
      indices.size() * sizeof(uint32_t), indices.data(),
      VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

    cube_mesh.indices = uint32_t(indices.size());
  }

  // NOTE: Add models to scene
  models.emplace_back(Model{
    .mesh = plane_mesh,
    .transform = Transform{},
    .material = Material{
      .albedo_color = {1.0f, 1.0f, 1.0f},
      .shininess = 8.0f,
      .specular_color = {0.8f, 0.8f, 0.8f},
    }
  });

  models.emplace_back(Model{
    .mesh = cube_mesh,
    .transform = Transform{
      .position = {-2.0f, 0.5f, -1.5f},
    },
    .material = Material{
      .albedo_color = {1.0f, 0.0f, 0.0f},
      .shininess = 64.0f,
      .specular_color = {1.0f, 0.6f, 0.6f},
    }
  });

  models.emplace_back(Model{
    .mesh = cube_mesh,
    .transform = Transform{
      .position = {1.5f, 0.5f, -0.5f},
    },
    .material = Material{
      .albedo_color = {0.0f, 1.0f, 0.0f},
      .shininess = 32.0f,
      .specular_color = {0.6f, 1.0f, 0.6f},
    }
  });

  models.emplace_back(Model{
    .mesh = cube_mesh,
    .transform = Transform{
      .position = {0.0f, 0.5f, 1.0f},
    },
    .material = Material{
      .albedo_color = {0.0f, 0.0f, 1.0f},
      .shininess = 16.0f,
      .specular_color = {0.6f, 0.6f, 1.0f},
    }
  });

  // Allocate and update descriptor sets for each model
  for (size_t i = 0; i < models.size(); ++i) {
    VkDescriptorSetAllocateInfo info{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = descriptor_pool,
      .descriptorSetCount = 1,
      .pSetLayouts = &descriptor_set_layout,
    };

    if (vkAllocateDescriptorSets(device, &info, &models[i].descriptor_set) != VK_SUCCESS) {
      std::cerr << "Failed to allocate descriptor set for model " << i << "\n";
      continue;
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
      {
        .buffer = point_lights_buffer->buffer,
        .offset = 0,
        .range = max_point_lights * sizeof(PointLightGpu),
      },
      {
        .buffer = spot_lights_buffer->buffer,
        .offset = 0,
        .range = max_spot_lights * sizeof(SpotLightGpu),
      },
    };

    // Choose texture based on model index
    veekay::graphics::Texture* tex = textures[i % textures.size()];

    VkDescriptorImageInfo image_info{
      .sampler = texture_sampler,
      .imageView = tex->view,
      .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };

    VkDescriptorImageInfo shadow_image_info{
      .sampler = shadow.sampler,
      .imageView = shadow.depth_view,
      .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };

    std::array<VkDescriptorImageInfo, max_shadow_spot_lights> spot_shadow_image_infos{};
    for (size_t shadow_index = 0; shadow_index < max_shadow_spot_lights; ++shadow_index) {
      const ShadowPass& pass = spot_shadows[shadow_index];
      spot_shadow_image_infos[shadow_index] = VkDescriptorImageInfo{
        .sampler = pass.sampler,
        .imageView = pass.depth_view,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      };
    }

VkWriteDescriptorSet write_infos[] = {
      {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = models[i].descriptor_set,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .pBufferInfo = &buffer_infos[0],
      },
      {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = models[i].descriptor_set,
        .dstBinding = 1,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
        .pBufferInfo = &buffer_infos[1],
      },
      {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = models[i].descriptor_set,
        .dstBinding = 2,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &buffer_infos[2],
      },
      {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = models[i].descriptor_set,
        .dstBinding = 3,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &buffer_infos[3],
      },
      {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = models[i].descriptor_set,
        .dstBinding = 4,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .pImageInfo = &image_info,
      },
      {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = models[i].descriptor_set,
        .dstBinding = 5,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .pImageInfo = &shadow_image_info,
      },
      {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = models[i].descriptor_set,
        .dstBinding = 6,
        .dstArrayElement = 0,
        .descriptorCount = max_shadow_spot_lights,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .pImageInfo = spot_shadow_image_infos.data(),
      },
    };

    vkUpdateDescriptorSets(device, sizeof(write_infos) / sizeof(write_infos[0]),
                           write_infos, 0, nullptr);
  }
}

// NOTE: Destroy resources here, do not cause leaks in your program!
void shutdown() {
  VkDevice& device = veekay::app.vk_device;

  for (auto* tex : textures) {
    delete tex;
  }
  textures.clear();
  vkDestroySampler(device, texture_sampler, nullptr);

  destroyShadowPassResources(device, shadow);
  for (auto& pass : spot_shadows)
    destroyShadowPassResources(device, pass);

  delete cube_mesh.index_buffer;
  delete cube_mesh.vertex_buffer;

  delete plane_mesh.index_buffer;
  delete plane_mesh.vertex_buffer;

  delete model_uniforms_buffer;
  delete point_lights_buffer;
  delete spot_lights_buffer;
  delete scene_uniforms_buffer;

  vkDestroyPipeline(device, shadow.pipeline, nullptr);
  vkDestroyPipelineLayout(device, shadow.pipeline_layout, nullptr);
  vkDestroyDescriptorSetLayout(device, shadow.descriptor_set_layout, nullptr);
  vkDestroyShaderModule(device, shadow.vertex_shader_module, nullptr);

  vkDestroyPipeline(device, pipeline, nullptr);
  vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
  vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
  vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
  vkDestroyShaderModule(device, fragment_shader_module, nullptr);
  vkDestroyShaderModule(device, vertex_shader_module, nullptr);
}

void update(double time) {
  ImGuiIO& io = ImGui::GetIO();

ImGui::Begin("Camera");
  const char* mode_labels[] = {"FreeFly (FPV)", "Orbit (Look-At)"};
  int mode_index = static_cast<int>(camera_mode);
  if (ImGui::Combo("Mode", &mode_index, mode_labels, IM_ARRAYSIZE(mode_labels))) {
    const CameraMode new_mode = static_cast<CameraMode>(mode_index);
    if (new_mode != camera_mode) {
      saveCameraState(camera_mode);
      if (camera_mode == CameraMode::FreeFly && new_mode == CameraMode::Orbit)
        captureOrbitStateFromCamera();
      camera_mode = new_mode;
      applyCameraState(camera_mode);
    }
  }

  if (camera_mode == CameraMode::Orbit) {
    bool orbit_changed = false;
    ImGui::Text("Target: (%.2f, %.2f, %.2f)", orbit_state.target.x, orbit_state.target.y, orbit_state.target.z);
    orbit_state.distance = std::clamp(orbit_state.distance, 0.5f, 100.0f);
    orbit_changed |= ImGui::SliderFloat("Distance", &orbit_state.distance, 0.5f, 80.0f);
    if (ImGui::Button("Focus current view")) {
      captureOrbitStateFromCamera();
      orbit_changed = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Focus origin")) {
      captureOrbitStateFromCamera({0.0f, 0.0f, 0.0f});
      orbit_changed = true;
    }
    if (orbit_changed)
      applyCameraState(CameraMode::Orbit);
  }

  ImGui::Separator();
  ImGui::Text("Position: (%.2f, %.2f, %.2f)", camera.position.x, camera.position.y, camera.position.z);
  ImGui::SliderFloat("FOV", &camera.fov, 30.0f, 90.0f);
  ImGui::SliderFloat("Near", &camera.near_plane, 0.01f, 2.0f);
  ImGui::SliderFloat("Far", &camera.far_plane, 10.0f, 500.0f);
  ImGui::End();

  ImGui::Begin("Lighting");
  ImGui::Text("Ambient");
  ImGui::ColorEdit3("Ambient Color", &ambient_light.color.x);
  ImGui::SliderFloat("Ambient Intensity", &ambient_light.intensity, 0.0f, 3.0f);
  ImGui::Separator();
  ImGui::Text("Directional");
  ImGui::ColorEdit3("Directional Color", &directional_light.color.x);
  ImGui::SliderFloat3("Directional Dir", &directional_light.direction.x, -1.0f, 1.0f);
  ImGui::SliderFloat("Directional Intensity", &directional_light.intensity, 0.0f, 5.0f);
  ImGui::Separator();
  if (ImGui::Button("Add point light") && point_lights.size() < max_point_lights) {
    point_lights.emplace_back(PointLight{});
  }

  for (size_t i = 0; i < point_lights.size(); ++i) {
    ImGui::PushID(static_cast<int>(i));
    std::string header = "Point light " + std::to_string(i);
    if (ImGui::CollapsingHeader(header.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::SliderFloat3("Position", &point_lights[i].position.x, -10.0f, 10.0f);
      ImGui::ColorEdit3("Color", &point_lights[i].color.x);
      ImGui::SliderFloat("Intensity", &point_lights[i].intensity, 0.0f, 50.0f);
      ImGui::SliderFloat("Radius", &point_lights[i].radius, 0.1f, 25.0f);
      if (ImGui::Button("Remove") && !point_lights.empty()) {
        point_lights.erase(point_lights.begin() + static_cast<long long>(i));
        ImGui::PopID();
        break;
      }
    }
    ImGui::PopID();
  }

  ImGui::Separator();
  ImGui::Text("Spot Lights");
  if (ImGui::Button("Add spot light") && spot_lights.size() < max_spot_lights) {
    spot_lights.emplace_back(SpotLight{});
  }

for (size_t i = 0; i < spot_lights.size(); ++i) {
    ImGui::PushID(static_cast<int>(i + 1000));
    std::string header = "Spot light " + std::to_string(i);
    if (ImGui::CollapsingHeader(header.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::SliderFloat3("Position", &spot_lights[i].position.x, -10.0f, 10.0f);
      ImGui::SliderFloat3("Direction", &spot_lights[i].direction.x, -1.0f, 1.0f);
      ImGui::ColorEdit3("Color", &spot_lights[i].color.x);
      ImGui::SliderFloat("Intensity", &spot_lights[i].intensity, 0.0f, 60.0f);
      ImGui::SliderFloat("Range", &spot_lights[i].range, 0.5f, 30.0f);
      float inner = spot_lights[i].inner_angle;
      float outer = spot_lights[i].outer_angle;
      if (ImGui::SliderFloat("Inner angle", &inner, 1.0f, 80.0f))
        spot_lights[i].inner_angle = inner;
      if (ImGui::SliderFloat("Outer angle", &outer, 1.0f, 90.0f))
        spot_lights[i].outer_angle = outer;
      spot_lights[i].outer_angle = std::max(spot_lights[i].outer_angle, spot_lights[i].inner_angle + 0.1f);
      if (ImGui::Button("Remove") && !spot_lights.empty()) {
        spot_lights.erase(spot_lights.begin() + static_cast<long long>(i));
        ImGui::PopID();
        break;
      }
    }
    ImGui::PopID();
  }
  ImGui::End();

  ImGui::Begin("Materials");
  for (size_t i = 0; i < models.size(); ++i) {
    ImGui::PushID(static_cast<int>(i));
    std::string label = "Model " + std::to_string(i);
    if (ImGui::CollapsingHeader(label.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::ColorEdit3("Albedo", &models[i].material.albedo_color.x);
      ImGui::ColorEdit3("Specular", &models[i].material.specular_color.x);
      ImGui::SliderFloat("Shininess", &models[i].material.shininess, 1.0f, 256.0f);
    }
    ImGui::PopID();
  }
  ImGui::End();

  static double previous_time = time;
  double delta_time = time - previous_time;
  if (delta_time < 0.0)
    delta_time = 0.0;
  previous_time = time;

  using namespace veekay::input;
  const bool block_camera = io.WantCaptureMouse  io.WantCaptureKeyboard;
  const float base_speed = 3.5f;
  float move_speed = base_speed * static_cast<float>(delta_time);
  if (io.KeyShift)
    move_speed *= 3.0f;
  move_speed = std::max(move_speed, 0.01f);

  if (!block_camera) {
    if (camera_mode == CameraMode::FreeFly) {
      bool changed = false;
      if (mouse::isButtonDown(mouse::Button::left)) {
        auto move_delta = mouse::cursorDelta();
        constexpr float sensitivity = 0.15f;
        free_fly_state.rotation.y -= move_delta.x * sensitivity;
        free_fly_state.rotation.x = std::clamp(free_fly_state.rotation.x - move_delta.y * sensitivity, -89.0f, 89.0f);
        changed = true;
      }

      const Camera::Axes axes = camera.axes();
      const veekay::vec3 forward = cameraViewForward(axes);
      const veekay::vec3 right = cameraViewRight(axes);
      const veekay::vec3 up = cameraViewUp(axes);

      if (keyboard::isKeyDown(keyboard::Key::w)) {
        free_fly_state.position += forward * move_speed;
        changed = true;
      }
      if (keyboard::isKeyDown(keyboard::Key::s)) {
        free_fly_state.position -= forward * move_speed;
        changed = true;
      }
      if (keyboard::isKeyDown(keyboard::Key::d)) {
        free_fly_state.position += right * move_speed;
        changed = true;
      }
      if (keyboard::isKeyDown(keyboard::Key::a)) {
        free_fly_state.position -= right * move_speed;
        changed = true;
      }
      if (keyboard::isKeyDown(keyboard::Key::q)) {
        free_fly_state.position += up * move_speed;
        changed = true;
      }
      if (keyboard::isKeyDown(keyboard::Key::z)) {
        free_fly_state.position -= up * move_speed;
        changed = true;
      }

if (changed)
        applyCameraState(CameraMode::FreeFly);
    } else {
      bool orbit_dirty = false;
      if (mouse::isButtonDown(mouse::Button::left)) {
        auto move_delta = mouse::cursorDelta();
        constexpr float sensitivity = 0.2f;
        orbit_state.yaw += move_delta.x * sensitivity;
        orbit_state.pitch = std::clamp(orbit_state.pitch - move_delta.y * sensitivity, -85.0f, 85.0f);
        orbit_dirty = true;
      }

      auto scroll = mouse::scrollDelta();
      if (std::abs(scroll.y) > 1e-3f) {
        orbit_state.distance -= scroll.y * 1.5f;
        orbit_dirty = true;
      }

      Camera::Axes axes = orbitAxesFromState(orbit_state);
      const veekay::vec3 view_forward = cameraViewForward(axes);
      const veekay::vec3 view_right = cameraViewRight(axes);

      veekay::vec3 forward = {view_forward.x, 0.0f, view_forward.z};
      if (veekay::vec3::squaredLength(forward) > 1e-4f)
        forward = veekay::vec3::normalized(forward);
      else
        forward = {0.0f, 0.0f, 0.0f};

      veekay::vec3 right = {view_right.x, 0.0f, view_right.z};
      if (veekay::vec3::squaredLength(right) > 1e-4f)
        right = veekay::vec3::normalized(right);
      else
        right = {1.0f, 0.0f, 0.0f};

      bool pan_changed = false;
      if (keyboard::isKeyDown(keyboard::Key::w)) {
        orbit_state.target += forward * move_speed;
        pan_changed = true;
      }
      if (keyboard::isKeyDown(keyboard::Key::s)) {
        orbit_state.target -= forward * move_speed;
        pan_changed = true;
      }
      if (keyboard::isKeyDown(keyboard::Key::d)) {
        orbit_state.target += right * move_speed;
        pan_changed = true;
      }
      if (keyboard::isKeyDown(keyboard::Key::a)) {
        orbit_state.target -= right * move_speed;
        pan_changed = true;
      }
      if (keyboard::isKeyDown(keyboard::Key::q)) {
        orbit_state.target += world_up * move_speed;
        pan_changed = true;
      }
      if (keyboard::isKeyDown(keyboard::Key::z)) {
        orbit_state.target -= world_up * move_speed;
        pan_changed = true;
      }

      if (pan_changed)
        orbit_dirty = true;

      if (orbit_dirty) {
        clampOrbitState(orbit_state);
        applyCameraState(CameraMode::Orbit);
      }
    }
  }

  float aspect_ratio = float(veekay::app.window_width) / float(veekay::app.window_height);
  veekay::vec3 dir = directional_light.direction;
  if (veekay::vec3::squaredLength(dir) < 1e-4f)
    dir = {0.0f, -1.0f, 0.0f};
  dir = veekay::vec3::normalized(dir);

  const veekay::vec3 shadow_focus{0.0f, 0.0f, 0.0f};
  const float shadow_distance = 20.0f;
  const float shadow_extent = 12.0f;
  const float shadow_near = 0.5f;
  const float shadow_far = 60.0f;
  const veekay::vec3 shadow_up = std::fabs(dir.y) > 0.9f ? veekay::vec3{0.0f, 0.0f, 1.0f} : world_up;
  const veekay::vec3 shadow_position = shadow_focus - dir * shadow_distance;
  const veekay::mat4 shadow_view = lookAt(shadow_position, shadow_focus, shadow_up);
  const veekay::mat4 shadow_proj = orthographic(
    -shadow_extent, shadow_extent,
    -shadow_extent, shadow_extent,
    shadow_near, shadow_far);
  shadow.view_projection = shadow_view * shadow_proj;

  if (shadow.uniform_buffer) {
    *reinterpret_cast<veekay::mat4*>(shadow.uniform_buffer->mapped_region) = shadow.view_projection;
  }

  spot_shadow_light_indices.fill(-1);
  active_spot_shadow_count = 0;
  std::array<veekay::mat4, max_shadow_spot_lights> spot_shadow_matrices{};

  if (!spot_lights.empty()) {
    struct SpotShadowCandidate {
      size_t index;
      float score;
    };

    std::vector<SpotShadowCandidate> candidates;
    candidates.reserve(spot_lights.size());

for (size_t i = 0; i < spot_lights.size(); ++i) {
      const SpotLight& light = spot_lights[i];
      const veekay::vec3 to_camera = camera.position - light.position;
      const float distance = veekay::vec3::length(to_camera);
      const float distance_factor = 1.0f / (distance + 1.0f);
      const float range_factor = std::clamp(light.range / (distance + light.range + 0.01f), 0.0f, 1.0f);
      float direction_factor = 0.0f;
      if (veekay::vec3::squaredLength(to_camera) > 1e-6f) {
        const veekay::vec3 to_camera_dir = veekay::vec3::normalized(to_camera);
        direction_factor = std::max(0.0f, veekay::vec3::dot(safeNormalize(light.direction, {0.0f, -1.0f, 0.0f}), to_camera_dir));
      }
      const float intensity_factor = std::clamp(light.intensity / 50.0f, 0.0f, 1.0f);
      const float score =
        distance_factor * 0.45f +
        range_factor * 0.2f +
        direction_factor * 0.2f +
        intensity_factor * 0.15f;
      candidates.push_back({i, score});
    }

    std::sort(candidates.begin(), candidates.end(), [](const SpotShadowCandidate& a, const SpotShadowCandidate& b) {
      return a.score > b.score;
    });

    active_spot_shadow_count = std::min(candidates.size(), static_cast<size_t>(max_shadow_spot_lights));
    for (size_t idx = 0; idx < active_spot_shadow_count; ++idx) {
      const SpotLight& light = spot_lights[candidates[idx].index];
      spot_shadow_light_indices[idx] = static_cast<int32_t>(candidates[idx].index);

      const veekay::vec3 direction = safeNormalize(light.direction, {0.0f, -1.0f, 0.0f});
      const veekay::vec3 up = std::fabs(direction.y) > 0.9f ? veekay::vec3{0.0f, 0.0f, 1.0f} : world_up;
      const float near_plane = std::max(0.05f, light.range * 0.05f);
      const float far_plane = std::max(light.range, near_plane + 0.1f);
      const float fov = std::clamp(light.outer_angle * 2.0f, 5.0f, 170.0f);
      const veekay::mat4 spot_view = lookAt(light.position, light.position + direction, up);
      const veekay::mat4 spot_proj = veekay::mat4::projection(fov, 1.0f, near_plane, far_plane);
      const veekay::mat4 view_proj = spot_view * spot_proj;

      spot_shadow_matrices[idx] = view_proj;
      if (spot_shadows[idx].uniform_buffer)
        *reinterpret_cast<veekay::mat4*>(spot_shadows[idx].uniform_buffer->mapped_region) = view_proj;
    }
  }

  for (size_t i = active_spot_shadow_count; i < max_shadow_spot_lights; ++i)
    spot_shadow_matrices[i] = veekay::mat4::identity();

  SceneUniforms scene_uniforms{};
  scene_uniforms.view_projection = camera.view_projection(aspect_ratio);
  scene_uniforms.shadow_view_projection = shadow.view_projection;
  scene_uniforms.camera_position = {camera.position.x, camera.position.y, camera.position.z, 1.0f};
  scene_uniforms.ambient_color = {ambient_light.color.x, ambient_light.color.y, ambient_light.color.z, ambient_light.intensity};
  scene_uniforms.directional_direction_intensity = {dir.x, dir.y, dir.z, directional_light.intensity};
  scene_uniforms.directional_color = {directional_light.color.x, directional_light.color.y, directional_light.color.z, 1.0f};
  const size_t light_count = std::min(point_lights.size(), static_cast<size_t>(max_point_lights));
  const size_t spot_count = std::min(spot_lights.size(), static_cast<size_t>(max_spot_lights));
  scene_uniforms.light_counts = {
    static_cast<float>(light_count),
    static_cast<float>(spot_count),
    0.0f,
    0.0f
  };
  for (size_t i = 0; i < max_shadow_spot_lights; ++i) {
    scene_uniforms.spot_shadow_view_projections[i] = spot_shadow_matrices[i];
  }
  scene_uniforms.spot_shadow_indices = {
    static_cast<float>(spot_shadow_light_indices[0]),
    static_cast<float>(spot_shadow_light_indices[1]),
    static_cast<float>(active_spot_shadow_count),
    0.0f
  };

  std::vector<ModelUniforms> model_uniforms(models.size());
  for (size_t i = 0, n = models.size(); i < n; ++i) {
    const Model& model = models[i];
    ModelUniforms& uniforms = model_uniforms[i];

uniforms.model = model.transform.matrix();
    uniforms.albedo_shininess = {
      model.material.albedo_color.x,
      model.material.albedo_color.y,
      model.material.albedo_color.z,
      std::max(model.material.shininess, 1.0f)
    };
    uniforms.specular_color = {
      model.material.specular_color.x,
      model.material.specular_color.y,
      model.material.specular_color.z,
      1.0f
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

  if (point_lights_buffer) {
    PointLightGpu* data = static_cast<PointLightGpu*>(point_lights_buffer->mapped_region);
    for (size_t i = 0; i < light_count; ++i) {
      const PointLight& light = point_lights[i];
      const float radius = std::max(light.radius, 0.1f);
      data[i].position_intensity = {light.position.x, light.position.y, light.position.z, light.intensity};
      data[i].color_radius = {light.color.x, light.color.y, light.color.z, radius};
    }
    for (size_t i = light_count; i < max_point_lights; ++i) {
      data[i].position_intensity = {0.0f, 0.0f, 0.0f, 0.0f};
      data[i].color_radius = {0.0f, 0.0f, 0.0f, 0.0f};
    }
  }

  if (spot_lights_buffer) {
    SpotLightGpu* data = static_cast<SpotLightGpu*>(spot_lights_buffer->mapped_region);
    const size_t count = std::min(spot_lights.size(), static_cast<size_t>(max_spot_lights));
    for (size_t i = 0; i < count; ++i) {
      const SpotLight& light = spot_lights[i];
      const veekay::vec3 direction = safeNormalize(light.direction, {0.0f, -1.0f, 0.0f});
      float inner_angle = std::clamp(light.inner_angle, 0.1f, 89.0f);
      float outer_angle = std::clamp(light.outer_angle, inner_angle + 0.1f, 89.0f);
      const float inner_cos = std::cos(toRadians(inner_angle));
      const float outer_cos = std::cos(toRadians(outer_angle));
      const float range = std::max(light.range, 0.1f);

      data[i].position_intensity = {light.position.x, light.position.y, light.position.z, light.intensity};
      data[i].direction_inner_cos = {direction.x, direction.y, direction.z, inner_cos};
      data[i].color_range = {light.color.x, light.color.y, light.color.z, range};
      data[i].params = {outer_cos, 0.0f, 0.0f, 0.0f};
    }
    for (size_t i = count; i < max_spot_lights; ++i) {
      data[i].position_intensity = {0.0f, 0.0f, 0.0f, 0.0f};
      data[i].direction_inner_cos = {0.0f, 0.0f, 0.0f, 0.0f};
      data[i].color_range = {0.0f, 0.0f, 0.0f, 0.0f};
      data[i].params = {0.0f, 0.0f, 0.0f, 0.0f};
    }
  }
}

void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
  auto transition_shadow_image = [&](ShadowPass& pass, VkImageLayout new_layout,
                                     VkAccessFlags dst_access, VkPipelineStageFlags dst_stage) {
    VkImageMemoryBarrier barrier{
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .oldLayout = pass.image_layout,
      .newLayout = new_layout,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = pass.depth_image,
      .subresourceRange = {
        .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1,
      },
    };

    if (pass.image_layout == VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL)
      barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    barrier.dstAccessMask = dst_access;

    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         dst_stage,
                         0,
                         0, nullptr,
                         0, nullptr,
                         1, &barrier);

    pass.image_layout = new_layout;
  };

  vkResetCommandBuffer(cmd, 0);

  { // NOTE: Start recording rendering commands
    VkCommandBufferBeginInfo info{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };

    vkBeginCommandBuffer(cmd, &info);
  }

  VkDeviceSize zero_offset = 0;
  const size_t model_uniforms_alignment =
    veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

  auto render_shadow_pass = [&](ShadowPass& pass, VkViewport viewport, VkRect2D scissor) {
    transition_shadow_image(pass,
                            VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT);

    VkRenderingAttachmentInfo depth_attachment{
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = pass.depth_view,
      .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .clearValue = {.depthStencil = {1.0f, 0}},
    };

    VkRenderingInfo rendering_info{
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = scissor,
      .layerCount = 1,
      .colorAttachmentCount = 0,
      .pDepthAttachment = &depth_attachment,
    };

    vkCmdBeginRendering(cmd, &rendering_info);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pass.pipeline);
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);
    vkCmdSetDepthBias(cmd, 1.25f, 0.0f, 1.75f);

    VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
    VkBuffer current_index_buffer = VK_NULL_HANDLE;

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

      uint32_t offset = static_cast<uint32_t>(i * model_uniforms_alignment);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pass.pipeline_layout,
                              0, 1, &pass.descriptor_set, 1, &offset);

      vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
    }

    vkCmdEndRendering(cmd);

    transition_shadow_image(pass,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            VK_ACCESS_SHADER_READ_BIT,
                            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
  };

  VkViewport shadow_viewport{
    .x = 0.0f,
    .y = 0.0f,
    .width = static_cast<float>(ShadowPass::map_size),
    .height = static_cast<float>(ShadowPass::map_size),
    .minDepth = 0.0f,
    .maxDepth = 1.0f,
  };

  VkRect2D shadow_scissor{
    .offset = {0, 0},
    .extent = {ShadowPass::map_size, ShadowPass::map_size},
  };

  render_shadow_pass(shadow, shadow_viewport, shadow_scissor);

  for (size_t i = 0; i < active_spot_shadow_count; ++i) {
    render_shadow_pass(spot_shadows[i], shadow_viewport, shadow_scissor);
  }

  VkClearValue clear_color{.color = {{0.1f, 0.1f, 0.12f, 1.0f}}};
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

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

  VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
  VkBuffer current_index_buffer = VK_NULL_HANDLE;

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

    uint32_t offset = static_cast<uint32_t>(i * model_uniforms_alignment);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
                            0, 1, &model.descriptor_set, 1, &offset);

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
