#version 450

layout (location = 0) in vec3 v_position;
layout (location = 1) in vec3 v_normal;
layout (location = 2) in vec2 v_uv;

layout (location = 0) out vec3 f_position;
layout (location = 1) out vec3 f_normal;
layout (location = 2) out vec2 f_uv;

const uint MAX_POINT_LIGHTS = 4u;

struct DirectionalLight {
vec4 direction_intensity;
vec4 color;
};

struct PointLight {
vec4 position_intensity;
vec4 color;
};

layout (binding = 0, std140) uniform SceneUniforms {
    mat4 view_projection;
    vec4 camera_position;
    vec4 ambient_color;
    vec4 diffuse_color;
    vec4 light_mode;
    DirectionalLight directional_light;
    vec4 point_light_count;
    PointLight point_lights[MAX_POINT_LIGHTS];
} scene;

layout (binding = 1, std140) uniform ModelUniforms {
    mat4 model;
    vec4 ambient_color;
    vec4 diffuse_color;
    vec4 specular_color_shininess;
} model_uniforms;

layout (binding = 3, std430) buffer SharedStorageData {
    vec4 lighting_scale;
    vec4 uv_tiling_attenuation;
} shared_data;

void main() {
    vec4 position = model_uniforms.model * vec4(v_position, 1.0f);
    mat3 normal_matrix = transpose(inverse(mat3(model_uniforms.model)));
vec3 normal = normalize(normal_matrix * v_normal);

    gl_Position = scene.view_projection * position;

f_position = position.xyz;
f_normal = normal;
    float uv_scale = max(shared_data.uv_tiling_attenuation.x, 0.01f);
    f_uv = v_uv * uv_scale;
}
