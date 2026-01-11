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
    vec4 shadow_info;
} model_uniforms;

void main() {
    vec4 position = model_uniforms.model * vec4(v_position, 1.0f);
    bool is_shadow = model_uniforms.shadow_info.x > 0.5f;

    if (is_shadow) {
        vec3 light_raw = -scene.directional_light.direction_intensity.xyz;
        float length_light = length(light_raw);
        vec3 light_dir = length_light > 1e-4f ? light_raw / length_light : vec3(0.0f, -1.0f, 0.0f);
        float denom = light_dir.y;
        float safe_denom = abs(denom) < 1e-4f ? (denom < 0.0f ? -1e-4f : 1e-4f) : denom;
        float t = (model_uniforms.shadow_info.y - position.y) / safe_denom;
        vec3 projected = position.xyz + light_dir * t;
        projected.y += model_uniforms.shadow_info.z;

        position = vec4(projected, 1.0f);
    }

    mat3 normal_matrix = transpose(inverse(mat3(model_uniforms.model)));
vec3 normal = normalize(normal_matrix * v_normal);
    if (is_shadow) {
        normal = vec3(0.0f, 1.0f, 0.0f);
    }

    gl_Position = scene.view_projection * position;

f_position = position.xyz;
f_normal = normal;
f_uv = v_uv;
}
