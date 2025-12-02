#version 450

layout (location = 0) in vec3 v_position;
layout (location = 1) in vec3 v_normal;
layout (location = 2) in vec2 v_uv;

layout (location = 0) out vec3 f_position;
layout (location = 1) out vec3 f_normal;
layout (location = 2) out vec2 f_uv;

layout(push_constant) uniform PushConstants {
    uint draw_shadow;
    float ground_height;
} push_constants;

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

void main() {
    vec4 position = model_uniforms.model * vec4(v_position, 1.0f);
    mat3 normal_matrix = transpose(inverse(mat3(model_uniforms.model)));
    vec3 normal = normalize(normal_matrix * v_normal);

    if (push_constants.draw_shadow == 1u) {
        vec3 projected = position.xyz;
        float bias = 0.0015f;
        uint mode = uint(scene.light_mode.x + 0.5f);

        if (mode == 1u) { // Directional
            vec3 light_dir = normalize(-scene.directional_light.direction_intensity.xyz);
            float denom = light_dir.y;
            if (abs(denom) > 1e-4f) {
                float t = (push_constants.ground_height - projected.y) / denom;
                projected += light_dir * t;
            }
        } else { // Diffuse (origin) or point
            vec3 light_pos = vec3(0.0f);
            if (mode == 2u && uint(scene.point_light_count.x + 0.5f) > 0u) {
                light_pos = scene.point_lights[0].position_intensity.xyz;
            }

            vec3 dir = projected - light_pos;
            float denom = dir.y;
            if (abs(denom) > 1e-4f) {
                float t = (push_constants.ground_height - light_pos.y) / denom;
                projected = light_pos + dir * t;
            }
        }

        projected.y = push_constants.ground_height + bias;
        position = vec4(projected, 1.0f);
        normal = vec3(0.0f, 1.0f, 0.0f);
    }

    gl_Position = scene.view_projection * position;

    f_position = position.xyz;
    f_normal = normal;
    f_uv = v_uv;
}
