#version 450

layout (location = 0) in vec3 v_position;

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
    mat4 light_view_projection;
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
    vec4 material_options;
} model_uniforms;

void main() {
    vec4 world_position = model_uniforms.model * vec4(v_position, 1.0f);
    gl_Position = scene.light_view_projection * world_position;
}
