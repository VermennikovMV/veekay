#version 450

layout (location = 0) in vec3 v_position;

layout (binding = 0, std140) uniform ShadowScene {
    mat4 view_projection;
} shadow_scene;

layout (binding = 1, std140) uniform ModelUniforms {
    mat4 model;
    vec4 albedo_shininess;
    vec4 specular_color;
} model_uniforms;

void main() {
    gl_Position = shadow_scene.view_projection * model_uniforms.model * vec4(v_position, 1.0);
}
