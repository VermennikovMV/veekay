#version 450

layout (location = 0) in vec3 v_position;
layout (location = 1) in vec3 v_normal;
layout (location = 2) in vec2 v_uv;

layout (location = 0) out vec3 f_position;
layout (location = 1) out vec3 f_normal;
layout (location = 2) out vec2 f_uv;
layout (location = 3) out vec4 f_shadow_coord;
layout (location = 4) out vec4 f_spot_shadow0;
layout (location = 5) out vec4 f_spot_shadow1;

const uint MAX_SPOT_SHADOWS = 2u;

struct SceneUniforms {
    mat4 view_projection;
    mat4 shadow_view_projection;
    vec4 camera_position;
    vec4 ambient_color;
    vec4 directional_direction_intensity;
    vec4 directional_color;
    vec4 light_counts;
    mat4 spot_shadow_view_projections[MAX_SPOT_SHADOWS];
    vec4 spot_shadow_indices;
};

layout (binding = 0, std140) uniform SceneBuffer {
    SceneUniforms scene;
};

layout (binding = 1, std140) uniform ModelUniforms {
    mat4 model;
    vec4 albedo_shininess;
    vec4 specular_color;
} model_uniforms;

void main() {
    vec4 world_position = model_uniforms.model * vec4(v_position, 1.0);
    mat3 normal_matrix = transpose(inverse(mat3(model_uniforms.model)));

    f_position = world_position.xyz;
    f_normal = normalize(normal_matrix * v_normal);
    f_uv = v_uv;
    f_shadow_coord = scene.shadow_view_projection * world_position;
    f_spot_shadow0 = scene.spot_shadow_view_projections[0] * world_position;
    f_spot_shadow1 = scene.spot_shadow_view_projections[1] * world_position;

    gl_Position = scene.view_projection * world_position;
}
