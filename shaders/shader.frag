#version 450

layout (location = 0) in vec3 f_position;
layout (location = 1) in vec3 f_normal;
layout (location = 2) in vec2 f_uv;

layout (location = 0) out vec4 final_color;

struct DirectionalLight {
        vec3 direction; float intensity;
        vec3 color; float _pad0;
};

layout (binding = 0, std140) uniform SceneUniforms {
        mat4 view_projection;
        vec3 camera_position; float ambient_strength;
        DirectionalLight directional_light;
} scene;

layout (binding = 1, std140) uniform ModelUniforms {
        mat4 model;
        vec3 albedo_color; float _pad0;
};

void main() {
        vec3 base_color = albedo_color;
        vec3 normal = normalize(f_normal);
        vec3 view_direction = normalize(scene.camera_position - f_position);

        vec3 color = scene.ambient_strength * base_color;

        vec3 directional_dir = normalize(-scene.directional_light.direction);
        float directional_diff = max(dot(normal, directional_dir), 0.0f);
        vec3 directional_halfway = normalize(directional_dir + view_direction);
        float directional_spec = pow(max(dot(normal, directional_halfway), 0.0f), 32.0f);
        color += scene.directional_light.color * scene.directional_light.intensity *
                 (directional_diff * base_color + directional_spec);

        final_color = vec4(color, 1.0f);
}
