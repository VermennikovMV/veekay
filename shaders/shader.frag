#version 450

layout (location = 0) in vec3 f_position;
layout (location = 1) in vec3 f_normal;
layout (location = 2) in vec2 f_uv;

layout (location = 0) out vec4 final_color;

const uint MAX_POINT_LIGHTS = 8;

layout (binding = 0, std140) uniform SceneUniforms {
        mat4 view_projection;
        vec3 camera_position; float ambient_strength;
        struct {
                vec3 direction; float intensity;
                vec3 color; float _pad0;
        } directional_light;
        uint point_light_count; vec3 _pad1;
        struct {
                vec3 position; float intensity;
                vec3 color; float _pad0;
        } point_lights[MAX_POINT_LIGHTS];
} scene;

layout (binding = 1, std140) uniform ModelUniforms {
        mat4 model;
        vec3 albedo_color;
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

        for (uint i = 0; i < scene.point_light_count; ++i) {
                vec3 light_vector = scene.point_lights[i].position - f_position;
                float distance_sq = max(dot(light_vector, light_vector), 0.0001f);
                vec3 light_dir = light_vector * inversesqrt(distance_sq);
                float attenuation = scene.point_lights[i].intensity / distance_sq;

                float diff = max(dot(normal, light_dir), 0.0f);
                vec3 halfway_dir = normalize(light_dir + view_direction);
                float spec = pow(max(dot(normal, halfway_dir), 0.0f), 32.0f);

                color += scene.point_lights[i].color * attenuation * (diff * base_color + spec);
        }

        final_color = vec4(color, 1.0f);
}
