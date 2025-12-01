#version 450

layout (location = 0) in vec3 f_position;
layout (location = 1) in vec3 f_normal;
layout (location = 2) in vec2 f_uv;

layout (location = 0) out vec4 final_color;

struct PointLight {
        vec3 position; float intensity;
        vec3 color; float _pad0;
};

layout (binding = 0, std140) uniform SceneUniforms {
        mat4 view_projection;
        vec3 camera_position; float ambient_strength;
        vec3 directional_direction; float directional_intensity;
        vec3 directional_color; float _pad1;
        PointLight point_lights[4];
        uint point_light_count; vec3 _pad2;
};

layout (binding = 1, std140) uniform ModelUniforms {
        mat4 model;
        vec3 albedo_color; float shininess;
};

void main() {
        vec3 normal = normalize(f_normal);
        vec3 view_direction = normalize(camera_position - f_position);

        vec3 color = albedo_color * ambient_strength;

        vec3 dir_direction = normalize(-directional_direction);
        float dir_intensity = max(dot(normal, dir_direction), 0.0f) * directional_intensity;

        vec3 halfway_dir = normalize(dir_direction + view_direction);
        float spec = pow(max(dot(normal, halfway_dir), 0.0f), shininess);
        vec3 specular = directional_color * spec * directional_intensity;

        color += albedo_color * directional_color * dir_intensity + specular;

        for (uint i = 0; i < point_light_count; ++i) {
                vec3 to_light = point_lights[i].position - f_position;
                float distance_sq = max(dot(to_light, to_light), 0.0001);
                float attenuation = point_lights[i].intensity / distance_sq;

                vec3 light_dir = normalize(to_light);
                float diff = max(dot(normal, light_dir), 0.0f);

                vec3 point_halfway = normalize(light_dir + view_direction);
                float point_spec = pow(max(dot(normal, point_halfway), 0.0f), shininess);

                vec3 diffuse_color = albedo_color * point_lights[i].color * diff * attenuation;
                vec3 specular_color = point_lights[i].color * point_spec * attenuation;

                color += diffuse_color + specular_color;
        }

        final_color = vec4(color, 1.0f);
}
