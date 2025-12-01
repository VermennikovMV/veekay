#version 450

layout (location = 0) in vec3 f_position;
layout (location = 1) in vec3 f_normal;
layout (location = 2) in vec2 f_uv;

layout (location = 0) out vec4 final_color;

struct DirectionalLight {
        vec3 direction; float intensity;
        vec3 color; float _pad0;
};

struct PointLight {
        vec3 position; float intensity;
        vec3 color; float radius;
};

layout (binding = 0, std140) uniform SceneUniforms {
        mat4 view_projection;
        vec3 camera_position; float _pad0;
        vec3 ambient_color; float ambient_intensity;
        DirectionalLight directional_lights[2];
        PointLight point_lights[4];
        int directional_light_count;
        int point_light_count;
        vec2 _pad1;
};

layout (binding = 1, std140) uniform ModelUniforms {
        mat4 model;
        vec3 albedo_color; float shininess;
        vec3 specular_color; float _pad2;
};

void main() {
        vec3 normal = normalize(f_normal);
        vec3 view_dir = normalize(camera_position - f_position);

        vec3 color = ambient_color * ambient_intensity * albedo_color;

        for (int i = 0; i < directional_light_count; ++i) {
                DirectionalLight light = directional_lights[i];
                vec3 light_dir = normalize(-light.direction);
                vec3 radiance = light.color * light.intensity;

                float diffuse = max(dot(normal, light_dir), 0.0);
                vec3 halfway_dir = normalize(light_dir + view_dir);
                float specular_term = pow(max(dot(normal, halfway_dir), 0.0), shininess);

                color += (albedo_color * diffuse + specular_color * specular_term) * radiance;
        }

        for (int i = 0; i < point_light_count; ++i) {
                PointLight light = point_lights[i];
                vec3 to_light = light.position - f_position;
                float distance = length(to_light);
                vec3 light_dir = to_light / distance;

                float attenuation = light.intensity / (1.0 + (distance * distance) / max(light.radius * light.radius, 0.0001));
                vec3 radiance = light.color * attenuation;

                float diffuse = max(dot(normal, light_dir), 0.0);
                vec3 halfway_dir = normalize(light_dir + view_dir);
                float specular_term = pow(max(dot(normal, halfway_dir), 0.0), shininess);

                color += (albedo_color * diffuse + specular_color * specular_term) * radiance;
        }

        final_color = vec4(color, 1.0f);
}
