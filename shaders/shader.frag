#version 450

layout (location = 0) in vec3 f_position;
layout (location = 1) in vec3 f_normal;
layout (location = 2) in vec2 f_uv;

layout (location = 0) out vec4 final_color;

struct DirectionalLight {
        vec4 direction; // .w stores intensity
        vec4 color;
};

struct PointLight {
        vec4 position; // .w stores intensity
        vec4 color;
};

layout (binding = 1, std140) uniform ModelUniforms {
        mat4 model;
        vec3 albedo_color;
};

layout (binding = 0, std140) uniform SceneUniforms {
        mat4 view_projection;
        vec4 camera_position;
        vec4 ambient_color; // .w stores ambient strength

        DirectionalLight directional_light;
        uint point_light_count;
        vec3 _pad0;

        PointLight point_lights[4];
};

void main() {
        vec3 normal = normalize(f_normal);
        vec3 view_dir = normalize(camera_position.xyz - f_position);

        vec3 result = ambient_color.rgb * ambient_color.w;

        // Directional light
        vec3 light_dir = normalize(-directional_light.direction.xyz);
        float diff = max(dot(normal, light_dir), 0.0f);
        vec3 diffuse = directional_light.color.rgb * directional_light.direction.w * diff;

        vec3 halfway_dir = normalize(light_dir + view_dir);
        float spec = pow(max(dot(normal, halfway_dir), 0.0f), 32.0f);
        vec3 specular = directional_light.color.rgb * directional_light.direction.w * spec;
        result += diffuse + specular;

        // Point lights
        for (uint i = 0; i < point_light_count; ++i) {
                PointLight light = point_lights[i];

                vec3 direction = light.position.xyz - f_position;
                float distance = length(direction);
                vec3 point_dir = direction / distance;

                float attenuation = 1.0f / (distance * distance);
                float point_diff = max(dot(normal, point_dir), 0.0f);

                vec3 point_half = normalize(point_dir + view_dir);
                float point_spec = pow(max(dot(normal, point_half), 0.0f), 32.0f);

                vec3 light_color = light.color.rgb * light.position.w * attenuation;
                result += light_color * point_diff;
                result += light_color * point_spec;
        }

        result *= albedo_color;

        final_color = vec4(result, 1.0f);
}
