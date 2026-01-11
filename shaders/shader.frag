#version 450

layout (location = 0) in vec3 f_position;
layout (location = 1) in vec3 f_normal;
layout (location = 2) in vec2 f_uv;

layout (location = 0) out vec4 final_color;

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

layout (binding = 2) uniform sampler2D model_texture;

vec3 calculateDirectional(vec3 normal, vec3 view_dir, vec3 diffuse_albedo, vec3 specular_color, float shininess) {
    vec3 light_dir = normalize(-scene.directional_light.direction_intensity.xyz);
    vec3 light_color = scene.directional_light.color.rgb * scene.directional_light.direction_intensity.w;
float diff = max(dot(normal, light_dir), 0.0f);
vec3 diffuse = diff * light_color * diffuse_albedo;
vec3 half_vector = normalize(light_dir + view_dir);
float spec = pow(max(dot(normal, half_vector), 0.0f), shininess);
vec3 specular = spec * light_color * specular_color;
return diffuse + specular;
}

vec3 calculateDiffuse(vec3 position, vec3 normal, vec3 diffuse_albedo) {
    vec3 offset = -position;
    float distance = length(offset);
    vec3 light_dir = distance > 0.0f ? offset / distance : vec3(0.0f, 1.0f, 0.0f);
    float attenuation = 1.0f / (1.0f + distance * distance);
    float intensity = max(scene.diffuse_color.w, 0.0f);
    vec3 light_color = scene.diffuse_color.rgb * intensity * attenuation;
float diff = max(dot(normal, light_dir), 0.0f);
return diff * light_color * diffuse_albedo;
}

vec3 calculatePoint(PointLight light, vec3 position, vec3 normal, vec3 view_dir, vec3 diffuse_albedo, vec3 specular_color, float shininess) {
    vec3 offset = light.position_intensity.xyz - position;
    float distance = length(offset);
    vec3 light_dir = distance > 0.0f ? offset / distance : vec3(0.0f, 1.0f, 0.0f);
    float attenuation = light.position_intensity.w / (1.0f + distance * distance);
    vec3 light_color = light.color.rgb * attenuation;
    float diff = max(dot(normal, light_dir), 0.0f);
    vec3 diffuse = diff * light_color * diffuse_albedo;
    vec3 half_vector = normalize(light_dir + view_dir);
    float spec = pow(max(dot(normal, half_vector), 0.0f), shininess);
    vec3 specular = spec * light_color * specular_color;
    return diffuse + specular;
}

void main() {
    vec3 normal = normalize(f_normal);
    if (model_uniforms.shadow_info.x > 0.5f) {
        float alpha = clamp(model_uniforms.shadow_info.w, 0.0f, 1.0f);
        final_color = vec4(vec3(0.0f), alpha);
        return;
    }

    vec3 texture_color = texture(model_texture, f_uv).rgb;
    vec3 ambient_albedo = model_uniforms.ambient_color.rgb * texture_color;
    vec3 diffuse_albedo = model_uniforms.diffuse_color.rgb * texture_color;
    vec3 specular_color = model_uniforms.specular_color_shininess.rgb;
    float shininess = max(model_uniforms.specular_color_shininess.w, 1.0f);
    vec3 view_dir = normalize(scene.camera_position.xyz - f_position);
    uint mode = uint(scene.light_mode.x + 0.5f);
    vec3 color = scene.ambient_color.rgb * ambient_albedo;

    if (mode == 0u) {
        color += calculateDiffuse(f_position, normal, diffuse_albedo);
    } else if (mode == 1u) {
        color += calculateDirectional(normal, view_dir, diffuse_albedo, specular_color, shininess);
    } else if (mode == 2u) {
        uint count = min(uint(scene.point_light_count.x), MAX_POINT_LIGHTS);
        for (uint i = 0; i < count; ++i) {
            color += calculatePoint(scene.point_lights[i], f_position, normal, view_dir, diffuse_albedo, specular_color, shininess);
        }
    }
final_color = vec4(color, 1.0f);
}
