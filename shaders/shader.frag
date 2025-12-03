#version 450

layout (location = 0) in vec3 f_position;
layout (location = 1) in vec3 f_normal;
layout (location = 2) in vec2 f_uv;
layout (location = 3) in vec4 f_shadow_coord;
layout (location = 4) in vec4 f_spot_shadow0;
layout (location = 5) in vec4 f_spot_shadow1;

layout (location = 0) out vec4 final_color;

const uint MAX_POINT_LIGHTS = 16u;
const uint MAX_SPOT_LIGHTS = 8u;
const uint MAX_SPOT_SHADOWS = 2u;

struct PointLightGpu {
    vec4 position_intensity;
    vec4 color_radius;
};

struct SpotLightGpu {
    vec4 position_intensity;
    vec4 direction_inner_cos;
    vec4 color_range;
    vec4 params; // x - outer cos
};

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

layout (binding = 2) readonly buffer PointLights {
    PointLightGpu point_lights[];
};

layout (binding = 3) readonly buffer SpotLights {
    SpotLightGpu spot_lights[];
};

layout (binding = 4) uniform sampler2D albedo_texture;
layout (binding = 5) uniform sampler2DShadow shadow_map;
layout (binding = 6) uniform sampler2DShadow spot_shadow_maps[MAX_SPOT_SHADOWS];

float sample_shadow(vec4 shadow_coord, sampler2DShadow map_sampler, float bias)
{
    vec3 proj = shadow_coord.xyz / shadow_coord.w;
    proj = proj * 0.5 + 0.5;
    if (proj.x < 0.0 || proj.x > 1.0 || proj.y < 0.0 || proj.y > 1.0)
        return 1.0;
    return texture(map_sampler, vec3(proj.xy, proj.z - bias));
}

vec3 calculate_directional(vec3 normal, vec3 view_dir, vec3 albedo, vec3 specular_color, float shininess)
{
    vec3 light_dir = normalize(-scene.directional_direction_intensity.xyz);
    vec3 light_color = scene.directional_color.rgb * scene.directional_direction_intensity.w;
    float diff = max(dot(normal, light_dir), 0.0);
    vec3 diffuse = diff * light_color * albedo;
    vec3 half_vector = normalize(light_dir + view_dir);
    float spec = pow(max(dot(normal, half_vector), 0.0), shininess);
    vec3 specular = spec * light_color * specular_color;

    float shadow_factor = sample_shadow(f_shadow_coord, shadow_map, 0.0015);
    return (diffuse + specular) * shadow_factor;
}

vec3 calculate_point(PointLightGpu light, vec3 position, vec3 normal, vec3 view_dir, vec3 albedo, vec3 specular_color, float shininess)
{
    vec3 offset = light.position_intensity.xyz - position;
    float distance = length(offset);
    float range = max(light.color_radius.w, 0.001);
    float attenuation = clamp(1.0 - distance / range, 0.0, 1.0);
    attenuation *= attenuation;
    vec3 light_dir = distance > 0.0 ? offset / distance : vec3(0.0, 1.0, 0.0);

    vec3 light_color = light.color_radius.rgb * light.position_intensity.w * attenuation;
    float diff = max(dot(normal, light_dir), 0.0);
    vec3 diffuse = diff * light_color * albedo;
    vec3 half_vector = normalize(light_dir + view_dir);
    float spec = pow(max(dot(normal, half_vector), 0.0), shininess);
    vec3 specular = spec * light_color * specular_color;
    return diffuse + specular;
}

vec3 calculate_spot(SpotLightGpu light, vec3 position, vec3 normal, vec3 view_dir, vec3 albedo, vec3 specular_color, float shininess, float shadow_factor)
{
    vec3 offset = light.position_intensity.xyz - position;
    float distance = length(offset);
    float range = max(light.color_range.w, 0.001);
    vec3 light_dir = distance > 0.0 ? offset / distance : vec3(0.0, 1.0, 0.0);

    float attenuation = clamp(1.0 - distance / range, 0.0, 1.0);
    attenuation *= attenuation;

    float cos_theta = dot(light_dir, -light.direction_inner_cos.xyz);
    float intensity = smoothstep(light.params.x, light.direction_inner_cos.w, cos_theta);
    vec3 light_color = light.color_range.rgb * light.position_intensity.w * attenuation * intensity;

    float diff = max(dot(normal, light_dir), 0.0);
    vec3 diffuse = diff * light_color * albedo;
    vec3 half_vector = normalize(light_dir + view_dir);
    float spec = pow(max(dot(normal, half_vector), 0.0), shininess);
    vec3 specular = spec * light_color * specular_color;
    return (diffuse + specular) * shadow_factor;
}

void main() {
    vec3 normal = normalize(f_normal);
    vec3 view_dir = normalize(scene.camera_position.xyz - f_position);
    float shininess = max(model_uniforms.albedo_shininess.w, 1.0);
    vec3 albedo = model_uniforms.albedo_shininess.rgb * texture(albedo_texture, f_uv).rgb;
    vec3 specular_color = model_uniforms.specular_color.rgb;

    vec3 color = scene.ambient_color.rgb * scene.ambient_color.a * albedo;

    color += calculate_directional(normal, view_dir, albedo, specular_color, shininess);

    uint point_count = uint(scene.light_counts.x + 0.5);
    for (uint i = 0; i < point_count && i < MAX_POINT_LIGHTS; ++i) {
        color += calculate_point(point_lights[i], f_position, normal, view_dir, albedo, specular_color, shininess);
    }

    uint spot_count = uint(scene.light_counts.y + 0.5);
    int shadow_index0 = int(scene.spot_shadow_indices.x + 0.5);
    int shadow_index1 = int(scene.spot_shadow_indices.y + 0.5);
    int active_shadow_count = int(scene.spot_shadow_indices.z + 0.5);

    for (uint i = 0; i < spot_count && i < MAX_SPOT_LIGHTS; ++i) {
        float shadow_factor = 1.0;
        if (active_shadow_count > 0 && int(i) == shadow_index0)
            shadow_factor = sample_shadow(f_spot_shadow0, spot_shadow_maps[0], 0.002);
        else if (active_shadow_count > 1 && int(i) == shadow_index1)
            shadow_factor = sample_shadow(f_spot_shadow1, spot_shadow_maps[1], 0.002);

        color += calculate_spot(spot_lights[i], f_position, normal, view_dir, albedo, specular_color, shininess, shadow_factor);
    }

    final_color = vec4(color, 1.0);
}
