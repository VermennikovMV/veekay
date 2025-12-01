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
};

layout (binding = 1, std140) uniform ModelUniforms {
mat4 model;
vec4 albedo_specular;
vec4 material_params;
};

vec3 calculateDirectional(vec3 normal, vec3 view_dir, vec3 albedo, float specular_strength, float shininess) {
vec3 light_dir = normalize(-directional_light.direction_intensity.xyz);
vec3 light_color = directional_light.color.rgb * directional_light.direction_intensity.w;
float diff = max(dot(normal, light_dir), 0.0f);
vec3 diffuse = diff * light_color * albedo;
vec3 half_vector = normalize(light_dir + view_dir);
float spec = pow(max(dot(normal, half_vector), 0.0f), shininess);
vec3 specular = specular_strength * spec * light_color;
return diffuse + specular;
}

vec3 calculatePoint(PointLight light, vec3 position, vec3 normal, vec3 view_dir, vec3 albedo, float specular_strength, float shininess) {
vec3 offset = light.position_intensity.xyz - position;
float distance = length(offset);
vec3 light_dir = offset / max(distance, 0.0001f);
 float attenuation = light.position_intensity.w / (1.0f + distance * distance);
vec3 light_color = light.color.rgb * attenuation;
float diff = max(dot(normal, light_dir), 0.0f);
vec3 diffuse = diff * light_color * albedo;
vec3 half_vector = normalize(light_dir + view_dir);
float spec = pow(max(dot(normal, half_vector), 0.0f), shininess);
vec3 specular = specular_strength * spec * light_color;
return diffuse + specular;
}

void main() {
vec3 normal = normalize(f_normal);
vec3 albedo = albedo_specular.rgb;
float specular_strength = albedo_specular.w;
float shininess = max(material_params.x, 1.0f);
vec3 view_dir = normalize(camera_position.xyz - f_position);
uint mode = uint(light_mode.x + 0.5f);
vec3 color = ambient_color.rgb * albedo;

if (mode == 0u) {
color += diffuse_color.rgb * diffuse_color.w * albedo;
} else if (mode == 1u) {
color += calculateDirectional(normal, view_dir, albedo, specular_strength, shininess);
} else if (mode == 2u) {
uint count = min(uint(point_light_count.x), MAX_POINT_LIGHTS);
for (uint i = 0; i < count; ++i) {
color += calculatePoint(point_lights[i], f_position, normal, view_dir, albedo, specular_strength, shininess);
}
}
final_color = vec4(color, 1.0f);
}
