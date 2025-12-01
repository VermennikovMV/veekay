#version 450

layout (location = 0) in vec3 v_position;
layout (location = 1) in vec3 v_normal;
layout (location = 2) in vec2 v_uv;

layout (location = 0) out vec3 f_position;
layout (location = 1) out vec3 f_normal;
layout (location = 2) out vec2 f_uv;

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
        vec3 specular_color; float _pad0;
};

void main() {
	vec4 position = model * vec4(v_position, 1.0f);
	vec4 normal = model * vec4(v_normal, 0.0f);

	gl_Position = view_projection * position;

	f_position = position.xyz;
	f_normal = normal.xyz;
	f_uv = v_uv;
}
