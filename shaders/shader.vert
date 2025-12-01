#version 450

layout (location = 0) in vec3 v_position;
layout (location = 1) in vec3 v_normal;
layout (location = 2) in vec2 v_uv;

layout (location = 0) out vec3 f_position;
layout (location = 1) out vec3 f_normal;
layout (location = 2) out vec2 f_uv;

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
	vec4 position = model * vec4(v_position, 1.0f);
	vec4 normal = model * vec4(v_normal, 0.0f);

	gl_Position = view_projection * position;

	f_position = position.xyz;
	f_normal = normal.xyz;
	f_uv = v_uv;
}
