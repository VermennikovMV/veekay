#version 450

layout (location = 0) in vec3 v_position;
layout (location = 1) in vec3 v_normal;
layout (location = 2) in vec2 v_uv;

layout (location = 0) out vec3 f_position;
layout (location = 1) out vec3 f_normal;
layout (location = 2) out vec2 f_uv;

struct DirectionalLight {
        vec4 direction; // .w stores intensity
        vec4 color;
};

struct PointLight {
        vec4 position; // .w stores intensity
        vec4 color;
};

layout (binding = 0, std140) uniform SceneUniforms {
        mat4 view_projection;
        vec4 camera_position;
        vec4 ambient_color;

        DirectionalLight directional_light;
        uint point_light_count;
        vec3 _pad0;

        PointLight point_lights[4];
};

layout (binding = 1, std140) uniform ModelUniforms {
	mat4 model;
	vec3 albedo_color;
};

void main() {
        vec4 position = model * vec4(v_position, 1.0f);
        vec3 normal = mat3(transpose(inverse(model))) * v_normal;

        gl_Position = view_projection * position;

        f_position = position.xyz;
        f_normal = normal;
        f_uv = v_uv;
}
