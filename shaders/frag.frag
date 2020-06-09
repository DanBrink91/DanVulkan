#version 450
#extension GL_ARB_separate_shader_objects : enable
layout(binding = 1) uniform sampler2D texSampler;

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec3 fragNormal;
layout(location = 3) in vec3 fragPos;
layout(location = 4) in float time;

layout(location = 0) out vec4 outColor;
#define M_PI 3.1415926535897932384626433832795

void main() {
	vec4 color = texture(texSampler, fragTexCoord);
    outColor = color; 
}