#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable


layout(binding = 5) uniform sampler2D texSampler[];

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) flat in int matID;



layout(location = 0) out vec4 outColor;
#define M_PI 3.1415926535897932384626433832795

void main() {
    outColor = vec4(texture(texSampler[matID], fragTexCoord).rgb, 1.0);
}