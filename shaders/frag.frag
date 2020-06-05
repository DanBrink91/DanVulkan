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
	
	// -0.7 -> 0.7 on this model
	float xToAngle = (fragPos.x + 0.3) * 2 * M_PI;  
	
	float shoreline = 0.1 + sin(xToAngle + time) * 0.03;
	float axis = fragPos.z;

	if(abs(axis - shoreline) < 0.003)
	{
		color = vec4(1.0, 1.0, 1.0, 1.0);
	}
	else if (axis < shoreline)
	{
		color.b += 0.35;
	}
    outColor = color; 
}