#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable


layout(binding = 5) uniform sampler2D texSampler[];

struct MaterialData
{
    vec4 albedoTint;

    float tilingX;
    float tilingY;
    float reflectance;
    float unused0; // pad to vec4

    int albedoTexture;
    int normalTexture;
    int roughnessTexture;
    int unused01; // pad to vec4
};

layout(set = 0, binding = 1) readonly buffer materialData
{
    MaterialData mats[];
};
layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) flat in int matID;
layout(location = 3) in vec3 fragPos;
layout(location = 4) in vec3 viewPos;
layout(location = 5) in vec3 lightPos;




layout(location = 0) out vec4 outColor;
#define M_PI 3.1415926535897932384626433832795


void main() {
	// vec3 viewPos = vec3(1010, 0, 123230);
	// vec3 lightPos = vec3(1, 0, 0);
	MaterialData md = mats[matID];
	vec3 diffuseColor = texture(texSampler[md.albedoTexture], fragTexCoord).rgb;
	
	vec3 normal =  texture(texSampler[md.normalTexture], fragTexCoord).rgb;
	normal = normalize(normal * 2.0 - 1.0);

	vec3 ambient = diffuseColor * 0.1   ;

	vec3 lightDir = normalize(lightPos - fragPos);
	float diff = max(dot(lightDir, normal), 0.0);
	vec3 diffuse = (diff  * diffuseColor);


	vec3 result = ambient + diffuse;
	if (md.roughnessTexture >= 0)
	{
		vec3 specularColor = texture(texSampler[md.roughnessTexture], fragTexCoord).rgb;

		vec3 viewDir = normalize(viewPos - fragPos); 
		vec3 reflectDir = reflect(-lightDir, normal);
		vec3 halfwayDir = normalize(lightDir + viewDir);
		float spec = pow(max(dot(normal, halfwayDir), 0.0), md.reflectance);
		
		vec3 specular =  spec * specularColor * 0.2;
		result += specular;
	}
    outColor = vec4(result, 1.0);
}