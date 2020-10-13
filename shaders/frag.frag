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

layout(location = 0) in vec3 vertNormal;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in flat int matID;
layout(location = 3) in vec3 inLightVec;
layout(location = 4) in vec3 inViewVec;
layout(location = 5) in vec3 inNormal;
layout(location = 6) in vec3 inTangent;



layout(location = 0) out vec4 outColor;
#define M_PI 3.1415926535897932384626433832795

vec3 BlinnPhong(vec3 normal, vec3 positon_worldspace, vec3 lightPosition_worldspace,
 vec3 lightDirection_tangentspace, vec3 eyeDirection_tangentspace,
 vec3 lightColor, vec3 diffuseColor, vec3 specularIntensity)
{
	vec3 lightDir = normalize(lightDirection_tangentspace);
	float diff = clamp(dot(normal, lightDir), 0, 1);

	float distance = length(lightPosition_worldspace - positon_worldspace);
	vec3 diffuse = diff  * diffuseColor * lightColor;

	// 
	vec3 E = normalize(eyeDirection_tangentspace);

	vec3 R = reflect(-lightDir, normal);
	float spec = pow(clamp(dot(E, R), 0, 1), 32);
	vec3 specular =  spec * specularIntensity * lightColor ;

	float attenuation  =  0.1;//(1.0 /( 1.0 + 0.007 * distance + 0.0002 * (distance * distance)));

	diffuse *= attenuation;
	specular *= attenuation;

	return diffuse;
}

void main() {
	vec3 lightColor = vec3(1);
	float LightPower = 5;
	
	MaterialData md = mats[matID];
	vec3 diffuseColor = texture(texSampler[md.albedoTexture], fragTexCoord).rgb;
	// Gamma correct
	//diffuseColor = pow(diffuseColor, vec3(1.0/2.2));

	vec3 normal;
	if (md.normalTexture >= 0)
	{
		vec3 N = normalize(inNormal);
		vec3 T = normalize(inTangent);
		vec3 B = cross(inNormal, inTangent);
		mat3 TBN = mat3(T, B, N);
		normal = normalize(TBN *texture(texSampler[md.normalTexture], fragTexCoord).rgb * 2.0 - vec3(1.0));
	 	// normal = vertNormal;
	}
	else 
	{
		normal = vertNormal;
	}


	const float ambient = 0.1;

	vec3 specularIntensity;
	if (md.roughnessTexture >= 0)
	{
		specularIntensity = texture(texSampler[md.roughnessTexture], fragTexCoord).rgb;
	}
	else
	{
		specularIntensity = vec3(0.0);
	}

	vec3 L = normalize(inLightVec);
	vec3 V = normalize(inViewVec);
	vec3 R = reflect(-L, normal);

	vec3 diffuse = max(dot(normal, L), ambient).rrr;
	vec3 specular = specularIntensity * pow(max(dot(R, V), 0.0), md.reflectance);

	vec3 result = diffuse *  diffuseColor + specular;
	// Gamma corrected
	// result = pow(diffuseColor, vec3(1.0/2.2));
	// result = normal;
    outColor = vec4(result, 1.0);
}