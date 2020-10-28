#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_ARB_shader_group_vote : enable


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
layout(location = 7) in vec3 lightPosition;
layout(location = 8) in vec3 pos;






layout(location = 0) out vec4 outColor;
#define M_PI 3.1415926535897932384626433832795

const float ambient = 0.15;


vec3 BlinnPhong(vec3 normal, vec3 diffuseColor, vec3 specularIntensity, vec3 lightColor, 
	vec3 lightDirection, vec3 viewDirection, vec3 lightPosition, vec3 fragPos, float reflectance)
{
	vec3 L = normalize(lightDirection);
	vec3 V = normalize(viewDirection);
	vec3 R = reflect(-L, normal);
	
	vec3 diffuse = lightColor * diffuseColor* max(dot(normal, L), ambient).rrr;
	vec3 specular = lightColor * specularIntensity* pow(max(dot(R, V), 0.0), reflectance);
	
	float distance = length(lightPosition - fragPos) / 1000;

	float attenuation  = 1.0 / distance;// / ( 1.0 + 0.007 * distance + 0.0002 * (distance * distance));

	diffuse *= attenuation;
	specular *= attenuation;

	return diffuse + specular;
}

void main() {
	vec3 lightColor = vec3(1, 1, 1);
	float LightPower = 5;
	
	MaterialData md = mats[matID];
	vec3 diffuseColor = texture(texSampler[md.albedoTexture], fragTexCoord).rgb;
	
	// Gamma correct
	//diffuseColor = pow(diffuseColor, vec3(1.0/2.2));

	vec3 normal;
	bool useBump = md.normalTexture >= 0;
	//useBump = allInvocationsARB(useBump);
	if (useBump)
	{
		vec3 N = normalize(inNormal);
		vec3 T = normalize(inTangent);
		vec3 B = cross(inNormal, inTangent);
		mat3 TBN = mat3(T, B, N);
		normal = normalize(TBN *texture(texSampler[md.normalTexture], fragTexCoord).rgb * 2.0 - vec3(1.0));
	 	// normal = normalize(vertNormal);
	}
	else 
	{
		normal = normalize(vertNormal);
	}



	vec3 specularIntensity;
	if (md.roughnessTexture >= 0)
	{
		specularIntensity = texture(texSampler[md.roughnessTexture], fragTexCoord).rgb;
	}
	else
	{
		specularIntensity = vec3(0.0);
	}

	

	vec3 result =  BlinnPhong(normal, diffuseColor, specularIntensity, lightColor, 
	inLightVec, inViewVec, lightPosition, pos, md.reflectance);

    outColor = vec4(result, 1.0);

}