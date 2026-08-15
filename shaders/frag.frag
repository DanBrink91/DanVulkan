#version 460
#extension GL_EXT_nonuniform_qualifier : require

const float Pi = 3.14159265358979323846;
const int AlphaOpaque = 0;
const int AlphaMask = 1;
const int AlphaBlend = 2;

layout(set = 0, binding = 0) uniform UniformBufferObject
{
    mat4 view;
    mat4 projection;
    vec4 cameraPositionTime;
    uvec4 lightingCounts;
    vec4 environmentTintIntensity;
    vec4 environmentControls;
} ubo;

layout(set = 0, binding = 5) uniform sampler2D textures[];
layout(set = 0, binding = 8) uniform sampler2D irradianceMap;
layout(set = 0, binding = 9) uniform sampler2D prefilteredSpecularMap;
layout(set = 0, binding = 10) uniform sampler2D environmentBrdfMap;

struct PointLightData
{
    vec4 positionRange;
    vec4 colorIntensity;
};

layout(set = 0, binding = 7) readonly buffer PointLights
{
    PointLightData pointLights[];
};

struct MaterialData
{
    vec4 baseColorFactor;
    vec4 emissiveMetallic;
    vec4 roughnessNormalOcclusionAlpha;
    vec4 textureTiling;
    ivec4 textureIndices;
    ivec4 materialFlags;
};

layout(set = 0, binding = 1) readonly buffer Materials
{
    MaterialData materials[];
};

layout(location = 0) in vec2 inTexCoord;
layout(location = 1) flat in int inMaterialIndex;
layout(location = 2) in vec3 inWorldNormal;
layout(location = 3) in vec4 inWorldTangent;
layout(location = 4) in vec3 inWorldPosition;

layout(location = 0) out vec4 outColor;

float distributionGgx(vec3 normal, vec3 halfway, float roughness)
{
    float alpha = roughness * roughness;
    float alphaSquared = alpha * alpha;
    float nDotH = max(dot(normal, halfway), 0.0);
    float denominator = nDotH * nDotH * (alphaSquared - 1.0) + 1.0;
    return alphaSquared / max(Pi * denominator * denominator, 0.0001);
}

float geometrySchlickGgx(float nDotDirection, float roughness)
{
    float radius = roughness + 1.0;
    float k = radius * radius / 8.0;
    return nDotDirection / max(nDotDirection * (1.0 - k) + k, 0.0001);
}

float geometrySmith(vec3 normal, vec3 viewDirection, vec3 lightDirection, float roughness)
{
    return geometrySchlickGgx(max(dot(normal, viewDirection), 0.0), roughness) *
           geometrySchlickGgx(max(dot(normal, lightDirection), 0.0), roughness);
}

vec3 fresnelSchlick(float cosTheta, vec3 reflectance)
{
    return reflectance + (1.0 - reflectance) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 fresnelSchlickRoughness(float cosTheta, vec3 reflectance, float roughness)
{
    return reflectance + (max(vec3(1.0 - roughness), reflectance) - reflectance) *
        pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec2 environmentUv(vec3 direction)
{
    direction = normalize(direction);
    float rotation = ubo.environmentControls.x;
    float sine = sin(rotation);
    float cosine = cos(rotation);
    direction.xz = mat2(cosine, -sine, sine, cosine) * direction.xz;
    return vec2(atan(direction.z, direction.x) / (2.0 * Pi) + 0.5,
        acos(clamp(direction.y, -1.0, 1.0)) / Pi);
}

vec3 materialNormal(MaterialData material, vec2 uv)
{
    vec3 normal = normalize(inWorldNormal);
    if (material.materialFlags.z != 0 && !gl_FrontFacing)
    {
        normal = -normal;
    }
    if (material.textureIndices.y < 0)
    {
        return normal;
    }

    vec3 tangent = normalize(inWorldTangent.xyz -
        normal * dot(normal, inWorldTangent.xyz));
    vec3 bitangent = normalize(cross(normal, tangent)) * inWorldTangent.w;
    vec3 sampled = texture(textures[nonuniformEXT(material.textureIndices.y)], uv).xyz * 2.0 - 1.0;
    sampled.xy *= material.roughnessNormalOcclusionAlpha.y;
    return normalize(mat3(tangent, bitangent, normal) * sampled);
}

void main()
{
    MaterialData material = materials[inMaterialIndex];
    vec2 uv = inTexCoord * material.textureTiling.xy;

    vec4 baseColor = material.baseColorFactor;
    if (material.textureIndices.x >= 0)
    {
        baseColor *= texture(textures[nonuniformEXT(material.textureIndices.x)], uv);
    }

    int alphaMode = material.materialFlags.y;
    if (alphaMode == AlphaMask && baseColor.a < material.roughnessNormalOcclusionAlpha.w)
    {
        discard;
    }

    if (material.materialFlags.w != 0)
    {
        outColor = vec4(baseColor.rgb, alphaMode == AlphaBlend ? baseColor.a : 1.0);
        return;
    }

    float metallic = material.emissiveMetallic.w;
    float roughness = material.roughnessNormalOcclusionAlpha.x;
    if (material.textureIndices.z >= 0)
    {
        vec4 packed = texture(textures[nonuniformEXT(material.textureIndices.z)], uv);
        roughness *= packed.g;
        metallic *= packed.b;
    }
    metallic = clamp(metallic, 0.0, 1.0);
    roughness = clamp(roughness, 0.045, 1.0);

    float occlusion = 1.0;
    if (material.textureIndices.w >= 0)
    {
        float sampledOcclusion = texture(textures[nonuniformEXT(material.textureIndices.w)], uv).r;
        occlusion = mix(1.0, sampledOcclusion,
            clamp(material.roughnessNormalOcclusionAlpha.z, 0.0, 1.0));
    }

    vec3 emissive = material.emissiveMetallic.xyz;
    if (material.materialFlags.x >= 0)
    {
        emissive *= texture(textures[nonuniformEXT(material.materialFlags.x)], uv).rgb;
    }

    vec3 normal = materialNormal(material, uv);
    vec3 viewDirection = normalize(ubo.cameraPositionTime.xyz - inWorldPosition);
    vec3 reflectance = mix(vec3(0.04), baseColor.rgb, metallic);
    float nDotV = max(dot(normal, viewDirection), 0.0);
    vec3 direct = vec3(0.0);
    for (uint lightIndex = 0; lightIndex < ubo.lightingCounts.x; ++lightIndex)
    {
        PointLightData light = pointLights[lightIndex];
        vec3 lightOffset = light.positionRange.xyz - inWorldPosition;
        float lightDistanceSquared = max(dot(lightOffset, lightOffset), 0.01);
        float lightDistance = sqrt(lightDistanceSquared);
        vec3 lightDirection = lightOffset / lightDistance;
        vec3 halfway = normalize(viewDirection + lightDirection);
        vec3 fresnel = fresnelSchlick(max(dot(halfway, viewDirection), 0.0), reflectance);
        float distribution = distributionGgx(normal, halfway, roughness);
        float geometry = geometrySmith(normal, viewDirection, lightDirection, roughness);
        vec3 specular = distribution * geometry * fresnel /
            max(4.0 * nDotV * max(dot(normal, lightDirection), 0.0), 0.0001);
        vec3 diffuseWeight = (1.0 - fresnel) * (1.0 - metallic);
        float rangeWeight = 1.0;
        if (light.positionRange.w > 0.0)
        {
            float normalizedDistance = lightDistance / light.positionRange.w;
            rangeWeight = pow(clamp(1.0 - pow(normalizedDistance, 4.0), 0.0, 1.0), 2.0);
        }
        vec3 radiance = light.colorIntensity.rgb * light.colorIntensity.w *
            rangeWeight / lightDistanceSquared;
        float nDotL = max(dot(normal, lightDirection), 0.0);
        direct += (diffuseWeight * baseColor.rgb / Pi + specular) * radiance * nDotL;
    }

    float maximumEnvironmentLod =
        float(max(textureQueryLevels(prefilteredSpecularMap) - 1, 0));
    vec3 environmentFresnel = fresnelSchlickRoughness(nDotV, reflectance, roughness);
    vec3 environmentDiffuse = textureLod(irradianceMap, environmentUv(normal), 0.0).rgb;
    vec3 reflection = reflect(-viewDirection, normal);
    vec3 environmentSpecular = textureLod(prefilteredSpecularMap,
        environmentUv(reflection), roughness * maximumEnvironmentLod).rgb;
    vec2 environmentBrdf = texture(environmentBrdfMap, vec2(nDotV, roughness)).rg;
    vec3 diffuseEnvironment = (1.0 - environmentFresnel) * (1.0 - metallic) *
        baseColor.rgb * environmentDiffuse / Pi * ubo.environmentControls.y;
    vec3 specularEnvironment = environmentSpecular *
        (environmentFresnel * environmentBrdf.x + environmentBrdf.y) *
        ubo.environmentControls.z;
    vec3 ambient = (diffuseEnvironment + specularEnvironment) *
        ubo.environmentTintIntensity.rgb * ubo.environmentTintIntensity.w * occlusion;
    vec3 color = ambient + direct + emissive;
    color = color / (color + vec3(1.0));

    outColor = vec4(color, alphaMode == AlphaBlend ? baseColor.a : 1.0);
}
