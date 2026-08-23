#version 460
#extension GL_EXT_nonuniform_qualifier : require

const float Pi = 3.14159265358979323846;
const int AlphaOpaque = 0;
const int AlphaMask = 1;
const int AlphaBlend = 2;

struct DirectionalLightData
{
    vec4 directionIntensity;
    vec4 colorShadow;
};

layout(set = 0, binding = 0) uniform UniformBufferObject
{
    mat4 view;
    mat4 projection;
    mat4 inverseViewProjection;
    mat4 directionalShadowViewProjection;
    vec4 cameraPositionTime;
    vec4 vegetationInteractorPositionRadius;
    uvec4 lightingCounts;
    vec4 environmentTintIntensity;
    vec4 environmentControls;
    vec4 atmosphereSkyZenithIntensity;
    vec4 atmosphereSkyHorizonExponent;
    vec4 atmosphereFogColorDensity;
    vec4 atmosphereFogParameters;
    vec4 atmosphereScatteringParameters;
    DirectionalLightData directionalLights[4];
    vec4 atmosphereCloudShapeParameters;
    vec4 atmosphereCloudMovementParameters;
    vec4 atmosphereCloudLightingParameters;
} ubo;

layout(set = 0, binding = 5) uniform sampler2D textures[];
layout(set = 0, binding = 8) uniform sampler2D irradianceMap;
layout(set = 0, binding = 9) uniform sampler2D prefilteredSpecularMap;
layout(set = 0, binding = 10) uniform sampler2D environmentBrdfMap;
layout(set = 0, binding = 11) uniform sampler2DShadow directionalShadowMap;

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
layout(location = 5) in vec3 inVertexColor;
layout(location = 6) in float inSurfaceCoverage;

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
    vec3 bitangent = normalize(cross(normal, tangent)) * sign(inWorldTangent.w);
    vec3 sampled = texture(textures[nonuniformEXT(material.textureIndices.y)], uv).xyz * 2.0 - 1.0;
    sampled.xy *= material.roughnessNormalOcclusionAlpha.y;
    return normalize(mat3(tangent, bitangent, normal) * sampled);
}

float hash21(vec2 point)
{
    vec3 p = fract(vec3(point.xyx) * vec3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}

float valueNoise(vec2 point)
{
    vec2 cell = floor(point);
    vec2 local = fract(point);
    vec2 blend = local * local * (3.0 - 2.0 * local);
    float top = mix(hash21(cell), hash21(cell + vec2(1.0, 0.0)), blend.x);
    float bottom = mix(hash21(cell + vec2(0.0, 1.0)),
        hash21(cell + vec2(1.0, 1.0)), blend.x);
    return mix(top, bottom, blend.y);
}

float fractalNoise(vec2 point)
{
    float result = valueNoise(point) * 0.5714;
    result += valueNoise(point * 2.03 + vec2(17.1, 9.2)) * 0.2857;
    result += valueNoise(point * 4.11 + vec2(-5.7, 23.4)) * 0.1429;
    return result;
}

float cloudLayerDensity(vec2 worldPosition, float scaleMultiplier,
    float coverageBias, vec2 layerOffset)
{
    float coverage = clamp(ubo.atmosphereCloudShapeParameters.x + coverageBias, 0.0, 1.0);
    float density = ubo.atmosphereCloudShapeParameters.y;
    if (coverage <= 0.001 || density <= 0.001)
    {
        return 0.0;
    }
    vec2 windOffset = ubo.atmosphereCloudMovementParameters.xy *
        ubo.cameraPositionTime.w * ubo.atmosphereCloudMovementParameters.z;
    vec2 point = (worldPosition - windOffset) *
        ubo.atmosphereCloudShapeParameters.z * scaleMultiplier + layerOffset;
    float field = valueNoise(point) * 0.57;
    field += valueNoise(point * 2.03 + vec2(13.7, -8.2)) * 0.29;
    field += valueNoise(point * 4.11 + vec2(-4.3, 19.6)) * 0.14;
    float threshold = 1.0 - coverage;
    float softness = ubo.atmosphereCloudShapeParameters.w;
    float shape = smoothstep(threshold - softness, threshold + softness, field);
    return clamp(shape * density, 0.0, 1.0);
}

float cloudSunTransmission(vec3 worldPosition, vec3 towardSun)
{
    float altitude = ubo.atmosphereCloudMovementParameters.w;
    float separation = ubo.atmosphereCloudLightingParameters.w;
    float verticalDirection = max(towardSun.y, 0.06);
    float lowerDistance = max((altitude - worldPosition.y) / verticalDirection, 0.0);
    float upperDistance = max((altitude + separation - worldPosition.y) /
        verticalDirection, 0.0);
    vec2 lowerPosition = worldPosition.xz + towardSun.xz * lowerDistance;
    vec2 upperPosition = worldPosition.xz + towardSun.xz * upperDistance;
    float lower = cloudLayerDensity(lowerPosition, 1.0, 0.0, vec2(0.0));
    float upper = cloudLayerDensity(upperPosition, 1.63, -0.14, vec2(31.7, -17.2));
    float cloudDensity = 1.0 - (1.0 - lower) * (1.0 - upper * 0.72);
    return exp(-cloudDensity * 4.8);
}

vec3 perturbNormalFromHeight(vec3 worldPosition, vec3 surfaceNormal,
    float detailHeight)
{
    // Screen-space derivatives convert the procedural height into a surface gradient without
    // requiring four additional noise evaluations per fragment.
    vec3 positionDx = dFdx(worldPosition);
    vec3 positionDy = dFdy(worldPosition);
    float heightDx = dFdx(detailHeight);
    float heightDy = dFdy(detailHeight);
    vec3 gradientX = cross(positionDy, surfaceNormal);
    vec3 gradientY = cross(surfaceNormal, positionDx);
    float determinant = dot(positionDx, gradientX);
    vec3 gradient = sign(determinant) *
        (heightDx * gradientX + heightDy * gradientY);
    return normalize(abs(determinant) * surfaceNormal - gradient);
}

float directionalShadowVisibility(vec3 worldPosition, vec3 normal, vec3 lightDirection)
{
    vec4 clip = ubo.directionalShadowViewProjection * vec4(worldPosition, 1.0);
    vec3 projected = clip.xyz / max(clip.w, 0.00001);
    vec2 uv = projected.xy * 0.5 + 0.5;
    if (projected.z <= 0.0 || projected.z >= 1.0 ||
        any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0))))
    {
        return 1.0;
    }
    float bias = max(0.00008, 0.00042 *
        (1.0 - max(dot(normal, lightDirection), 0.0)));
    vec2 texel = 1.0 / vec2(textureSize(directionalShadowMap, 0));
    float visibility = 0.0;
    for (int y = -1; y <= 1; ++y)
    {
        for (int x = -1; x <= 1; ++x)
        {
            visibility += texture(directionalShadowMap,
                vec3(uv + vec2(x, y) * texel, projected.z - bias));
        }
    }
    return visibility / 9.0;
}

float volumetricShadowVisibility(vec3 worldPosition, out bool insideShadowVolume)
{
    vec4 clip = ubo.directionalShadowViewProjection * vec4(worldPosition, 1.0);
    vec3 projected = clip.xyz / max(clip.w, 0.00001);
    vec2 uv = projected.xy * 0.5 + 0.5;
    insideShadowVolume = projected.z > 0.0 && projected.z < 1.0 &&
        all(greaterThanEqual(uv, vec2(0.0))) && all(lessThanEqual(uv, vec2(1.0)));
    if (!insideShadowVolume)
    {
        return 1.0;
    }
    return texture(directionalShadowMap, vec3(uv, projected.z - 0.00018));
}

float atmosphereOpticalDepth(vec3 worldPosition)
{
    vec3 camera = ubo.cameraPositionTime.xyz;
    vec3 segment = worldPosition - camera;
    float distanceToSurface = length(segment);
    if (distanceToSurface <= 0.0001 || ubo.atmosphereFogColorDensity.w <= 0.0)
    {
        return 0.0;
    }

    float baseHeight = ubo.atmosphereFogParameters.x;
    float heightFalloff = ubo.atmosphereFogParameters.y;
    float midpointHeight = (camera.y + worldPosition.y) * 0.5;
    float heightDensity = exp(clamp(
        -(midpointHeight - baseHeight) * heightFalloff, -7.0, 7.0));

    vec2 midpoint = (camera.xz + worldPosition.xz) * 0.5;
    float slowTime = ubo.cameraPositionTime.w * 0.018;
    float bankSignal = 0.5 + 0.25 * (
        sin(midpoint.x * 2.2 - 3.159 + slowTime) +
        sin(midpoint.y * 1.7 + 6.416 - slowTime * 0.72));
    float mistBank = smoothstep(0.52, 0.82, bankSignal);
    // Wide low-density gaps alternate with opaque banks. The demo begins inside a bank so the
    // local effect is immediately legible, while world-space phase keeps streamed chunks joined.
    float bankDensity = mix(0.42, 5.0, mistBank);
    float densityVariation = mix(1.0, bankDensity,
        ubo.atmosphereFogParameters.w);
    return max(ubo.atmosphereFogColorDensity.w * heightDensity * densityVariation *
        distanceToSurface, 0.0);
}

vec3 godRayScattering(vec3 worldPosition, float fogOpacity)
{
    if (ubo.lightingCounts.z == 0U ||
        ubo.atmosphereScatteringParameters.x <= 0.0 || fogOpacity <= 0.001)
    {
        return vec3(0.0);
    }

    vec3 segment = worldPosition - ubo.cameraPositionTime.xyz;
    float fullDistance = length(segment);
    float marchDistance = min(fullDistance, ubo.atmosphereScatteringParameters.y);
    if (marchDistance <= 0.05)
    {
        return vec3(0.0);
    }

    vec3 rayDirection = segment / max(fullDistance, 0.0001);
    float jitter = hash21(gl_FragCoord.xy + vec2(ubo.cameraPositionTime.w * 0.013));
    float visibility = 0.0;
    float coveredSamples = 0.0;
    const int RayMarchSteps = 4;
    for (int step = 0; step < RayMarchSteps; ++step)
    {
        float alongRay = (float(step) + 0.35 + jitter * 0.3) /
            float(RayMarchSteps) * marchDistance;
        bool insideShadowVolume = false;
        float sampleVisibility = volumetricShadowVisibility(
            ubo.cameraPositionTime.xyz + rayDirection * alongRay,
            insideShadowVolume);
        if (insideShadowVolume)
        {
            visibility += sampleVisibility;
            coveredSamples += 1.0;
        }
    }
    if (coveredSamples < 0.5)
    {
        return vec3(0.0);
    }
    visibility /= coveredSamples;

    uint lightIndex = ubo.lightingCounts.z - 1U;
    DirectionalLightData sun = ubo.directionalLights[lightIndex];
    vec3 lightTravelDirection = normalize(sun.directionIntensity.xyz);
    float scatteringCosine = clamp(dot(lightTravelDirection, -rayDirection), -1.0, 1.0);
    const float anisotropy = 0.58;
    float phaseDenominator = 1.0 + anisotropy * anisotropy -
        2.0 * anisotropy * scatteringCosine;
    float phase = (1.0 - anisotropy * anisotropy) /
        (4.0 * Pi * pow(max(phaseDenominator, 0.001), 1.5));
    float shaftVisibility = pow(smoothstep(0.05, 0.88, visibility), 1.35);
    vec3 sunRadiance = sun.colorShadow.rgb * sun.directionIntensity.w;
    float cloudTransmission = cloudSunTransmission(
        ubo.cameraPositionTime.xyz + rayDirection * (marchDistance * 0.5),
        -lightTravelDirection);
    return sunRadiance * phase * shaftVisibility * fogOpacity *
        ubo.atmosphereScatteringParameters.x * cloudTransmission;
}

void main()
{
    MaterialData material = materials[inMaterialIndex];
    vec2 uv = inTexCoord * material.textureTiling.xy;
    float surfaceMarker = abs(inWorldTangent.w);
    bool vegetationSurface = surfaceMarker > 1.5 && surfaceMarker < 2.5;
    bool generatedTerrainSurface = surfaceMarker > 2.5;
    float terrainCoverage = generatedTerrainSurface
        ? clamp(inSurfaceCoverage, 0.0, 1.0) : 0.0;
    float terrainDetailHeight = 0.0;
    float terrainRoughnessTarget = 0.96;
    float terrainPebbleMask = 0.0;

    vec4 baseColor = material.baseColorFactor * vec4(inVertexColor, 1.0);
    if (material.textureIndices.x >= 0 && !generatedTerrainSurface)
    {
        baseColor *= texture(textures[nonuniformEXT(material.textureIndices.x)], uv);
    }
    if (vegetationSurface)
    {
        float centerHighlight = 1.0 - abs(fract(uv.x) * 2.0 - 1.0);
        float rootLight = mix(0.56, 1.0, smoothstep(0.02, 0.78, uv.y));
        float edgeGlow = smoothstep(0.55, 1.0, abs(uv.x * 2.0 - 1.0));
        baseColor.rgb *= rootLight *
            mix(0.92, 1.055, centerHighlight * centerHighlight) *
            mix(1.0, 1.035, edgeGlow);
    }
    else if (generatedTerrainSurface)
    {
        // All layers are sampled in world space, so their scale and phase remain continuous as
        // terrain chunks stream independently. Coverage can come from either the procedural
        // field or a future authored paint map.
        vec2 world = inWorldPosition.xz;
        float coverage = terrainCoverage;
        float dirtWeight = 1.0 - coverage;
        float macroNoise = fractalNoise(world * 1.65 + vec2(13.7, -8.4));
        float middleNoise = fractalNoise(world * 13.0 + vec2(-21.3, 31.8));
        float fineNoise = valueNoise(world * 72.0 + vec2(7.4, 19.1));
        float mossMask = smoothstep(0.34, 0.76,
            macroNoise * 0.72 + middleNoise * 0.28);

        vec3 forestSoil = mix(vec3(0.075, 0.090, 0.035),
            vec3(0.14, 0.135, 0.050), middleNoise);
        vec3 forestMoss = mix(vec3(0.070, 0.205, 0.026),
            vec3(0.145, 0.330, 0.052), fineNoise);
        vec3 forestFloor = mix(forestSoil, forestMoss, mossMask * 0.82);
        float forestTextureHeight = 0.5;
        if (material.textureIndices.x >= 0)
        {
            // One source image covers roughly a third of a terrain world unit. Mirroring it in
            // both axes makes the borders mathematically continuous; broad procedural tinting
            // keeps the repeated capture from forming an obvious checkerboard at a distance.
            vec2 forestTexturePosition = world * 6.25;
            vec2 forestTexturePhase = fract(forestTexturePosition * 0.5) * 2.0;
            vec2 forestTextureUv = 1.0 - abs(forestTexturePhase - 1.0);
            vec3 capturedForest = texture(
                textures[nonuniformEXT(material.textureIndices.x)], forestTextureUv).rgb;
            forestTextureHeight = dot(capturedForest, vec3(0.2126, 0.7152, 0.0722));
            capturedForest = capturedForest * vec3(1.16, 1.20, 1.10) +
                vec3(0.008, 0.006, 0.003);
            capturedForest *= mix(0.86, 1.10,
                macroNoise * 0.72 + middleNoise * 0.28);
            forestFloor = mix(forestFloor, capturedForest, 0.90);
        }

        float compactedNoise = fractalNoise(world * 7.5 + vec2(4.0, 27.0));
        float pathGrain = valueNoise(world * 58.0 + vec2(-15.0, 3.0));
        vec3 packedDirt = mix(vec3(0.245, 0.175, 0.072),
            vec3(0.455, 0.335, 0.135), compactedNoise);
        packedDirt *= mix(0.86, 1.08, pathGrain);
        float pathTextureHeight = 0.5;
        if (material.textureIndices.w >= 0 && dirtWeight > 0.001)
        {
            // The terrain material reserves its occlusion descriptor for a second albedo layer.
            // Mirrored world mapping keeps this capture continuous across both its own borders
            // and independently streamed terrain chunks.
            vec2 pathTexturePosition = world * 7.0 + vec2(0.37, -0.19);
            vec2 pathTexturePhase = fract(pathTexturePosition * 0.5) * 2.0;
            vec2 pathTextureUv = 1.0 - abs(pathTexturePhase - 1.0);
            vec3 capturedPath = texture(
                textures[nonuniformEXT(material.textureIndices.w)], pathTextureUv).rgb;
            pathTextureHeight = dot(capturedPath, vec3(0.2126, 0.7152, 0.0722));
            capturedPath = capturedPath * vec3(1.07, 1.04, 0.96) +
                vec3(0.006, 0.004, 0.002);
            capturedPath *= mix(0.90, 1.08,
                compactedNoise * 0.68 + macroNoise * 0.32);
            packedDirt = mix(packedDirt, capturedPath, 0.92);
        }

        // Sparse embedded stones use one deterministic cell lookup and an antialiased edge.
        vec2 pebblePosition = world * 43.0;
        vec2 pebbleCell = floor(pebblePosition);
        vec2 pebbleLocal = fract(pebblePosition);
        float pebbleRandom = hash21(pebbleCell + vec2(67.0, 11.0));
        vec2 pebbleCenter = vec2(
            hash21(pebbleCell + vec2(3.0, 29.0)),
            hash21(pebbleCell + vec2(47.0, 5.0)));
        pebbleCenter = mix(vec2(0.28), vec2(0.72), pebbleCenter);
        vec2 pebbleOffset = pebbleLocal - pebbleCenter;
        pebbleOffset.x *= mix(0.75, 1.45,
            hash21(pebbleCell + vec2(19.0, 53.0)));
        float pebbleRadius = mix(0.105, 0.235, pebbleRandom);
        float pebbleDistance = length(pebbleOffset);
        float pebbleEdge = max(fwidth(pebbleDistance), 0.015);
        terrainPebbleMask = dirtWeight * step(0.83, pebbleRandom) *
            (1.0 - smoothstep(pebbleRadius, pebbleRadius + pebbleEdge,
                pebbleDistance));
        vec3 pebbleColor = mix(vec3(0.20, 0.185, 0.145),
            vec3(0.43, 0.405, 0.315), pebbleRandom);
        packedDirt = mix(packedDirt, pebbleColor, terrainPebbleMask * 0.88);

        // The semantic transition is deliberately softer than a color-only blend. Moss and
        // darker roots encroach into the feathered edge instead of outlining the path.
        float rootDensity = smoothstep(0.08, 0.92, coverage);
        float pathEdge = 4.0 * coverage * dirtWeight;
        forestFloor = mix(forestFloor, forestFloor * vec3(0.82, 1.02, 0.72),
            pathEdge * 0.20);
        vec3 layeredGround = mix(packedDirt, forestFloor, rootDensity);
        baseColor.rgb = layeredGround;
        baseColor.rgb *= mix(0.96, mix(0.90, 1.02, macroNoise),
            rootDensity * 0.20);

        // A shader-only short-growth layer fills gaps without allocating more instances.
        vec2 microPosition = world * 145.0;
        vec2 cell = floor(microPosition);
        vec2 local = fract(microPosition);
        float randomValue = hash21(cell);
        float lean = (hash21(cell + vec2(17.0, 41.0)) * 2.0 - 1.0) * 0.34;
        float strandCenter = 0.5 + (local.y - 0.48) * lean;
        float strandDistance = abs(local.x - strandCenter);
        float antialiasWidth = max(fwidth(strandDistance), 0.018);
        float strand = 1.0 - smoothstep(0.045, 0.045 + antialiasWidth,
            strandDistance);
        strand *= smoothstep(0.02, 0.20, local.y) *
            (1.0 - smoothstep(0.72, 0.99, local.y));
        float patchNoise = mix(0.76, 1.0,
            hash21(floor(world * 23.0)));
        vec3 undergrowthColor = mix(vec3(0.055, 0.19, 0.022),
            vec3(0.17, 0.42, 0.045), randomValue);
        baseColor.rgb = mix(baseColor.rgb,
            baseColor.rgb * vec3(0.48, 0.63, 0.38), coverage * 0.30 * patchNoise);
        baseColor.rgb = mix(baseColor.rgb, undergrowthColor,
            coverage * strand * 0.62);

        // These amplitudes are in world units. They alter lighting only; terrain collision and
        // silhouettes continue to use the streamed CPU mesh.
        terrainDetailHeight = (macroNoise - 0.5) * 0.00115 +
            (middleNoise - 0.5) * 0.00048 +
            (fineNoise - 0.5) * 0.00016 +
            (forestTextureHeight - 0.32) * coverage * 0.00062 +
            (pathTextureHeight - 0.34) * dirtWeight * 0.00054 +
            terrainPebbleMask * 0.00072 - dirtWeight *
                (compactedNoise - 0.5) * 0.00034;
        terrainRoughnessTarget = mix(
            mix(0.90, 0.95, compactedNoise),
            mix(0.965, 0.99, middleNoise), coverage);
        terrainRoughnessTarget = mix(terrainRoughnessTarget, 0.78,
            terrainPebbleMask * 0.55);
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
    if (generatedTerrainSurface)
    {
        roughness = mix(roughness, terrainRoughnessTarget, 0.84);
    }

    float occlusion = 1.0;
    if (material.textureIndices.w >= 0 && !generatedTerrainSurface)
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
    if (generatedTerrainSurface)
    {
        normal = perturbNormalFromHeight(
            inWorldPosition, normal, terrainDetailHeight);
    }
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
        float rawNDotL = dot(normal, lightDirection);
        float nDotL = max(rawNDotL, 0.0);
        vec3 vegetationLight = vec3(0.0);
        if (vegetationSurface)
        {
            // Thin leaves retain readable form at grazing angles and transmit warm green light
            // when the source lies behind them.
            nDotL = clamp((rawNDotL + 0.32) / 1.32, 0.0, 1.0);
            float backLighting = max(-rawNDotL, 0.0);
            float forwardScatter = pow(max(dot(-lightDirection, viewDirection), 0.0), 3.0);
            vec3 transmissionTint = baseColor.rgb * vec3(0.58, 1.0, 0.30);
            vegetationLight += transmissionTint * backLighting *
                mix(0.16, 0.62, forwardScatter);

            vec3 bladeTangent = normalize(inWorldTangent.xyz);
            float tangentHalfway = clamp(abs(dot(bladeTangent, halfway)), 0.0, 1.0);
            float longitudinalHighlight = pow(
                sqrt(max(1.0 - tangentHalfway * tangentHalfway, 0.0)), 18.0);
            vegetationLight += mix(vec3(0.018), baseColor.rgb, 0.22) *
                longitudinalHighlight * mix(0.32, 1.0, nDotL);
        }
        direct += ((diffuseWeight * baseColor.rgb / Pi + specular) * nDotL +
            vegetationLight) * radiance;
    }

    for (uint lightIndex = 0; lightIndex < ubo.lightingCounts.y; ++lightIndex)
    {
        DirectionalLightData light = ubo.directionalLights[lightIndex];
        vec3 lightDirection = normalize(-light.directionIntensity.xyz);
        vec3 halfway = normalize(viewDirection + lightDirection);
        vec3 fresnel = fresnelSchlick(max(dot(halfway, viewDirection), 0.0), reflectance);
        float distribution = distributionGgx(normal, halfway, roughness);
        float geometry = geometrySmith(normal, viewDirection, lightDirection, roughness);
        vec3 specular = distribution * geometry * fresnel /
            max(4.0 * nDotV * max(dot(normal, lightDirection), 0.0), 0.0001);
        vec3 diffuseWeight = (1.0 - fresnel) * (1.0 - metallic);
        float rawNDotL = dot(normal, lightDirection);
        float nDotL = max(rawNDotL, 0.0);
        vec3 vegetationLight = vec3(0.0);
        if (vegetationSurface)
        {
            nDotL = clamp((rawNDotL + 0.32) / 1.32, 0.0, 1.0);
            float backLighting = max(-rawNDotL, 0.0);
            float forwardScatter = pow(max(dot(-lightDirection, viewDirection), 0.0), 3.0);
            vec3 transmissionTint = baseColor.rgb * vec3(0.58, 1.0, 0.30);
            vegetationLight += transmissionTint * backLighting *
                mix(0.16, 0.62, forwardScatter);
            vec3 bladeTangent = normalize(inWorldTangent.xyz);
            float tangentHalfway = clamp(abs(dot(bladeTangent, halfway)), 0.0, 1.0);
            float longitudinalHighlight = pow(
                sqrt(max(1.0 - tangentHalfway * tangentHalfway, 0.0)), 18.0);
            vegetationLight += mix(vec3(0.018), baseColor.rgb, 0.22) *
                longitudinalHighlight * mix(0.32, 1.0, nDotL);
        }
        float visibility = 1.0;
        if (ubo.lightingCounts.z == lightIndex + 1U)
        {
            visibility = directionalShadowVisibility(
                inWorldPosition, normal, lightDirection);
            // Preserve a little transmitted light through dense vegetation shadows.
            visibility = vegetationSurface ? mix(0.18, 1.0, visibility) : visibility;
        }
        if (lightIndex == 0U)
        {
            float cloudVisibility = cloudSunTransmission(inWorldPosition, lightDirection);
            visibility *= mix(1.0, cloudVisibility,
                ubo.atmosphereCloudLightingParameters.x);
        }
        vec3 radiance = light.colorShadow.rgb * light.directionIntensity.w;
        direct += (((diffuseWeight * baseColor.rgb / Pi + specular) * nDotL +
            vegetationLight) * radiance) * visibility;
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
    if (vegetationSurface)
    {
        vec3 backEnvironment = textureLod(
            irradianceMap, environmentUv(-normal), 0.0).rgb;
        ambient += baseColor.rgb * backEnvironment * vec3(0.045, 0.085, 0.025) *
            ubo.environmentTintIntensity.rgb * ubo.environmentTintIntensity.w;
    }
    vec3 color = ambient + direct + emissive;
    float opticalDepth = atmosphereOpticalDepth(inWorldPosition);
    float fogOpacity = min(1.0 - exp(-opticalDepth),
        ubo.atmosphereFogParameters.z);
    color = mix(color, ubo.atmosphereFogColorDensity.rgb, fogOpacity);
    color += godRayScattering(inWorldPosition, fogOpacity);
    color = color / (color + vec3(1.0));

    outColor = vec4(color, alphaMode == AlphaBlend ? baseColor.a : 1.0);
}
