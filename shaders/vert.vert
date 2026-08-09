#version 460
#extension GL_ARB_shader_draw_parameters : require

layout(set = 0, binding = 0) uniform UniformBufferObject
{
    mat4 view;
    mat4 projection;
    vec4 cameraPositionTime;
    vec4 lightPosition;
} ubo;

struct Vertex
{
    vec3 position;
    float unused0;
    vec3 normal;
    float unused1;
    vec3 color;
    float unused2;
    vec2 texCoord;
    float unused3;
    float unused4;
    vec3 tangent;
    float tangentSign;
    uvec4 joints;
    vec4 weights;
};

layout(set = 0, binding = 4) readonly buffer Vertices
{
    Vertex vertices[];
};

struct TransformData
{
    mat4 model;
};

layout(set = 0, binding = 3) readonly buffer Transforms
{
    TransformData transforms[];
};

layout(set = 0, binding = 6) readonly buffer JointMatrices
{
    mat4 jointMatrices[];
};

struct DrawData
{
    int materialIndex;
    int transformIndex;
    int vertexOffset;
    int jointOffset;
};

layout(set = 0, binding = 2) readonly buffer Draws
{
    DrawData draws[];
};

layout(location = 0) out vec2 outTexCoord;
layout(location = 1) flat out int outMaterialIndex;
layout(location = 2) out vec3 outWorldNormal;
layout(location = 3) out vec4 outWorldTangent;
layout(location = 4) out vec3 outWorldPosition;

void main()
{
    DrawData draw = draws[gl_BaseInstance];
    Vertex vertex = vertices[gl_VertexIndex];
    mat4 model = transforms[draw.transformIndex].model;
    vec3 position = vertex.position;
    vec3 normal = vertex.normal;
    vec3 tangent = vertex.tangent;
    if (draw.jointOffset >= 0)
    {
        mat4 skin = vertex.weights.x * jointMatrices[draw.jointOffset + int(vertex.joints.x)] +
            vertex.weights.y * jointMatrices[draw.jointOffset + int(vertex.joints.y)] +
            vertex.weights.z * jointMatrices[draw.jointOffset + int(vertex.joints.z)] +
            vertex.weights.w * jointMatrices[draw.jointOffset + int(vertex.joints.w)];
        position = (skin * vec4(position, 1.0)).xyz;
        normal = mat3(skin) * normal;
        tangent = mat3(skin) * tangent;
    }
    vec4 worldPosition = model * vec4(position, 1.0);
    mat3 normalMatrix = transpose(inverse(mat3(model)));

    gl_Position = ubo.projection * ubo.view * worldPosition;
    outTexCoord = vertex.texCoord;
    outMaterialIndex = draw.materialIndex;
    outWorldNormal = normalize(normalMatrix * normal);
    outWorldTangent = vec4(normalize(mat3(model) * tangent),
        vertex.tangentSign * sign(determinant(mat3(model))));
    outWorldPosition = worldPosition.xyz;
}
