#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shader_draw_parameters : enable


layout(binding = 0) uniform UniformBufferObject {
	mat4 view;
	mat4 proj;
	float time;
	vec3 cameraPos;
    vec3 lightPos;
    float unused0;
    vec4 unused1;
} ubo;

struct Vertex 
{
    vec3 pos;
    float unused0;
    vec3 normal;
    float unused01;
    vec3 color;
    float unused02;
    vec2 texCoord;
    float unused03;
    float unused04;
    vec3 tangent;
    float unused05;
};

layout(set = 0, binding = 4) readonly buffer Verticies 
{
    Vertex verticies[];
};

struct TransformData
{
    mat4 model;
};

layout(set = 0, binding = 3) readonly buffer Transforms
{
    TransformData transforms[];
};

struct DrawData
{
    int materialIndex; // Index into material buffer
    int transformIndex; // Index into transform buffer
    int vertexOffset; // used to lookup attributes in vertex storage buffer
    int unused; // vec4 padding
};
layout(set = 0, binding = 2) readonly buffer Draws
{
    DrawData drawData[];
};

layout(location = 0) out vec3 vertNormal;
layout(location = 1) out vec2 fragTexCoord;
layout(location = 2) out int matID;
layout(location = 3) out vec3 outLightVec;
layout(location = 4) out vec3 outViewVec;
layout(location = 5) out vec3 outNormal;
layout(location = 6) out vec3 outTangent;

//layout(location = 9) out mat3 TBN;




void main() {
    int drawID = gl_VertexIndex;
    DrawData d = drawData[gl_DrawIDARB];

    Vertex vert = verticies[gl_VertexIndex];
    TransformData t = transforms[d.transformIndex];

    vec4 positionLocal = vec4(vert.pos, 1.0);
    gl_Position =  (ubo.proj * ubo.view * t.model * positionLocal);


    fragTexCoord = vert.texCoord;
    vertNormal = vert.normal;
    matID = d.materialIndex;

    outTangent = ( t.model * vec4(vert.tangent, 1.0)).xyz;;
    outNormal = (t.model * vec4(vert.normal, 1.0)).xyz;
    vec4 pos = t.model * positionLocal;

    outLightVec = ubo.lightPos.xyz - pos.xyz;
    outViewVec = ubo.cameraPos.xyz - pos.xyz;
}