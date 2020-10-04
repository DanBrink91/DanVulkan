#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shader_draw_parameters : enable


layout(binding = 0) uniform UniformBufferObject {
	mat4 view;
	mat4 proj;
	float time;
	vec3 cameraPos;
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

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexCoord;
layout(location = 2) out int matID;
layout(location = 3) out int drawID;
layout(location = 4) out DrawData d;


void main() {
    drawID = gl_VertexIndex;
    d = drawData[gl_DrawIDARB];

    Vertex vert = verticies[gl_VertexIndex + d.vertexOffset - 6];
    TransformData t = transforms[0];

    vec4 positionLocal = vec4(vert.pos, 1.0);
    // ubo.proj * ubo.view * t.model *
    gl_Position =  positionLocal;
    
    fragTexCoord = vert.texCoord;
    fragColor = vert.color;
    matID = d.materialIndex;
}