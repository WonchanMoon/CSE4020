from OpenGL.GL import *
from glfw.GLFW import *
import glm
import ctypes
import numpy as np
import os

g_cam_ang = .1
g_cam_ang2 = .1
g_cam_height = 0
g_cam_left_right = 0
g_cam_dis = 1
g_P_sel = 1
xpos_past = 0
ypos_past = 0
H_sel = 0
wire_sel = 0
vertices_drop = []


# now projection matrix P is a global variable so that it can be accessed from main() and framebuffer_size_callback()
g_P = glm.mat4()

g_vertex_shader_src = '''
#version 330 core

layout (location = 0) in vec3 vin_pos; 
layout (location = 1) in vec3 vin_normal; 

out vec3 vout_surface_pos;
out vec3 vout_normal;
out vec4 vout_color;

uniform mat4 MVP;
uniform mat4 M;
uniform vec3 color;

void main()
{
    vec4 p3D_in_hcoord = vec4(vin_pos.xyz, 1.0);
    gl_Position = MVP * p3D_in_hcoord;

    vout_surface_pos = vec3(M * vec4(vin_pos, 1));
    vout_normal = normalize( mat3(inverse(transpose(M)) ) * vin_normal);
    vout_color = vec4(color, 1.);
}
'''
g_vertex_shader_src_color_frame = '''
#version 330 core

layout (location = 0) in vec3 vin_pos; 
layout (location = 1) in vec3 vin_color; 

out vec4 vout_color;

uniform mat4 MVP;

void main()
{
    // 3D points in homogeneous coordinates
    vec4 p3D_in_hcoord = vec4(vin_pos.xyz, 1.0);

    gl_Position = MVP * p3D_in_hcoord;

    vout_color = vec4(vin_color, 1.);
}
'''

g_fragment_shader_src = '''
#version 330 core

in vec3 vout_surface_pos;
in vec3 vout_normal;
in vec4 vout_color;

out vec4 FragColor;

uniform vec3 view_pos;

void main()
{
    // light and material properties
    vec3 light_pos = vec3(20,30,0);
    vec3 light_color = vec3(1,1,1);

    vec3 light_pos2 = vec3(-20,30,0);
    vec3 light_color2 = vec3(1,1,1);

    vec3 light_pos3 = vec3(0,-30,20);
    vec3 light_color3 = vec3(1,1,1);

    vec3 light_pos4 = vec3(0,-30,-20);
    vec3 light_color4 = vec3(1,1,1);
    
    // vec3 material_color = vec3(1,1,1);
    vec3 material_color = vout_color.xyz;
    float material_shininess = 32.0;

    // light components
    vec3 light_ambient = 0.1*light_color;
    vec3 light_diffuse = light_color;
    vec3 light_specular = light_color;

    vec3 light_ambient2 = 0.1*light_color2;
    vec3 light_diffuse2 = light_color2;
    vec3 light_specular2 = light_color2;

    vec3 light_ambient3 = 0.1*light_color3;
    vec3 light_diffuse3 = light_color3;
    vec3 light_specular3 = light_color3;

    vec3 light_ambient4 = 0.1*light_color4;
    vec3 light_diffuse4 = light_color4;
    vec3 light_specular4 = light_color4;

    // material components
    vec3 material_ambient = material_color;
    vec3 material_diffuse = material_color;
    vec3 material_specular = light_color;  // for non-metal material

    vec3 material_specular2 = light_color2;  // for non-metal material

    vec3 material_specular3 = light_color3;  // for non-metal material

    vec3 material_specular4 = light_color4;  // for non-metal material
    
    // ambient
    vec3 ambient = light_ambient * material_ambient;

    vec3 ambient2 = light_ambient2 * material_ambient;

    vec3 ambient3 = light_ambient3 * material_ambient;

    vec3 ambient4 = light_ambient4 * material_ambient;

    // for diffiuse and specular
    vec3 normal = normalize(vout_normal);
    vec3 surface_pos = vout_surface_pos;
    vec3 light_dir = normalize(light_pos - surface_pos);

    vec3 light_dir2 = normalize(light_pos2 - surface_pos);

    vec3 light_dir3 = normalize(light_pos3 - surface_pos);

    vec3 light_dir4 = normalize(light_pos4 - surface_pos);

    // diffuse
    float diff = max(dot(normal, light_dir), 0);
    vec3 diffuse = diff * light_diffuse * material_diffuse;

    float diff2 = max(dot(normal, light_dir2), 0);
    vec3 diffuse2 = diff2 * light_diffuse2 * material_diffuse;

    float diff3 = max(dot(normal, light_dir3), 0);
    vec3 diffuse3 = diff3 * light_diffuse3 * material_diffuse;

    float diff4 = max(dot(normal, light_dir4), 0);
    vec3 diffuse4 = diff4 * light_diffuse4 * material_diffuse;

    // specular
    vec3 view_dir = normalize(view_pos - surface_pos);
    vec3 reflect_dir = reflect(-light_dir, normal);
    float spec = pow( max(dot(view_dir, reflect_dir), 0.0), material_shininess);
    vec3 specular = spec * light_specular * material_specular;

    vec3 reflect_dir2 = reflect(-light_dir2, normal);
    float spec2 = pow( max(dot(view_dir, reflect_dir2), 0.0), material_shininess);
    vec3 specular2 = spec2 * light_specular2 * material_specular;

    vec3 reflect_dir3 = reflect(-light_dir3, normal);
    float spec3 = pow( max(dot(view_dir, reflect_dir3), 0.0), material_shininess);
    vec3 specular3 = spec3 * light_specular3 * material_specular;

    vec3 reflect_dir4 = reflect(-light_dir4, normal);
    float spec4 = pow( max(dot(view_dir, reflect_dir4), 0.0), material_shininess);
    vec3 specular4 = spec4 * light_specular4 * material_specular;

    vec3 color = ambient + diffuse + specular;
    vec3 color2 = ambient2 + diffuse2 + specular2;
    vec3 color3 = ambient3 + diffuse3 + specular3;
    vec3 color4 = ambient4 + diffuse4 + specular4;
    color = color + color2 + color3 + color4;
    FragColor = vec4(color, 1.);
}
'''
g_fragment_shader_src_frame = '''
#version 330 core

in vec4 vout_color;

out vec4 FragColor;

void main()
{
    FragColor = vout_color;
}
'''

class Node:
    def __init__(self, parent, shape_transform, color):
        # hierarchy
        self.parent = parent
        self.children = []
        if parent is not None:
            parent.children.append(self)

        # transform
        self.transform = glm.mat4()
        self.global_transform = glm.mat4()

        # shape
        self.shape_transform = shape_transform
        self.color = color

    def set_transform(self, transform):
        self.transform = transform

    def update_tree_global_transform(self):
        if self.parent is not None:
            self.global_transform = self.parent.get_global_transform() * self.transform
        else:
            self.global_transform = self.transform

        for child in self.children:
            child.update_tree_global_transform()

    def get_global_transform(self):
        return self.global_transform
    def get_shape_transform(self):
        return self.shape_transform
    def get_color(self):
        return self.color


def load_shaders(vertex_shader_source, fragment_shader_source):
    # build and compile our shader program
    # ------------------------------------
    
    # vertex shader 
    vertex_shader = glCreateShader(GL_VERTEX_SHADER)    # create an empty shader object
    glShaderSource(vertex_shader, vertex_shader_source) # provide shader source code
    glCompileShader(vertex_shader)                      # compile the shader object
    
    # check for shader compile errors
    success = glGetShaderiv(vertex_shader, GL_COMPILE_STATUS)
    if (not success):
        infoLog = glGetShaderInfoLog(vertex_shader)
        print("ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" + infoLog.decode())
        
    # fragment shader
    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)    # create an empty shader object
    glShaderSource(fragment_shader, fragment_shader_source) # provide shader source code
    glCompileShader(fragment_shader)                        # compile the shader object
    
    # check for shader compile errors
    success = glGetShaderiv(fragment_shader, GL_COMPILE_STATUS)
    if (not success):
        infoLog = glGetShaderInfoLog(fragment_shader)
        print("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" + infoLog.decode())

    # link shaders
    shader_program = glCreateProgram()               # create an empty program object
    glAttachShader(shader_program, vertex_shader)    # attach the shader objects to the program object
    glAttachShader(shader_program, fragment_shader)
    glLinkProgram(shader_program)                    # link the program object

    # check for linking errors
    success = glGetProgramiv(shader_program, GL_LINK_STATUS)
    if (not success):
        infoLog = glGetProgramInfoLog(shader_program)
        print("ERROR::SHADER::PROGRAM::LINKING_FAILED\n" + infoLog.decode())
        
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    return shader_program    # return the shader program


def key_callback(window, key, scancode, action, mods):
    global g_P_sel, H_sel, wire_sel
    if key==GLFW_KEY_ESCAPE and action==GLFW_PRESS:
        glfwSetWindowShouldClose(window, GLFW_TRUE)
    else:
        if action==GLFW_PRESS or action==GLFW_REPEAT:
            if key==GLFW_KEY_V:
                if g_P_sel:
                    g_P_sel=0
                    print('toggle: orthogonal')
                else:
                    g_P_sel=1
                    print('toggle: perspective')
            if key==GLFW_KEY_H:
                if not H_sel:
                    H_sel=1
                    print('toggle: animating')
                # else:
                #     H_sel=0
                #     print('toggle: single mesh')
            if key==GLFW_KEY_Z:
                if not wire_sel:
                    wire_sel=1
                    print('toggle: wireframe')
                else:
                    wire_sel=0
                    print('toggle: solid')



def cursor_callback(window, xpos, ypos):
    global g_cam_ang, g_cam_ang2, g_cam_height, g_cam_left_right, g_cam_xpan, g_cam_zpan, xpos_past, ypos_past

    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) != GLFW_RELEASE and glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_RELEASE):
        # print('left mouse cursor moving: (%d, %d)'%(xpos, ypos))
        if xpos_past < xpos:
            if np.sign(np.sin(g_cam_ang2)) == 1:
                g_cam_ang -= np.radians(3)
            else:
                g_cam_ang += np.radians(3)
        if xpos_past > xpos:
            if np.sign(np.sin(g_cam_ang2)) == 1:
                g_cam_ang += np.radians(3)
            else:
                g_cam_ang -= np.radians(3)
        if ypos_past < ypos:
            g_cam_ang2 -= np.radians(3)
        if ypos_past > ypos:
            g_cam_ang2 += np.radians(3)
        
    elif(glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) != GLFW_RELEASE and glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE):
        # print('right mouse cursor moving: (%d, %d)'%(xpos, ypos))
        if xpos_past < xpos:
            g_cam_left_right -= .05
        if xpos_past > xpos:
            g_cam_left_right += .05
        if ypos_past < ypos:
            g_cam_height += .05
        if ypos_past > ypos:
                g_cam_height -= .05
                
    xpos_past = xpos
    ypos_past = ypos
    
    # print('mouse cursor moving: (%d, %d)'%(xpos, ypos))

def framebuffer_size_callback(window, width, height):
    global g_P, g_P_sel

    glViewport(0, 0, width, height)

    ortho_height = 10.
    ortho_width = ortho_height * width/height
    if g_P_sel:
        g_P = glm.perspective(20, width/height, .01, 3)
    else:
        g_P = glm.ortho(-ortho_width*.5,ortho_width*.5, -ortho_height*.5,ortho_height*.5, -3,3)


def scroll_callback(window, xoffset, yoffset):
    # print('mouse wheel scroll: %d, %d'%(xoffset, yoffset))
    global g_cam_dis
    
    if yoffset<0:
        g_cam_dis *= 1.1
    elif yoffset>0:
        g_cam_dis /= 1.1
    # print('g_cam_dis: %f'%(g_cam_dis))
    

def drop_callback(window, paths):
    global vertices_drop, H_sel

    if(H_sel):
        H_sel = 0
        print('toggle: single mesh')

    # name = paths[0].split('/')
    name = os.path.split(paths[0])
    name = name[-1]
    # name = name[len(name)-1]
    print('Obj file name:',name)

    f = open(paths[0], 'r')
    v = f.read().split('v ')
    del v[0]
    for i in range(len(v)):
        v[i] = v[i].split('\n')[0]
        if v[i][0] == ' ':
            v[i] = v[i][1::]
    # print('v: ', v)
    # f.close

    f.seek(0)
    vn = f.read().split('vn ')
    del vn[0]
    for i in range(len(vn)):
        vn[i] = vn[i].split('\n')[0]
    # print('vn: ', vn)

    f.seek(0)
    face = f.read().split('f ')
    del face[0]
    for i in range(len(face)):
        face[i] = face[i].split('\n')[0]
        if face[i][-1] == ' ':
            face[i] = face[i][:-1:]
    # print('face: ', face)
    print('Total number of faces:',len(face))
    fw3 = 0
    fw4 = 0
    fwm = 0
    for i in range(len(face)):
        num = face[i].count(' ')
        if num == 2:
            fw3+=1
        elif num == 3:
            fw4+=1
        elif num > 3:
            fwm+=1
    print('Number of faces with 3 vertices:', fw3)
    print('Number of faces with 4 vertices:', fw4)
    print('Number of faces with more than 4 vertices:', fwm)
    vertices_drop = []
    for i in range(len(face)):
        temp = face[i].split(' ')
        for j in range(len(temp)-2):
            for k in [0, j+1, j+2]:
                v_int = list(map(float, v[int(temp[k].split('/')[0])-1].split(' ')))
                vertices_drop.append(v_int[0])
                vertices_drop.append(v_int[1])
                vertices_drop.append(v_int[2])
                vn_int = list(map(float, vn[int(temp[k].split('/')[2])-1].split(' ')))
                vertices_drop.append(vn_int[0])
                vertices_drop.append(vn_int[1])
                vertices_drop.append(vn_int[2])

    # print(vertices)
    # print(len(vertices))
    # print(len(vertices_drop)/6)
    f.close

def obj_to_vertices(paths):

    f = open(paths, 'r')
    v = f.read().split('v ')
    del v[0]
    for i in range(len(v)):
        v[i] = v[i].split('\n')[0]
        if v[i][0] == ' ':
            v[i] = v[i][1::]

    f.seek(0)
    vn = f.read().split('vn ')
    del vn[0]
    for i in range(len(vn)):
        vn[i] = vn[i].split('\n')[0]

    f.seek(0)
    face = f.read().split('f ')
    del face[0]
    for i in range(len(face)):
        face[i] = face[i].split('\n')[0]
        if face[i][-1] == ' ':
            face[i] = face[i][:-1:]

    vertices = []
    for i in range(len(face)):
        temp = face[i].split(' ')
        for j in range(len(temp)-2):
            for k in [0, j+1, j+2]:
                v_int = list(map(float, v[int(temp[k].split('/')[0])-1].split(' ')))
                vertices.append(v_int[0])
                vertices.append(v_int[1])
                vertices.append(v_int[2])
                vn_int = list(map(float, vn[int(temp[k].split('/')[2])-1].split(' ')))
                vertices.append(vn_int[0])
                vertices.append(vn_int[1])
                vertices.append(vn_int[2])
    f.close
    
    return vertices

def prepare_vao_drop():
    global vertices_drop
    vertices_drop = glm.array(np.array(vertices_drop, dtype='float32'))
    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices_drop.nbytes, vertices_drop.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex normals
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_vao_tree():
    vertices = obj_to_vertices('fattree.obj')
    vertices = glm.array(np.array(vertices, dtype='float32'))
    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex normals
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_vao_bird():
    vertices = obj_to_vertices('12213_Bird_v1_l3.obj')
    vertices = glm.array(np.array(vertices, dtype='float32'))
    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex normals
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_vao_dog():
    vertices = obj_to_vertices('10680_Dog_v2.obj')
    vertices = glm.array(np.array(vertices, dtype='float32'))
    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex normals
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_vao_cloud():
    vertices = obj_to_vertices('cloud.obj')
    vertices = glm.array(np.array(vertices, dtype='float32'))
    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex normals
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_vao_heart():
    vertices = obj_to_vertices('12190_Heart_v1_L3.obj')
    vertices = glm.array(np.array(vertices, dtype='float32'))
    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex normals
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_vao_frame():
    # prepare vertex data (in main memory)
    vertices = glm.array(glm.float32,
        # position        # color
         -1.0, 0.0, 0.0,  1.0, 1.0, 1.0, # x-axis start
         1.0, 0.0, 0.0,  1.0, 1.0, 1.0, # x-axis end
         # 0.0, -1.0, 0.0,  0.0, 0.0, 0.0, # y-axis start
         # 0.0, 1.0, 0.0,  0.0, 0.0, 0.0, # y-axis end
         0.0, 0.0, -1.0,  1.0, 1.0, 1.0, # z-axis start
         0.0, 0.0, 1.0,  1.0, 1.0, 1.0, # z-axis end
    )

    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_vao_frame_0():
    # prepare vertex data (in main memory)
    vertices = glm.array(glm.float32,
        # position        # color
         -4.5, 0.0, 0.0,  1.0, 0.0, 0.0, # x-axis start
         4.5, 0.0, 0.0,  1.0, 0.0, 0.0, # x-axis end
         # 0.0, -1.0, 0.0,  0.0, 0.0, 0.0, # y-axis start
         # 0.0, 1.0, 0.0,  0.0, 0.0, 0.0, # y-axis end
         0.0, 0.0, -4.5,  0.0, 0.0, 1.0, # z-axis start
         0.0, 0.0, 4.5,  0.0, 0.0, 1.0, # z-axis end
    )

    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def draw_frame(vao, MVP, MVP_loc):
    glBindVertexArray(vao)
    glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
    glDrawArrays(GL_LINES, 0, 4)
    
def draw_frame_array(vao, MVP, MVP_loc):
    glBindVertexArray(vao)
    n = 15
    for i in range(1,n):
        for k in range(1,n):
            MVP_frame = MVP * glm.translate(glm.vec3(.3*i, 0, .3*k)) * glm.scale(glm.vec3(.3,1,.3))
            glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP_frame))
            glDrawArrays(GL_LINES, 0, 4)
    for i in range(1,n):
        for k in range(1,n):
            MVP_frame = MVP * glm.translate(glm.vec3(.3*i*(-1), 0, .3*k*(-1))) * glm.scale(glm.vec3(.3,1,.3))
            glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP_frame))
            glDrawArrays(GL_LINES, 0, 4)
    for i in range(1,n):
        for k in range(1,n):
            MVP_frame = MVP * glm.translate(glm.vec3(.3*i*(-1), 0, .3*k)) * glm.scale(glm.vec3(.3,1,.3))
            glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP_frame))
            glDrawArrays(GL_LINES, 0, 4)
    for i in range(1,n):
        for k in range(1,n):
            MVP_frame = MVP * glm.translate(glm.vec3(.3*i, 0, .3*k*(-1))) * glm.scale(glm.vec3(.3,1,.3))
            glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP_frame))
            glDrawArrays(GL_LINES, 0, 4)

def draw_node(vao, node, VP, MVP_loc, color_loc):
    MVP = VP * node.get_global_transform() * node.get_shape_transform()
    color = node.get_color()

    glBindVertexArray(vao)
    glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
    glUniform3f(color_loc, color.r, color.g, color.b)
    # glDrawArrays(GL_TRIANGLES, 0, 6) 노드마다 다르니 함수 호출 뒤 따로 실행



def main():
    global vertices_drop, g_P, g_P_sel, H_sel, wire_sel
    # initialize glfw
    if not glfwInit():
        return
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)   # OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3)
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)  # Do not allow legacy OpenGl API calls
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE) # for macOS

    # create a window and OpenGL context
    window = glfwCreateWindow(800, 800, '2019008813', None, None)
    if not window:
        glfwTerminate()
        return
    glfwMakeContextCurrent(window)

    # register event callbacks
    glfwSetKeyCallback(window, key_callback)
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback)
    glfwSetDropCallback(window, drop_callback)
    glfwSetScrollCallback(window, scroll_callback)
    glfwSetCursorPosCallback(window, cursor_callback)

    # load shaders
    shader_program = load_shaders(g_vertex_shader_src, g_fragment_shader_src)
    shader_for_frame = load_shaders(g_vertex_shader_src_color_frame, g_fragment_shader_src_frame)



    # get uniform locations
    MVP_loc_frame = glGetUniformLocation(shader_for_frame, 'MVP')
    MVP_loc = glGetUniformLocation(shader_program, 'MVP')
    M_loc = glGetUniformLocation(shader_program, 'M')
    color_loc = glGetUniformLocation(shader_program, 'color')
    view_pos_loc = glGetUniformLocation(shader_program, 'view_pos')

    # prepare vaos
    vao_frame = prepare_vao_frame()
    vao_frame_0 = prepare_vao_frame_0()
    vao_tree = prepare_vao_tree()
    vao_bird = prepare_vao_bird()
    vao_cloud = prepare_vao_cloud()
    vao_dog = prepare_vao_dog()
    vao_heart = prepare_vao_heart()

    # create a hirarchical model - Node(parent, shape_transform, color)
    base = Node(None, glm.scale((.05,.05,.05)), glm.vec3(0,1,0))
    cloud = Node(base,glm.translate((0,.5,0))*glm.scale((.2,.2,.2)), glm.vec3(1,1,1))
    cloud2 = Node(base,glm.translate((0,.5,-1.5))*glm.scale((.2,.2,.2)), glm.vec3(1,1,1))
    dog = Node(base, glm.translate((.5,0,0))*glm.rotate(np.radians(-90),(1,0,0))*glm.scale((.01,.01,.01)), glm.vec3(1,1,0))
    bird = Node(dog, glm.translate((.5,.5,.5))*glm.rotate(np.radians(-90),(0,1,0))*glm.rotate(np.radians(-90),(1,0,0))*glm.scale((.01,.01,.01)), glm.vec3(1,0.5,0))
    heart = Node(dog, glm.translate((.5,.3,.2))*glm.rotate(np.radians(-90),(0,1,0))*glm.rotate(np.radians(-90),(1,0,0))*glm.scale((.01,.01,.01)), glm.vec3(1,0.1,0.1))

    # initialize projection matrix
    ortho_height = 10.
    ortho_width = ortho_height * 800/800    # initial width/height
    
    pan = glm.vec3(0,0,0) #innitial pan
    zoom = glm.vec3(0,0,0) #innitial zoom
    
    # loop until the user closes the window
    while not glfwWindowShouldClose(window):
        # enable depth test (we'll see details later)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        # projection matrix
        # P = glm.perspective(45, 1, 1, 20)
        P = glm.perspective(20, 1, .01, 10)

        if g_P_sel:
            g_P = glm.perspective(20, 1, .01, 10)
        else:
            #ortho zoom
            g_P = glm.ortho(-ortho_width*g_cam_dis*.1,ortho_width*g_cam_dis*.1, -ortho_height*g_cam_dis*.1,ortho_height*g_cam_dis*.1, -10,10)
        P = g_P

        # render in "wireframe mode"
        if wire_sel:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        
        # view matrix
        # rotate camera position with g_cam_ang, g_cam_ang2
        # move camera up & down along V with g_cam_height, left & right along U with g_cam_left_right
        view_pos = glm.vec3(np.sin(g_cam_ang)*np.sin(g_cam_ang2),np.cos(g_cam_ang2),np.cos(g_cam_ang)*np.sin(g_cam_ang2))+pan+zoom
        V = glm.lookAt(view_pos,
                       glm.vec3(0,0,0)+pan,
                       glm.vec3(0,np.sign(np.sin(g_cam_ang2)),0))

        #left,right,up,down panning(tlanslate camera and target point along U, V)
        pan = glm.vec3(V[0][0]*g_cam_left_right + V[0][1]*g_cam_height,
                       V[1][0]*g_cam_left_right + V[1][1]*g_cam_height,
                       V[2][0]*g_cam_left_right + V[2][1]*g_cam_height)
        
        #zoom (tlanslate camera along W)
        zoom = glm.vec3(V[0][2]*g_cam_dis, V[1][2]*g_cam_dis, V[2][2]*g_cam_dis)

        # animating
        t = glfwGetTime()

        # rotation
        th = np.radians(t*90)
        R = glm.rotate(th, glm.vec3(0,1,0))

        M = glm.mat4()

        # # try applying rotation
        # M = R

        # draw world frame
        glUseProgram(shader_for_frame)
        draw_frame(vao_frame_0, P*V*M, MVP_loc_frame)
        draw_frame_array(vao_frame, P*V*M, MVP_loc_frame)

        # update uniforms
        MVP = P*V*M
        glUseProgram(shader_program)
        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
        glUniformMatrix4fv(M_loc, 1, GL_FALSE, glm.value_ptr(M))
        glUniform3f(view_pos_loc, view_pos.x, view_pos.y, view_pos.z)

        
        #draw drop w.r.t. the current frame MVP
        if(not H_sel):
            vao_drop = prepare_vao_drop()
            glBindVertexArray(vao_drop)
            glUniform3f(color_loc, 1, 1, 1)
            glDrawArrays(GL_TRIANGLES, 0, int(len(vertices_drop)/6))

        #draw animating hierachical model
        else:
            # set local transformations of each node
            base.set_transform(glm.rotate(t*0.1, (0,-1,0)))
            cloud.set_transform(glm.translate((0,(glm.cos(t)+1)*0.3,0)))
            cloud2.set_transform(glm.translate((0,(glm.sin(t)+1)*0.3,0)))
            dog.set_transform(glm.translate((0, 0, glm.sin(t))))
            bird.set_transform(glm.translate((0,(glm.sin(t)+1.5)*0.5,0)))
            heart.set_transform(glm.scale(((glm.cos(t)+1.2)*0.5,(glm.cos(t)+1.2)*0.5,(glm.cos(t)+1.2)*0.5)))


            # recursively update global transformations of all nodes
            base.update_tree_global_transform()

            # draw nodes
            glUseProgram(shader_program)
            # draw_node(vao_tree, base, P*V, MVP_loc, color_loc)
            # glDrawArrays(GL_TRIANGLES, 0, 1500)
            draw_node(vao_tree, base, P*V, MVP_loc, color_loc)
            glDrawArrays(GL_TRIANGLES, 0, 7638)
            draw_node(vao_cloud, cloud, P*V, MVP_loc, color_loc)
            glDrawArrays(GL_TRIANGLES, 0, 416016)
            draw_node(vao_cloud, cloud2, P*V, MVP_loc, color_loc)
            glDrawArrays(GL_TRIANGLES, 0, 416016)
            draw_node(vao_dog, dog, P*V, MVP_loc, color_loc)
            glDrawArrays(GL_TRIANGLES, 0, 298272)
            draw_node(vao_bird, bird, P*V, MVP_loc, color_loc)
            glDrawArrays(GL_TRIANGLES, 0, 384384)
            draw_node(vao_heart, heart, P*V, MVP_loc, color_loc)
            glDrawArrays(GL_TRIANGLES, 0, 33792)

        # swap front and back buffers
        glfwSwapBuffers(window)

        # poll events
        glfwPollEvents()

    # terminate glfw
    glfwTerminate()

if __name__ == "__main__":
    main()
