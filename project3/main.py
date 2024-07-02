from OpenGL.GL import *
from glfw.GLFW import *
import glm
import ctypes
import numpy as np
import os # 파일 path

g_cam_ang = .1
g_cam_ang2 = .1
g_cam_height = 0
g_cam_left_right = 0
g_cam_dis = 1
g_P_sel = 1
xpos_past = 0
ypos_past = 0

drop = 0 # drop?
first = 0 # run animate?
line_sel = 1 # line?
move_sel = 0 # animate?
shape_trans = ['1','1','1']
size_ratio = 1

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
    
    // vec3 material_color = vec3(1,1,1);
    vec3 material_color = vout_color.xyz;
    float material_shininess = 32.0;

    // light components
    vec3 light_ambient = 0.1*light_color;
    vec3 light_diffuse = light_color;
    vec3 light_specular = light_color;

    // material components
    vec3 material_ambient = material_color;
    vec3 material_diffuse = material_color;
    vec3 material_specular = light_color;  // for non-metal material
    
    // ambient
    vec3 ambient = light_ambient * material_ambient;

    // for diffiuse and specular
    vec3 normal = normalize(vout_normal);
    vec3 surface_pos = vout_surface_pos;
    vec3 light_dir = normalize(light_pos - surface_pos);

    // diffuse
    float diff = max(dot(normal, light_dir), 0);
    vec3 diffuse = diff * light_diffuse * material_diffuse;

    // specular
    vec3 view_dir = normalize(view_pos - surface_pos);
    vec3 reflect_dir = reflect(-light_dir, normal);
    float spec = pow( max(dot(view_dir, reflect_dir), 0.0), material_shininess);
    vec3 specular = spec * light_specular * material_specular;

    vec3 color = ambient + diffuse + specular;
    color = color;
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
    def __init__(self, parent, link_transform_from_parent, shape_transform, color):
        # hierarchy
        self.parent = parent
        self.children = []
        if parent is not None:
            parent.children.append(self)

        # transform
        self.link_transform_from_parent = link_transform_from_parent
        self.joint_transform = glm.mat4()
        self.global_transform = glm.mat4()

        # shape
        self.shape_transform = shape_transform
        self.color = color

    def set_joint_transform(self, joint_transform):
        self.joint_transform = joint_transform

    def update_tree_global_transform(self):
        if self.parent is not None:
            self.global_transform = self.parent.get_global_transform() * self.link_transform_from_parent * self.joint_transform
        else:
            self.global_transform = self.link_transform_from_parent * self.joint_transform

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
    global g_P_sel, line_sel, move_sel, size_ratio
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
            if key==GLFW_KEY_1:
                if not line_sel:
                    line_sel=1
                    print('toggle: line')
            if key==GLFW_KEY_2:
                if line_sel:
                    line_sel=0
                    print('toggle: box')
            if key==GLFW_KEY_SPACE:
                if not move_sel:
                    move_sel=1
                    print('toggle: animate')
            if key==GLFW_KEY_3:
                    size_ratio*=0.1
                    print('size: down')
            if key==GLFW_KEY_4:
                    size_ratio*=10
                    print('size: up')




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
    global move_sel, line_sel, node_list, node_channel, node_offset, node_parent, drop, change, fps, first
    
    drop = 1
    first = 1

    if(move_sel):
        move_sel = 0
    if(not line_sel):
        line_sel = 1

    path = paths[0]
    # print(path)

    name = os.path.split(path)
    name = name[-1]
    print('File name:',name)
    f = open(path, 'r')

    hierarchy = f.read().split('MOTION\n')[0].split('\n')
    f.seek(0)
    motion = f.read().split('MOTION\n')[1].split('\n') #각은 degree

    # print(hierarchy)
    # print(motion)

    node_list = []
    node_offset = []
    node_channel = []
    node_parent = ['']
    parent_stack = []

    for i in range(len(hierarchy)): #read line by line
        j = hierarchy[i]
        if 'ROOT ' in j: #ROOT
            node_list.append(j.split('ROOT ')[1])
            parent_stack.append(j.split('ROOT ')[1])
            node_channel.append(hierarchy[i+3].split('CHANNELS ')[1])
            
        elif '}' in j:
            del parent_stack[-1]

        elif 'JOINT ' in j:
            node_list.append(j.split('JOINT ')[1])
            node_channel.append(hierarchy[i+3].split('CHANNELS ')[1])
            node_parent.append(parent_stack[-1])
            parent_stack.append(j.split('JOINT ')[1])

        elif 'End Site' in j:
            node_list.append('End Site')
            node_parent.append(parent_stack[-1])
            parent_stack.append('error')

        elif 'OFFSET ' in j:
            node_offset.append(j.split('OFFSET ')[1])

    joint_list = node_list[:]
    for i in joint_list:
        if 'End Site' in i:
            joint_list.remove(i)

    if('\t' in motion[0].split(' ')[0]):
        Frame_num = int(motion[0].split('\t')[1])
        fps = float(motion[1].split('\t')[1])
    else:
        Frame_num = int(motion[0].split(' ')[1])
        fps = float(motion[1].split(' ')[2])
    change = motion[2:-1]
    
    print('Number of frames:', Frame_num)
    print('FPS:', 1/fps)
    print('Number of joints:',len(joint_list))
    print('List of all joint names:',joint_list)
    # print(motion)
    # print(node_list)
    # print(node_offset)
    # print(node_channel)
    # print(node_parent)
    

    f.close

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

def prepare_vao_line():
    global shape_trans, size_ratio
    # prepare vertex data (in main memory)
    # Node 하나 그릴 때마다 끝 점 바꿔주기
    vertices = glm.array(glm.float32,
        # position        # color
         0.0, 0.0, 0.0,  1.0, 1.0, 1.0, # line start
         float(shape_trans[0])*size_ratio,float(shape_trans[1])*size_ratio,float(shape_trans[2])*size_ratio,  0.0, 1.0, 1.0, # line end
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

def prepare_vao_cube():
    global shape_trans, size_ratio
    # prepare vertex data (in main memory)
    # 8 vertices
    # 직접 바뀐 점에 따라 변환된 기저 구현
    x = float(shape_trans[0])*size_ratio
    y = float(shape_trans[1])*size_ratio
    z = float(shape_trans[2])*size_ratio
    z_vec = glm.vec3(x,y,z) # 새 z축
    y_vec = glm.normalize((z, 0, -x)) * 0.01 # 새 y축
    if(x==0 and z ==0): # y축이 0벡터가 되면 안되기 때문에 
        y_vec = glm.normalize((0,z,-y)) * 0.01
    x_vec = glm.cross(z_vec, y_vec) # x축은 z와 y의 외적
    x_vec = glm.normalize(x_vec) *0.01
    # 점들 계산
    v0 = y_vec + x_vec
    v1 = y_vec - x_vec
    v2 = -1*v0
    v3 = -1*v1
    v4 = z_vec + v0
    v5 = z_vec + v1
    v6 = z_vec + v2
    v7 = z_vec + v3

    vertices = glm.array(glm.float32,
        # position      color
        v0.x ,  v0.y ,  v0.z ,  1, 1, 1, # v0
        v1.x ,  v1.y ,  v1.z ,  1, 1, 1, # v1
        v2.x ,  v2.y ,  v2.z ,  1, 1, 1, # v2
        v3.x ,  v3.y ,  v3.z ,  1, 1, 1, # v3

        v4.x ,  v4.y ,  v4.z  ,  1, 1, 1, # v4
        v5.x ,  v5.y ,  v5.z  ,  1, 1, 1, # v5
        v6.x ,  v6.y ,  v6.z  ,  1, 1, 1, # v6
        v7.x ,  v7.y ,  v7.z  ,  1, 1, 1, # v7
    )

    # prepare index data
    # 12 triangles
    indices = glm.array(glm.uint32,
        0,2,1,
        0,3,2,
        4,5,6,
        4,6,7,
        0,1,5,
        0,5,4,
        3,6,2,
        3,7,6,
        1,2,6,
        1,6,5,
        0,7,3,
        0,4,7,
    )

    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # create and activate EBO (element buffer object)
    EBO = glGenBuffers(1)   # create a buffer object ID and store it to EBO variable
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)  # activate EBO as an element buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # copy index data to EBO
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy index data to the currently bound element buffer

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
    if(color_loc != 0): #직선은 color 값 변경 x.
        glUniform3f(color_loc, color.r, color.g, color.b)
    # glDrawArrays(GL_TRIANGLES, 0, 6) 노드마다 다르니 함수 호출 뒤 따로 실행



def main():
    global g_P, g_P_sel, line_sel, move_sel, node_list, node_channel, node_offset, node_parent, drop, shape_trans, change, fps, first, size_ratio
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
    vao_line = prepare_vao_line()
    vao_cube = prepare_vao_cube()
    
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
            g_P = glm.perspective(20, 1, .01, 100)
        else:
            #ortho zoom
            g_P = glm.ortho(-ortho_width*g_cam_dis*.1,ortho_width*g_cam_dis*.1, -ortho_height*g_cam_dis*.1,ortho_height*g_cam_dis*.1, -10,10)
        P = g_P
        
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

        # list로 각 Node 관리
        Node_array = []
        shape_trans_all = [['0','0','0']]
        if(drop == 1):
            # base
            Node_array.append(Node(None, glm.mat4(),
                                        glm.mat4(), glm.vec3(0,1,1)))
                
            # create a hirarchical model - Node(parent, link_transform_from_parent, shape_transform, color)
            j = 1
            for i in node_parent[1:]:
                parent_idx = node_list.index(i)
                parent = Node_array[parent_idx]
                
                link_trans = node_offset[parent_idx].split(' ')
                shape_trans_all.append(node_offset[j].split(' ')) # shape transform은 box 구현할 때 같이 쓰이기 위해 따로 빼놈.
                Node_array.append(Node(parent, glm.translate((float(link_trans[0])*size_ratio,float(link_trans[1])*size_ratio,float(link_trans[2])*size_ratio)),
                                        glm.mat4(), glm.vec3(0,1,1)))
                j+=1

            Node_array[0].update_tree_global_transform()


        # draw skeleton
        if(move_sel == 0 and drop == 1):
            if(line_sel == 1):
                glUseProgram(shader_for_frame)
                for i in range(len(Node_array)):
                    shape_trans = shape_trans_all[i]
                    vao_line = prepare_vao_line()
                    draw_node(vao_line, Node_array[i], P*V, MVP_loc_frame, 0)
                    glDrawArrays(GL_LINES, 0, 2)
            else:
                glUseProgram(shader_program)
                for i in range(len(Node_array)):
                    shape_trans = shape_trans_all[i]
                    vao_cube = prepare_vao_cube()
                    draw_node(vao_cube, Node_array[i], P*V, MVP_loc, color_loc)
                    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, None)

        # draw animating hierachical model
        elif(move_sel == 1 and drop == 1):
            if(first == 1):
                glfwSetTime(0) # Set t = 0
                first = 0

            temp = change[int(t/fps)%(len(change)-1)].split(' ') # frame time에 맞게 반복되도록 계산
            if '\t' in change[0]:
                temp = change[int(t/fps)%(len(change)-1)].split('\t') # 파일 중에 띄어쓰기를 탭으로 인식하는 오류 수정
            if temp[0] == '': # 파일 중 첫번째가 쓸모없는 값이면 제거
                temp = temp[1:]
            # root를 옮겨줌.
            Node_array[0].link_transform_from_parent = (glm.translate((float(temp[0])*size_ratio,float(temp[1])*size_ratio,float(temp[2])*size_ratio)))

            # recursively update global transformations of all nodes
            Node_array[0].update_tree_global_transform()

            temp = temp[3:]

            l = 0
            for i in range(len(Node_array)):
                shape_trans = shape_trans_all[i]
                
                # End Site 회피
                end = []
                for k in range(len(node_list)):
                    if(node_list[k] == 'End Site'):
                        end.append(k)

                if(i not in end):
                    temp2 = node_channel[l].split(' ')
                    
                    if temp2[0] == '6': # root면 앞 4개는 이미 적용됨
                        temp2 = temp2[4:]
                    else:
                        temp2 = temp2[1:]

                    # euler angles
                    temp3 = []
                    for j in range(3):
                        if temp2[j][0] == 'X':
                            xang = glm.radians(float(temp[l*3+j]))
                            R = glm.rotate(xang, (1,0,0))
                        elif temp2[j][0] == 'Y':
                            yang = glm.radians(float(temp[l*3+j]))
                            R = glm.rotate(yang, (0,1,0))
                        else:
                            zang = glm.radians(float(temp[l*3+j]))
                            R = glm.rotate(zang, (0,0,1))
                        temp3.append(R)
                        
                    # set local transformations of each node
                    Node_array[i].set_joint_transform(temp3[0]*temp3[1]*temp3[2])
                    l+=1

                # recursively update global transformations of all nodes
                Node_array[i].update_tree_global_transform()
                if(line_sel == 1):
                    glUseProgram(shader_for_frame)
                    vao_line = prepare_vao_line()
                    draw_node(vao_line, Node_array[i], P*V, MVP_loc_frame, 0)
                    glDrawArrays(GL_LINES, 0, 2)
                else:
                    glUseProgram(shader_program)
                    vao_cube = prepare_vao_cube()
                    draw_node(vao_cube, Node_array[i], P*V, MVP_loc, color_loc)
                    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, None)
            

        # swap front and back buffers
        glfwSwapBuffers(window)

        # poll events
        glfwPollEvents()

    # terminate glfw
    glfwTerminate()

if __name__ == "__main__":
    main()
