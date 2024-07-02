from OpenGL.GL import *
from glfw.GLFW import *
import glm
import ctypes
import numpy as np

g_cam_ang = .1
g_cam_ang2 = .1
g_cam_height = 0
g_cam_left_right = 0
g_cam_dis = 1
g_P_sel = 1
xpos_past = 0
ypos_past = 0

# now projection matrix P is a global variable so that it can be accessed from main() and framebuffer_size_callback()
g_P = glm.mat4()


g_vertex_shader_src = '''
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

in vec4 vout_color;

out vec4 FragColor;

void main()
{
    FragColor = vout_color;
}
'''

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
    global g_P_sel
    if key==GLFW_KEY_ESCAPE and action==GLFW_PRESS:
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    else:
        if action==GLFW_PRESS or action==GLFW_REPEAT:
            if key==GLFW_KEY_V:
                if g_P_sel:
                    g_P_sel=0
                    print('toggle: orthogonal')
                else:
                    g_P_sel=1
                    print('toggle: perspective')

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

        
    
def scroll_callback(window, xoffset, yoffset):
    # print('mouse wheel scroll: %d, %d'%(xoffset, yoffset))
    global g_cam_dis
    
    if yoffset<0:
        g_cam_dis *= 1.1
    elif yoffset>0:
        g_cam_dis /= 1.1
    # print('g_cam_dis: %f'%(g_cam_dis))
    
def framebuffer_size_callback(window, width, height):
    global g_P, g_P_sel

    glViewport(0, 0, width, height)

    ortho_height = 10.
    ortho_width = ortho_height * width/height
    if g_P_sel:
        g_P = glm.perspective(20, width/height, .01, 3)
    else:
        g_P = glm.ortho(-ortho_width*.5,ortho_width*.5, -ortho_height*.5,ortho_height*.5, -3,3)
        

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

def main():
    global g_P, g_P_sel
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
    glfwSetScrollCallback(window, scroll_callback)
    glfwSetCursorPosCallback(window, cursor_callback)

    # load shaders
    shader_program = load_shaders(g_vertex_shader_src, g_fragment_shader_src)

    # get uniform locations
    MVP_loc = glGetUniformLocation(shader_program, 'MVP')
    
    # prepare vaos
    vao_frame = prepare_vao_frame()
    vao_frame_0 = prepare_vao_frame_0()
 
    # # viewport
    # glViewport(100,100, 200,200)

    # initialize projection matrix
    ortho_height = 10.
    ortho_width = ortho_height * 800/800    # initial width/height
    
    pan = glm.vec3(0,0,0) #innitial pan
    zoom = glm.vec3(0,0,0) #innitial zoom
    
    # loop until the user closes the window
    while not glfwWindowShouldClose(window):
        
        if g_P_sel:
            g_P = glm.perspective(20, 1, .01, 10)
        else:
            #ortho zoom
            g_P = glm.ortho(-ortho_width*g_cam_dis*.1,ortho_width*g_cam_dis*.1, -ortho_height*g_cam_dis*.1,ortho_height*g_cam_dis*.1, -10,10)
        
        # enable depth test (we'll see details later)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        # render in "wireframe mode"
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        glUseProgram(shader_program)
        
        # view matrix
        # rotate camera position with g_cam_ang, g_cam_ang2
        # move camera up & down along V with g_cam_height, left & right along U with g_cam_left_right
        V = glm.lookAt(glm.vec3(np.sin(g_cam_ang)*np.sin(g_cam_ang2),np.cos(g_cam_ang2),np.cos(g_cam_ang)*np.sin(g_cam_ang2))+pan+zoom,
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
        # th = np.radians(t*90)
        # R = glm.rotate(th, glm.vec3(1,0,0))
        # T = glm.translate()
        M = glm.mat4()
        # # try applying rotation
        # M = R
        # M = T

        # draw world frame
        draw_frame(vao_frame_0, g_P*V*M, MVP_loc)

        # draw world frame array
        draw_frame_array(vao_frame, g_P*V*M, MVP_loc)

        # swap front and back buffers
        glfwSwapBuffers(window)

        # poll events
        glfwPollEvents()

    # terminate glfw
    glfwTerminate()

if __name__ == "__main__":
    main()
