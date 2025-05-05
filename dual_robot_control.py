import numpy as np
import mujoco
import time
import os
from mujoco.glfw import glfw
import cv2
import traceback

print(f"Current working directory: {os.getcwd()}")

# OpenCV 테스트
print("Testing OpenCV...")
try:
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.imshow('Test', test_img)
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    print("OpenCV is working!")
except Exception as e:
    print(f"OpenCV test failed: {e}")

SIM_TASK_CONFIGS = {
    'camera_names': ['left_wrist_cam', 'right_wrist_cam']
}

try:
    # Load the model
    print("Loading model...")
    model = mujoco.MjModel.from_xml_path('/home/songjisu/Downloads/CookingBot/Mujoco/dual_arm_robot.xml')
    data = mujoco.MjData(model)
    print("Model loaded successfully!")
    
    # Initialize GLFW
    print("Initializing GLFW...")
    if not glfw.init():
        raise Exception("Failed to initialize GLFW")
    
    window = glfw.create_window(1200, 900, "Dual Robot Box Lifting", None, None)
    if not window:
        glfw.terminate()
        raise Exception("Failed to create GLFW window")
    
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    
    # Set up MuJoCo visualization
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

    # Find left_wrist_cam
    left_wrist_cam_id = None
    for i in range(model.ncam):
        if model.cam(i).name == "left_wrist_cam":
            left_wrist_cam_id = i
            print(f"Found left_wrist_cam with ID: {left_wrist_cam_id}")
            break
    
    if left_wrist_cam_id is None:
        print("Warning: left_wrist_cam not found in model")
        print("Available cameras:")
        for i in range(model.ncam):
            print(f"  {i}: {model.cam(i).name}")

    # Camera settings
    cam.distance = 1.2
    cam.elevation = -20.0
    cam.azimuth = 0.0
    cam.lookat = np.array([0.0, 0.0, 0.3])

    # Mouse interaction variables
    button_left = False
    button_middle = False
    button_right = False
    lastx = 0
    lasty = 0

    # Keyboard callback
    def keyboard(window, key, scancode, act, mods):
        if act == glfw.PRESS:
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, glfw.GLFW_TRUE)
                print("ESC pressed - closing application")
            elif key == glfw.KEY_R:
                reset_position()
                print("Reset to initial position")
            elif key == glfw.KEY_SPACE:
                box_lifting_task()
                print("Starting box lifting task")
            elif key == glfw.KEY_H:
                ready_position()
                print("Moving to ready position")
            elif key == glfw.KEY_Q:
                glfw.set_window_should_close(window, glfw.GLFW_TRUE)
                print("Q pressed - closing both windows")

    def mouse_button(window, button, act, mods):
        global button_left, button_middle, button_right
        if button == glfw.MOUSE_BUTTON_LEFT:
            button_left = (act == glfw.PRESS)
        elif button == glfw.MOUSE_BUTTON_MIDDLE:
            button_middle = (act == glfw.PRESS)
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            button_right = (act == glfw.PRESS)
        global lastx, lasty
        lastx, lasty = glfw.get_cursor_pos(window)

    def mouse_move(window, xpos, ypos):
        global lastx, lasty, button_left, button_middle, button_right
        if button_left:
            dy = 0.01 * (ypos - lasty)
            dx = 0.01 * (xpos - lastx)
            cam.elevation = np.clip(cam.elevation - dy*100, -90, 90)
            cam.azimuth = (cam.azimuth + dx*100) % 360
        elif button_middle:
            dx = 0.001 * (xpos - lastx)
            dy = 0.001 * (ypos - lasty)
            cam.lookat[0] += -dx*cam.distance
            cam.lookat[1] += dy*cam.distance
        elif button_right:
            dy = 0.01 * (ypos - lasty)
            cam.distance = np.clip(cam.distance + dy, 0.1, 5.0)
        lastx = xpos
        lasty = ypos

    def scroll(window, xoffset, yoffset):
        cam.distance = np.clip(cam.distance - 0.1 * yoffset, 0.1, 5.0)

    # Register callbacks
    glfw.set_key_callback(window, keyboard)
    glfw.set_mouse_button_callback(window, mouse_button)
    glfw.set_cursor_pos_callback(window, mouse_move)
    glfw.set_scroll_callback(window, scroll)

    # Create offscreen renderer for camera
    camera_width = 320
    camera_height = 240
    camera_context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
    
    def capture_camera():
        if left_wrist_cam_id is not None:
            # Create viewport for camera
            viewport = mujoco.MjrRect(0, 0, camera_width, camera_height)
            
            # Create image buffers
            rgb = np.zeros((camera_height, camera_width, 3), dtype=np.uint8)
            depth = np.zeros((camera_height, camera_width), dtype=np.float32)
            
            # Update scene with fixed camera
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
            camera.fixedcamid = left_wrist_cam_id
            
            # Render from camera
            mujoco.mjv_updateScene(model, data, opt, None, camera, mujoco.mjtCatBit.mjCAT_ALL.value, scene)
            mujoco.mjr_render(viewport, scene, camera_context)
            
            # Read pixels
            mujoco.mjr_readPixels(rgb, depth, viewport, camera_context)
            
            # Flip image (OpenGL to OpenCV) and rotate 90 degrees to fix orientation
            rgb = np.flipud(rgb)
            rgb = np.rot90(rgb, k=3)  # Rotate 270 degrees counterclockwise
            
            return rgb
        return None

    # Render function
    def render_scene():
        viewport_width, viewport_height = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)
        
        # Update and render main scene
        mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL.value, scene)
        mujoco.mjr_render(viewport, scene, context)
        
        # Overlay text
        text = [
            "ESC/Q: Exit both windows",
            "SPACE: Start task",
            "R: Reset",
            "H: Ready position",
            "Left wrist camera is displaying in separate window"
        ]
        overlay = "\n".join(text)
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, viewport, overlay, "", context)
        
        glfw.swap_buffers(window)
        return True

    # Get joints
    left_joints = [
        model.actuator('omx_left/joint1').id,
        model.actuator('omx_left/joint2').id,
        model.actuator('omx_left/joint3').id,
        model.actuator('omx_left/joint4').id,
        model.actuator('omx_left/gripper_joint').id
    ]

    right_joints = [
        model.actuator('omx_right/joint1').id,
        model.actuator('omx_right/joint2').id,
        model.actuator('omx_right/joint3').id,
        model.actuator('omx_right/joint4').id,
        model.actuator('omx_right/gripper_joint').id
    ]

    # Robot control functions
    def reset_position():
        data.ctrl[left_joints[0]] = 0.0
        data.ctrl[left_joints[1]] = -0.3
        data.ctrl[left_joints[2]] = 0.6
        data.ctrl[left_joints[3]] = 0.0
        data.ctrl[left_joints[4]] = 0.01
        
        data.ctrl[right_joints[0]] = 0.0
        data.ctrl[right_joints[1]] = -0.3
        data.ctrl[right_joints[2]] = 0.6
        data.ctrl[right_joints[3]] = 0.0
        data.ctrl[right_joints[4]] = 0.01

    def ready_position():
        data.ctrl[left_joints[0]] = 0.0
        data.ctrl[left_joints[1]] = -0.4
        data.ctrl[left_joints[2]] = 0.8
        data.ctrl[left_joints[3]] = 0.2
        data.ctrl[left_joints[4]] = 0.015
        
        data.ctrl[right_joints[0]] = 0.0
        data.ctrl[right_joints[1]] = -0.4
        data.ctrl[right_joints[2]] = 0.8
        data.ctrl[right_joints[3]] = 0.2
        data.ctrl[right_joints[4]] = 0.015

    def box_lifting_task():
        print("Starting box lifting task...")
        
        # 1. Move to box
        data.ctrl[left_joints[0]] = 0.0
        data.ctrl[left_joints[1]] = -0.6
        data.ctrl[left_joints[2]] = 1.0
        data.ctrl[left_joints[3]] = 0.3
        data.ctrl[left_joints[4]] = 0.015
        
        data.ctrl[right_joints[0]] = 0.0
        data.ctrl[right_joints[1]] = -0.6
        data.ctrl[right_joints[2]] = 1.0
        data.ctrl[right_joints[3]] = 0.3
        data.ctrl[right_joints[4]] = 0.015
        
        simulate(1.5)
        
        # 2. Close grippers
        data.ctrl[left_joints[4]] = -0.005
        data.ctrl[right_joints[4]] = -0.005
        simulate(1.0)
        
        # 3. Lift box
        for i in range(15):
            data.ctrl[left_joints[1]] = -0.6 + i * 0.04
            data.ctrl[right_joints[1]] = -0.6 + i * 0.04
            simulate(0.1)
        
        simulate(1.5)
        
        # 4. Pass box
        data.ctrl[left_joints[4]] = 0.015
        simulate(1.0)
        
        # 5. Left arm moves away
        data.ctrl[left_joints[2]] = 0.5
        data.ctrl[left_joints[1]] = -0.3
        simulate(1.0)
        
        # 6. Lower box
        for i in range(15):
            data.ctrl[right_joints[1]] = -0.0 - i * 0.04
            simulate(0.1)
        
        # 7. Release box
        data.ctrl[right_joints[4]] = 0.015
        simulate(1.0)
        
        # 8. Return to original
        reset_position()
        simulate(2.0)
        
        print("Box lifting task completed")

    def simulate(duration):
        start_time = time.time()
        while time.time() - start_time < duration:
            mujoco.mj_step(model, data)
            render_scene()
            glfw.poll_events()
            if glfw.window_should_close(window):
                return False
            time.sleep(0.01)
        return True

    # Create OpenCV window
    cv2.namedWindow('Left Wrist Camera', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Left Wrist Camera', 640, 480)
    cv2.moveWindow('Left Wrist Camera', 1250, 100)  # Position to the right of GLFW window
    print("OpenCV window created")

    # Initialize robots
    print("Initializing robots...")
    reset_position()
    
    # Display instructions
    print("\n=== Dual Robot Box Lifting Control ===")
    print("Two windows will be displayed:")
    print("1. Main simulation window (left)")
    print("2. Left wrist camera view (right)")
    print("\nControls:")
    print("ESC/Q: Exit both windows")
    print("SPACE: Start box lifting task")
    print("R: Reset position")
    print("H: Ready position")
    print("Mouse: Drag to rotate, middle-click to pan, right-click to zoom")
    
    # Force first render to ensure window is shown
    render_scene()
    glfw.poll_events()
    
    # Main loop
    while not glfw.window_should_close(window):
        # Advance simulation
        mujoco.mj_step(model, data)
        
        # Render main scene
        render_scene()
        
        # Capture and display camera
        camera_img = capture_camera()
        if camera_img is not None:
            cv2.imshow('Left Wrist Camera', camera_img)
            key = cv2.waitKey(1)
            
            if key == ord('q') or key == 27:  # q or ESC
                break
        
        # Process events
        glfw.poll_events()
        
        time.sleep(0.01)

except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()
finally:
    # Cleanup
    print("Cleaning up...")
    glfw.terminate()
    cv2.destroyAllWindows()
    print("Cleanup complete.")