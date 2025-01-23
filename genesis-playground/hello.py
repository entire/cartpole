import genesis as gs
import threading
import time

def main():
    ########################## init ##########################
    gs.init(backend=gs.cpu)

    ########################## create a scene ##########################
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(),
        viewer_options=gs.options.ViewerOptions(
            res=(800, 600),          # Explicit resolution
            refresh_rate=60,         # Monitor refresh rate
            max_FPS=60,             # Cap FPS
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_up=(0.0, 0.0, 1.0),
            camera_fov=40,
        ),
        show_viewer=True,
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
            gravity=(0.0, 0.0, -10.0),
        ),
    )

    ########################## entities ##########################
    plane = scene.add_entity(gs.morphs.Plane())
    r0 = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )

    ########################## build ##########################
    scene.build()

    # Run simulation in another thread using Genesis helper function
    def run_sim(scene):
        while True:
            scene.step()
            time.sleep(1/240)  # Cap at 240 Hz

    gs.tools.run_in_another_thread(fn=run_sim, args=(scene,))
    
    # Start viewer in main thread
    scene.viewer.start()

if __name__ == "__main__":
    main()