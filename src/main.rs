use glfw::{Action, Context, Key};
use nalgebra_glm as glm;
use std::ffi::CString;
use std::fs;
use std::io::{self, Write};
use std::sync::{Arc, Mutex};

mod camera;
mod physics;
mod render;
use camera::Camera;
use render::{generate_sphere, ShaderProgram, VertexArray};

// Helper function to get user input from the terminal
fn get_quantum_number(prompt: &str, default: i32) -> i32 {
    loop {
        print!("{} (default: {}): ", prompt, default);
        io::stdout().flush().unwrap();
        let mut input = String::new();
        io::stdin().read_line(&mut input).expect("Failed to read line");
        
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return default;
        }
        
        match trimmed.parse::<i32>() {
            Ok(num) => return num,
            Err(_) => println!("Invalid input. Please enter an integer or press Enter for default."),
        }
    }
}

fn main() {
    // --- Get Quantum Numbers from User ---
    println!("Enter initial quantum numbers for the simulation.");
    let n = get_quantum_number("Principal quantum number (n)", 2);
    let l = get_quantum_number("Azimuthal quantum number (l)", 1);
    let m = get_quantum_number("Magnetic quantum number (m)", 0);
    
    // --- Set initial physics state ---
    *physics::N.lock().unwrap() = n;
    *physics::L.lock().unwrap() = l;
    *physics::M.lock().unwrap() = m;

    // --- Standard Setup ---
    let mut glfw = glfw::init(glfw::fail_on_errors).unwrap();
    let (win_width, win_height) = (1280, 720);
    let (mut window, events) = glfw
        .create_window(win_width as u32, win_height as u32, "Atom Simulator", glfw::WindowMode::Windowed)
        .expect("Failed to create GLFW window.");
    window.make_current();
    window.set_key_polling(true);
    window.set_cursor_pos_polling(true);
    window.set_mouse_button_polling(true);
    window.set_scroll_polling(true);
    gl::load_with(|symbol| window.get_proc_address(symbol) as *const _);

    // --- Create rendering objects ---
    let shader_program = unsafe {
        let vs_src = CString::new(fs::read_to_string("src/vertex_shader.glsl").unwrap().as_bytes()).unwrap();
        let fs_src = CString::new(fs::read_to_string("src/fragment_shader.glsl").unwrap().as_bytes()).unwrap();
        ShaderProgram::new(&vs_src, &fs_src)
    };
    let sphere_vertices = generate_sphere(1.0, 10, 10);
    let sphere = unsafe { VertexArray::new(&sphere_vertices) };

    // --- Create Particles ---
    println!("\nGenerating particle set for n={}, l={}, m={}...", n, l, m);
    let particles = physics::generate_particles(100000);
    println!("Done.");

    // --- Create Camera ---
    let camera = Arc::new(Mutex::new(Camera::new(glm::vec3(0.0, 0.0, 0.0), 30.0)));

    // --- Create CStrings for uniform names ---
    let color_name = CString::new("ourColor").unwrap();
    let model_name = CString::new("model").unwrap();
    let view_name = CString::new("view").unwrap();
    let proj_name = CString::new("projection").unwrap();

    // Enable Depth Test
    unsafe { gl::Enable(gl::DEPTH_TEST); }

    while !window.should_close() {
        // --- Event Handling ---
        glfw.poll_events();
        for (_, event) in glfw::flush_messages(&events) {
            handle_window_event(&mut window, &event, &camera);
        }

        // --- Rendering ---
        unsafe {
            gl::ClearColor(0.3, 0.3, 0.3, 1.0);
            gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
            shader_program.use_program();

            let view = camera.lock().unwrap().get_view_matrix();
            let projection = glm::perspective(
                (win_width as f32) / (win_height as f32),
                glm::radians(&glm::vec1(45.0))[0],
                0.1,
                100.0,
            );
            shader_program.set_uniform_mat4(&view_name, &view);
            shader_program.set_uniform_mat4(&proj_name, &projection);

            sphere.bind();
            for particle in &particles {
                let mut model = glm::identity();
                let pos_f32 = glm::vec3(particle.position.x as f32, particle.position.y as f32, particle.position.z as f32);
                model = glm::translate(&model, &pos_f32);
                model = glm::scale(&model, &glm::vec3(0.05, 0.05, 0.05));
                shader_program.set_uniform_mat4(&model_name, &model);
                shader_program.set_uniform_4f(&color_name, particle.color.x, particle.color.y, particle.color.z, particle.color.w);
                gl::DrawArrays(gl::TRIANGLES, 0, sphere.vertex_count());
            }
        }
        window.swap_buffers();
    }
}

fn handle_window_event(window: &mut glfw::Window, event: &glfw::WindowEvent, camera: &Arc<Mutex<Camera>>) {
    match event {
        glfw::WindowEvent::Key(Key::Escape, _, Action::Press, _) => {
            window.set_should_close(true);
        }
        glfw::WindowEvent::CursorPos(x, y) => {
            camera.lock().unwrap().process_mouse_move(*x, *y);
        }
        glfw::WindowEvent::MouseButton(button, action, _) => {
            let pos = window.get_cursor_pos();
            camera.lock().unwrap().process_mouse_button(*button, *action, pos.0, pos.1);
        }
        glfw::WindowEvent::Scroll(_, y_offset) => {
            camera.lock().unwrap().process_scroll(*y_offset);
        }
        _ => {}
    }
}
