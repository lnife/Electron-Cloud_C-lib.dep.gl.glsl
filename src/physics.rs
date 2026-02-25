use nalgebra_glm as glm;
use rand::Rng;
use statrs::function::gamma;
use lazy_static::lazy_static;
use std::sync::Mutex;
use std::f64::consts::PI;

// --- Data Structures and Constants ---

pub struct Particle {
    pub position: glm::DVec3,
    pub color: glm::Vec4,
}

const A0: f64 = 1.0;

lazy_static! {
    pub static ref N: Mutex<i32> = Mutex::new(2);
    pub static ref L: Mutex<i32> = Mutex::new(1);
    pub static ref M: Mutex<i32> = Mutex::new(0);
}

// --- Particle Generation ---

pub fn generate_particles(num_particles: usize) -> Vec<Particle> {
    let n = *N.lock().unwrap();
    let l = *L.lock().unwrap();
    let m = *M.lock().unwrap();
    let mut particles = Vec::with_capacity(num_particles);

    for _ in 0..num_particles {
        let r = sample_r(n, l);
        let theta = sample_theta(l, m);
        let phi = sample_phi();
        let pos = spherical_to_cartesian(r, theta, phi);
        let color = get_particle_color(r, theta, phi, n, l, m);
        particles.push(Particle { position: pos, color });
    }
    particles
}

pub fn spherical_to_cartesian(r: f64, theta: f64, phi: f64) -> glm::DVec3 {
    let x = r * theta.sin() * phi.cos();
    let y = r * theta.cos();
    let z = r * theta.sin() * phi.sin();
    glm::vec3(x, y, z)
}

// --- Physics Calculations & Sampling ---

fn sample_phi() -> f64 {
    let mut rng = rand::thread_rng();
    rng.gen_range(0.0..2.0 * PI)
}

fn sample_r(n: i32, l: i32) -> f64 {
    lazy_static! { static ref CDF_CACHE: Mutex<Vec<(i32, i32, Vec<f64>)>> = Mutex::new(Vec::new()); }
    let mut cache = CDF_CACHE.lock().unwrap();
    if let Some(entry) = cache.iter().find(|(cn, cl, _)| *cn == n && *cl == l) {
        let cdf = &entry.2;
        let u: f64 = rand::thread_rng().gen();
        let idx = match cdf.binary_search_by(|v| v.partial_cmp(&u).unwrap()) { Ok(i) => i, Err(i) => i };
        let r_max = 10.0 * (n * n) as f64 * A0;
        return idx as f64 * (r_max / (cdf.len() - 1) as f64);
    }
    const N_CDF: usize = 4096;
    let r_max = 10.0 * (n * n) as f64 * A0;
    let mut cdf = vec![0.0; N_CDF];
    let dr = r_max / (N_CDF - 1) as f64;
    let mut sum = 0.0;
    for i in 0..N_CDF {
        let r = i as f64 * dr;
        let rho = 2.0 * r / (n as f64 * A0);
        let laguerre_l = associated_laguerre(n - l - 1, 2 * l + 1, rho);
        let norm_part1 = (2.0 / (n as f64 * A0)).powi(3);
        let norm_part2 = gamma::gamma((n - l) as f64) / (2.0 * n as f64 * gamma::gamma((n + l + 1) as f64));
        let norm = (norm_part1 * norm_part2).sqrt();
        let r_wave = norm * (-rho / 2.0).exp() * rho.powi(l) * laguerre_l;
        sum += r * r * r_wave * r_wave;
        cdf[i] = sum;
    }
    for val in cdf.iter_mut() { *val /= sum; }
    let cdf_clone = cdf.clone();
    cache.push((n, l, cdf));
    let u: f64 = rand::thread_rng().gen();
    let idx = match cdf_clone.binary_search_by(|v| v.partial_cmp(&u).unwrap()) { Ok(i) => i, Err(i) => i };
    idx as f64 * (r_max / (N_CDF - 1) as f64)
}

fn sample_theta(l: i32, m: i32) -> f64 {
    lazy_static! { static ref CDF_CACHE: Mutex<Vec<(i32, i32, Vec<f64>)>> = Mutex::new(Vec::new()); }
    let m_abs = m.abs();
    let mut cache = CDF_CACHE.lock().unwrap();
    if let Some(entry) = cache.iter().find(|(cl, cm, _)| *cl == l && *cm == m_abs) {
        let cdf = &entry.2;
        let u: f64 = rand::thread_rng().gen();
        let idx = match cdf.binary_search_by(|v| v.partial_cmp(&u).unwrap()) { Ok(i) => i, Err(i) => i };
        return idx as f64 * (PI / (cdf.len() - 1) as f64);
    }
    const N_CDF: usize = 2048;
    let mut cdf = vec![0.0; N_CDF];
    let d_theta = PI / (N_CDF - 1) as f64;
    let mut sum = 0.0;
    for i in 0..N_CDF {
        let theta = i as f64 * d_theta;
        let plm = associated_legendre(l, m_abs, theta.cos());
        sum += theta.sin() * plm * plm;
        cdf[i] = sum;
    }
    for val in cdf.iter_mut() { *val /= sum; }
    let cdf_clone = cdf.clone();
    cache.push((l, m_abs, cdf));
    let u: f64 = rand::thread_rng().gen();
    let idx = match cdf_clone.binary_search_by(|v| v.partial_cmp(&u).unwrap()) { Ok(i) => i, Err(i) => i };
    idx as f64 * (PI / (N_CDF - 1) as f64)
}

// --- Polynomials and Color ---

fn associated_laguerre(k: i32, alpha: i32, x: f64) -> f64 {
    if k == 0 { return 1.0; }
    let mut lm1 = 1.0 + alpha as f64 - x;
    if k == 1 { return lm1; }
    let mut lm2 = 1.0;
    let mut l_val = 0.0;
    for j in 2..=k {
        l_val = ((2.0 * j as f64 - 1.0 + alpha as f64 - x) * lm1 - (j as f64 - 1.0 + alpha as f64) * lm2) / j as f64;
        lm2 = lm1;
        lm1 = l_val;
    }
    l_val
}

fn associated_legendre(l: i32, m: i32, x: f64) -> f64 {
    let m_abs = m.abs();
    let mut pmm = 1.0;
    if m_abs > 0 {
        let somx2 = ((1.0 - x) * (1.0 + x)).sqrt();
        let mut fact = 1.0;
        for _ in 1..=m_abs {
            pmm *= -fact * somx2;
            fact += 2.0;
        }
    }
    if l == m_abs { return pmm; }
    let mut pm1m = x * (2 * m_abs + 1) as f64 * pmm;
    if l == m_abs + 1 { return pm1m; }
    let mut pmm_temp = pmm;
    for ll in (m_abs + 2)..=l {
        let pll = ((2 * ll - 1) as f64 * x * pm1m - (ll + m_abs - 1) as f64 * pmm_temp) / (ll - m_abs) as f64;
        pmm_temp = pm1m;
        pm1m = pll;
    }
    pm1m
}

fn get_particle_color(r: f64, theta: f64, _phi: f64, n: i32, l: i32, m: i32) -> glm::Vec4 {
    let rho = 2.0 * r / (n as f64 * A0);
    let laguerre = associated_laguerre(n - l - 1, 2 * l + 1, rho);
    let norm_part1 = (2.0 / (n as f64 * A0)).powi(3);
    let norm_part2 = gamma::gamma((n - l) as f64) / (2.0 * n as f64 * gamma::gamma((n + l + 1) as f64));
    let r_wave = (norm_part1 * norm_part2).sqrt() * (-rho / 2.0).exp() * rho.powi(l) * laguerre;
    let angular = associated_legendre(l, m.abs(), theta.cos());
    let intensity = r_wave * r_wave * angular * angular;
    heatmap_cool(intensity * 1.5 * (5.0f64).powi(n))
}

fn heatmap_cool(value: f64) -> glm::Vec4 {
    let value = value.max(0.0).min(1.0);
    let colors = [
        glm::vec4(0.0, 0.0, 0.0, 1.0),   // Black
        glm::vec4(0.0, 0.0, 0.5, 1.0),   // Dark Blue
        glm::vec4(0.0, 0.8, 1.0, 1.0),   // Cyan
        glm::vec4(1.0, 1.0, 1.0, 1.0),   // White
    ];
    let scaled_v = value * (colors.len() - 1) as f64;
    let i = scaled_v as usize;
    let next_i = (i + 1).min(colors.len() - 1);
    let local_t = scaled_v - i as f64;
    let c1 = colors[i];
    let c2 = colors[next_i];
    let r = c1.x + local_t as f32 * (c2.x - c1.x);
    let g = c1.y + local_t as f32 * (c2.y - c1.y);
    let b = c1.z + local_t as f32 * (c2.z - c1.z);
    glm::vec4(r, g, b, 1.0)
}
