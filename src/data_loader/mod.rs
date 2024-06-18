use bevy::prelude::*;
use config_file::FromConfigFile;
use serde::Deserialize;

use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

#[derive(Resource, Deserialize)]
pub struct Config {
    pub fullscreen: bool,
    pub show_cursor: bool,
    pub camera_shift: [f32; 2],
    pub camera_target: [f32; 2],
    pub camera_position: [f32; 3],
    pub camera_looking_at: [f32; 3],
    pub rotation_velocity_factor: f32,
    pub movement_velocity_factor: f32,
    pub create_sky_dome: bool,
    pub sky_dome_size: f32,
    pub sky_dome_color: [f32; 3],
    pub fog_color: [f32; 3],
    pub fog_falloff_start: f32,
    pub fog_falloff_end: f32,
    pub sphere_radius: f32,
    pub sphere_emissive_color: [f32; 4],
    pub generate_stars: bool,
    pub number_of_spheres: i32,
    pub lower_generation_space_limit: i32,
    pub higher_generation_space_limit: i32,
    pub motion_blur_shutter_angle: f32,
    pub motion_blur_sample_number: u32,
    pub bloom_intensity: f32,
    pub camera_attached_to_player: bool,
    pub player_color: [f32; 4],
    pub player_sphere_radius: f32,
    pub max_player_velocity: f32,
    pub restitution: f32,
    pub use_angular_velocity: bool,
    pub rotation_reduction_with_angular_velocity: f32,
    pub gravity_scale: f32,
}

#[derive(Resource, Deserialize, Debug)]
pub struct SkyMap {
    pub stars: Vec<[f32; 3]>,
    pub player_position: [f32; 3],
    pub player_looking_at: [f32; 3],
}

pub fn load_config_file(path: &str) -> Config {
    Config::from_config_file(path).unwrap()
}

pub fn load_skymap_file<P: AsRef<Path>>(path: P) -> Result<SkyMap, Box<dyn Error>> {
    let file = File::open(path).expect("There was an error when reading file");
    let reader = BufReader::new(file);
    let skymap = serde_json::from_reader(reader)?;
    Ok(skymap)
}
