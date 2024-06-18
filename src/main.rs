use std::error::Error;
use std::sync::{Arc, Mutex, mpsc};
use std::thread;

use bevy::{
    prelude::*,
    window::{
        WindowMode,
        PrimaryWindow
    },
    core_pipeline::{
        bloom::{BloomCompositeMode, BloomSettings},
        tonemapping::Tonemapping
    }
};

use bevy_rapier3d::prelude::*;

use rand::{thread_rng, Rng};

mod data_loader;
use data_loader::{load_config_file, load_skymap_file, Config, SkyMap};

#[derive(Component)]
struct Player {
    stars_eaten: i32
}

#[derive(Resource)]
struct ControlMessageSender {
    sender: mpsc::Sender<String>,
}

#[derive(Resource)]
struct ControlMessageReceiver {
    receiver: Arc<Mutex<mpsc::Receiver<String>>>
}

#[derive(Resource)]
struct ControlMessage {
    message: String
}

fn main() {
    let config: Config = load_config_file("./config.toml");
    let skymap: SkyMap = load_skymap_file("./skymap.json").unwrap();
    let (sender, receiver) = mpsc::channel::<String>();
    
    App::new()
        .insert_resource(config)
        .insert_resource(skymap)
        .insert_resource(ControlMessageSender { sender })
        .insert_resource(ControlMessageReceiver { 
            receiver: Arc::new(Mutex::new(receiver))
        })
        .insert_resource(ControlMessage { message: String::new() })
        .add_plugins(DefaultPlugins)
        .add_plugins(RapierPhysicsPlugin::<NoUserData>::default())
        .add_systems(
            Startup,
            (
                start_subscriber_sending_events,
                window_setup,
                player_setup,
                text_setup,
                camera_setup,
                scene_setup
            ).chain()
        )
        .add_systems(
            Update,
            (
                receive_message,
                control_player_with_keyboard,
                control_player_by_control_message,
                check_sensor_collisions
            ).chain()
        )
        .run();
}

fn start_subscriber_sending_events(
    control_message_sender: Res<ControlMessageSender>
) {
    let sender = control_message_sender.sender.clone();
    thread::spawn(move || {
        let context = zmq::Context::new();
        let subscriber = context.socket(zmq::SUB).unwrap();
        
        assert!(subscriber.connect("tcp://127.0.0.1:5678").is_ok());
        assert!(subscriber.set_subscribe(b"").is_ok());
        
        let subscriber = Arc::new(Mutex::new(subscriber));
        
        loop {
            let subscriber = subscriber.lock().unwrap();
            match subscriber.recv_msg(0) {
                Ok(msg) => {
                    let message = msg.as_str().unwrap();
                    sender.send(message.to_string()).unwrap();
                },
                Err(e) => {
                    eprintln!("Failed to receive message: {}", e);
                },
            }
        }
    });
}

fn receive_message(
    control_message_receiver: Res<ControlMessageReceiver>,
    mut control_message: ResMut<ControlMessage>
) {
    let receiver = control_message_receiver.receiver.lock().unwrap();
    control_message.message = String::new();
    if let Ok(message) = receiver.try_recv() {
        control_message.message = message;
    }
}

fn window_setup(
	mut windows: Query<&mut Window, With<PrimaryWindow>>,
	config: Res<Config>
) {
	let mut window = windows.single_mut();
	window.cursor.visible = config.show_cursor;
	if config.fullscreen {
	    window.mode = WindowMode::Fullscreen;
	}
}

fn player_setup(
    config: Res<Config>,
    skymap: Res<SkyMap>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>
) {
    commands.spawn(
        (
            PbrBundle {
                mesh: meshes.add(
                    Sphere {
                        radius: config.player_sphere_radius,
                        ..default()
                    }
                ),
                material: materials.add(StandardMaterial {
                    base_color: Color::rgba(
                        config.player_color[0],
                        config.player_color[1],
                        config.player_color[2],
                        config.player_color[3]
                ),
                    metallic: 0.5,
                    ..default()
                }),
                transform: Transform::from_xyz(
                    skymap.player_position[0],
                    skymap.player_position[1],
                    skymap.player_position[2]
                ).looking_at(Vec3::from_array(skymap.player_looking_at), Vec3::Y),
                ..default()
            },
            RigidBody::Dynamic,
            Collider::ball(config.player_sphere_radius),
            Restitution::coefficient(config.restitution),
            GravityScale(config.gravity_scale),
            Velocity {
                linvel: Vec3::ZERO,
                angvel: Vec3::ZERO
            },
            Player {stars_eaten: 0}
        )
    );
}

fn text_setup(
    player: Query<&Player>,
    mut commands: Commands
) {
    let player = player.single();

    commands.spawn(
        TextBundle::from_section(
            format!("Stars eaten: {}", player.stars_eaten),
            TextStyle {
                font_size: 30.0,
                color: Color::WHITE,
                ..default()
            }
        ).with_style(
                Style {
                    position_type: PositionType::Absolute,
                    top: Val::Px(12.0),
                    left: Val::Px(12.0),
                    ..default()
                }
            ),
    );
}

fn camera_setup(
    config: Res<Config>,
    mut commands: Commands
) {
    commands.spawn(
        (
            Camera3dBundle {
                camera: Camera {
                    hdr: true,
                    ..default()
                },
                tonemapping: Tonemapping::TonyMcMapface,
                transform: Transform::from_xyz(
                    config.camera_position[0],
                    config.camera_position[1],
                    config.camera_position[2]
                ).looking_at(Vec3::from_array(config.camera_looking_at), Vec3::Z),
                ..default()
            },
            BloomSettings {
                intensity: config.bloom_intensity,
                composite_mode: BloomCompositeMode::Additive,
                ..default()
            },
            FogSettings {
                color: Color::rgb_from_array(config.fog_color),
                falloff: FogFalloff::Linear {
                    start: config.fog_falloff_start,
                    end: config.fog_falloff_end,
                },
                ..default()
            }
        )
    );
}


fn scene_setup(
    config: Res<Config>,
    skymap: Res<SkyMap>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>
) {
    if config.create_sky_dome {
        commands.spawn(PbrBundle {
            mesh: meshes.add(Sphere::default()),
            material: materials.add(
                StandardMaterial {
                    unlit: true,
                    base_color: Color::rgb_from_array(config.sky_dome_color),
                    ..default()
                }
            ),
            transform: Transform::default().with_scale(Vec3::splat(-config.sky_dome_size)),
            ..default()
            }
        );
    }

    if config.generate_stars {
        let mut random_number_generator = thread_rng();
        let mut new_sphere_x: f32;
        let mut new_sphere_y: f32;
        let mut new_sphere_z: f32;

        for _ in 1..config.number_of_spheres {
            new_sphere_x = random_number_generator.gen_range(config.lower_generation_space_limit..config.higher_generation_space_limit) as f32;
            new_sphere_y = random_number_generator.gen_range(config.lower_generation_space_limit..config.higher_generation_space_limit) as f32;
            new_sphere_z = random_number_generator.gen_range(config.lower_generation_space_limit..config.higher_generation_space_limit) as f32;
        
            commands.spawn(
                (
                    PbrBundle {
                        mesh: meshes.add(Sphere::new(config.sphere_radius)),
                        material: materials.add(StandardMaterial {
                            emissive: Color::rgba_linear_from_array(config.sphere_emissive_color),
                            ..default()
                        }),
                        transform: Transform::from_xyz(new_sphere_x, new_sphere_y, new_sphere_z),
                        ..default()
                    },
                    Collider::ball(config.sphere_radius),
                    Sensor,
                    ActiveEvents::COLLISION_EVENTS,
                    ActiveCollisionTypes::STATIC_STATIC
                )
            );
        }
    } else {
        for star_position in skymap.stars.iter() {
            commands.spawn(
                (
                    PbrBundle {
                        mesh: meshes.add(Sphere::new(config.sphere_radius)),
                        material: materials.add(StandardMaterial {
                            emissive: Color::rgba_linear_from_array(config.sphere_emissive_color),                    
                            ..default()
                        }),
                        transform: Transform::from_xyz(star_position[0], star_position[1], star_position[2]),
                        ..default()
                    },
                    Collider::ball(config.sphere_radius),
                    Sensor,
                    ActiveEvents::COLLISION_EVENTS,
                    ActiveCollisionTypes::STATIC_STATIC
                )
            );
        }
    }
}

fn control_player_with_keyboard(
    mut player_query: Query<(&mut Velocity, &mut Transform), With<Player>>,
    mut camera_query: Query<&mut Transform, (With<Camera>, Without<Player>)>,
    time: Res<Time>,
    keycode: Res<ButtonInput<KeyCode>>,
    config: Res<Config>
) {
    let delta_t = time.delta_seconds();
    let (mut velocity, mut transform) = player_query.single_mut();
     
    if keycode.pressed(KeyCode::KeyW) {
        let movement_direction = transform.forward();
        velocity.linvel += movement_direction * config.movement_velocity_factor;
    } else if keycode.pressed(KeyCode::KeyS) {
        let movement_direction = transform.back();
        velocity.linvel += movement_direction * config.movement_velocity_factor;
    } 
    
    if keycode.pressed(KeyCode::KeyA) {
        let movement_direction = transform.left();
        velocity.linvel += movement_direction * config.movement_velocity_factor;
    } else if keycode.pressed(KeyCode::KeyD) {
        let movement_direction = transform.right();
        velocity.linvel += movement_direction * config.movement_velocity_factor;
    } 
    
    if keycode.pressed(KeyCode::KeyE) {
        let movement_direction = transform.up();
        velocity.linvel += movement_direction * config.movement_velocity_factor;
    } else if keycode.pressed(KeyCode::KeyQ) {
        let movement_direction = transform.down();
        velocity.linvel += movement_direction * config.movement_velocity_factor;
    }

    match config.use_angular_velocity {
        false => {
            if keycode.pressed(KeyCode::ArrowUp) {
                transform.rotate_local_x(config.rotation_velocity_factor * delta_t);
            } 
            if keycode.pressed(KeyCode::ArrowDown) {
                transform.rotate_local_x(-config.rotation_velocity_factor * delta_t);
            }
            if keycode.pressed(KeyCode::ArrowLeft) {
                transform.rotate_local_y(config.rotation_velocity_factor * delta_t);
            } 
            if keycode.pressed(KeyCode::ArrowRight) {
                transform.rotate_local_y(-config.rotation_velocity_factor * delta_t);
            }    
        },
        true => {
            if keycode.pressed(KeyCode::ArrowUp) {
                let rotation_direction = transform.right();
                velocity.angvel += rotation_direction * config.rotation_velocity_factor * config.rotation_reduction_with_angular_velocity;
            } 
    
            if keycode.pressed(KeyCode::ArrowDown) {
                let rotation_direction = transform.left();
                velocity.angvel += rotation_direction * config.rotation_velocity_factor * config.rotation_reduction_with_angular_velocity;
            }
    
            if keycode.pressed(KeyCode::ArrowLeft) {
                let rotation_direction = transform.up();
                velocity.angvel += rotation_direction * config.rotation_velocity_factor * config.rotation_reduction_with_angular_velocity;
            } 
    
            if keycode.pressed(KeyCode::ArrowRight) {
                let rotation_direction = transform.down();
                velocity.angvel += rotation_direction * config.rotation_velocity_factor * config.rotation_reduction_with_angular_velocity;
            }
        },
    }
    
    if config.camera_attached_to_player {
        let mut camera_transform = camera_query.single_mut();
        camera_transform.translation = transform.translation + (transform.back() * config.camera_shift[0]) + (transform.up() * config.camera_shift[1]);
        camera_transform.look_at(
            transform.translation + (transform.forward() * config.camera_target[0]),
            transform.up() * config.camera_target[1]
        );
    }
}

fn control_player_by_control_message(
    mut player_query: Query<(&mut Velocity, &mut Transform), With<Player>>,
    mut camera_query: Query<&mut Transform, (With<Camera>, Without<Player>)>,
    time: Res<Time>,
    config: Res<Config>,
    control_message: Res<ControlMessage>
) {
    let delta_t = time.delta_seconds();
    let (mut velocity, mut transform) = player_query.single_mut();
    let external_control_key = &control_message.message;

    let control_message_is = |key: &str| {
        control_message.message == key.to_string()
    };
     
    if control_message_is("w") {
        let movement_direction = transform.forward();
        velocity.linvel += movement_direction * config.movement_velocity_factor;
    } else if control_message_is("s") {
        let movement_direction = transform.back();
        velocity.linvel += movement_direction * config.movement_velocity_factor;
    } 
    
    if control_message_is("a") {
        let movement_direction = transform.left();
        velocity.linvel += movement_direction * config.movement_velocity_factor;
    } else if control_message_is("d") {
        let movement_direction = transform.right();
        velocity.linvel += movement_direction * config.movement_velocity_factor;
    } 
    
    if control_message_is("e") {
        let movement_direction = transform.up();
        velocity.linvel += movement_direction * config.movement_velocity_factor;
    } else if control_message_is("q") {
        let movement_direction = transform.down();
        velocity.linvel += movement_direction * config.movement_velocity_factor;
    }

    match config.use_angular_velocity {
        false => {
            if control_message_is("i") {
                transform.rotate_local_x(config.rotation_velocity_factor * delta_t);
            } 
            if control_message_is("k") {
                transform.rotate_local_x(-config.rotation_velocity_factor * delta_t);
            }
            if control_message_is("j") {
                transform.rotate_local_y(config.rotation_velocity_factor * delta_t);
            } 
            if control_message_is("l") {
                transform.rotate_local_y(-config.rotation_velocity_factor * delta_t);
            }    
        },
        true => {
            if control_message_is("i") {
                let rotation_direction = transform.right();
                velocity.angvel += rotation_direction * config.rotation_velocity_factor * config.rotation_reduction_with_angular_velocity;
            } 
    
            if control_message_is("k") {
                let rotation_direction = transform.left();
                velocity.angvel += rotation_direction * config.rotation_velocity_factor * config.rotation_reduction_with_angular_velocity;
            }
    
            if control_message_is("j") {
                let rotation_direction = transform.up();
                velocity.angvel += rotation_direction * config.rotation_velocity_factor * config.rotation_reduction_with_angular_velocity;
            } 
    
            if control_message_is("l") {
                let rotation_direction = transform.down();
                velocity.angvel += rotation_direction * config.rotation_velocity_factor * config.rotation_reduction_with_angular_velocity;
            }
        },
    }
    
    if config.camera_attached_to_player {
        let mut camera_transform = camera_query.single_mut();
        camera_transform.translation = transform.translation + (transform.back() * config.camera_shift[0]) + (transform.up() * config.camera_shift[1]);
        camera_transform.look_at(
            transform.translation + (transform.forward() * config.camera_target[0]),
            transform.up() * config.camera_target[1]
        );
    }
}

fn check_sensor_collisions(
    mut collision_events: EventReader<CollisionEvent>,
    collider_query: Query<&Collider>,
    mut text_query: Query<&mut Text>,
    mut player_query: Query<(Entity, &mut Player)>,
    mut transform_query: Query<&mut Transform>,
    config: Res<Config>
) {
    let (player_entity, mut player) = player_query.single_mut();
    
    let mut text = text_query.single_mut();
    let text = &mut text.sections[0].value;
    
    for event in collision_events.read() {
        match event {
            CollisionEvent::Started(collider1, collider2, _) => {        
                if collider_query.get(*collider1).is_ok() && collider_query.get(*collider2).is_ok() {
                    let player_index = player_entity.index();
                    let collider1_index = collider1.index();
                    let collider2_index = collider2.index();
                    match player_index {
                        collider1_index => {
                            if let Ok(mut transform) = transform_query.get_mut(*collider1) {
                                let mut random_number_generator = thread_rng();
                                transform.translation = Vec3::new(
                                    random_number_generator.gen_range(config.lower_generation_space_limit..config.higher_generation_space_limit) as f32,
                                    random_number_generator.gen_range(config.lower_generation_space_limit..config.higher_generation_space_limit) as f32,
                                    random_number_generator.gen_range(config.lower_generation_space_limit..config.higher_generation_space_limit) as f32
                                );
                            }
                        },
                        collider2_index => {
                            if let Ok(mut transform) = transform_query.get_mut(*collider2) {
                                let mut random_number_generator = thread_rng();
                                transform.translation = Vec3::new(
                                    random_number_generator.gen_range(config.lower_generation_space_limit..config.higher_generation_space_limit) as f32,
                                    random_number_generator.gen_range(config.lower_generation_space_limit..config.higher_generation_space_limit) as f32,
                                    random_number_generator.gen_range(config.lower_generation_space_limit..config.higher_generation_space_limit) as f32
                                );
                            }
                        }
                    }
                }
            },
            CollisionEvent::Stopped(collider1, collider2, _) => {
                if collider_query.get(*collider1).is_ok() && collider_query.get(*collider2).is_ok() {
                    player.stars_eaten += 1;
                }
            },
        }
    }

    *text = format!("Stars collected: {}", player.stars_eaten);
}
