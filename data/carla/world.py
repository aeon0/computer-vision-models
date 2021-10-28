import carla
import random


class World():
    def __init__(self, client: carla.Client):
        print("Available Maps:")
        print(client.get_available_maps())
        client.load_world('Town04')
        client.reload_world()
        self.carla_world: carla.World = client.get_world()
        blueprint_library: carla.BlueprintLibrary = self.carla_world.get_blueprint_library()

        # Create vehicle
        vehicle_bp = blueprint_library.filter('vehicle.citroen.c3')[0]
        spawn_point = self.carla_world.get_map().get_spawn_points()[86]
        self.ego_vehicle: carla.Vehicle = self.carla_world.spawn_actor(vehicle_bp, spawn_point)
        self.ego_vehicle.set_autopilot(False)

        for idx, waypoint in enumerate(self.carla_world.get_map().get_spawn_points()):
            self.carla_world.debug.draw_string(waypoint.location, str(idx), draw_shadow=False,
                                                color=carla.Color(r=255, g=0, b=0), life_time=1000,
                                                persistent_lines=True)

         # Move spectator cam into a topview close to the spawn point
        self.carla_world.get_spectator().set_transform(carla.Transform(
            self.ego_vehicle.get_location() + carla.Location(x=0, y=-10, z=100.0),
            carla.Rotation(pitch=-80)
        ))

        self.create_traffic(client, [88, 84, 73, 357])

    def create_traffic(self, client, waypoints):
        # Create some traffic
        tm: carla.TrafficManager = client.get_trafficmanager(8000)
        tm.set_global_distance_to_leading_vehicle(1.0)
        tm.set_hybrid_physics_mode(True)
        tm.global_percentage_speed_difference(98)
        # tm.set_random_device_seed(args.seed) # specific seed would always get same spawns
        vehicle_bp = self.carla_world.get_blueprint_library().filter("vehicle.*")
        walkers_bp = self.carla_world.get_blueprint_library().filter("walker.pedestrian.*")

        # filter vehicles that appreantly are "not safe" and "prone to accidents" lol
        vehicle_bp = [x for x in vehicle_bp if
            int(x.get_attribute('number_of_wheels')) == 4 and
            not x.id.endswith('isetta') and
            not x.id.endswith('carlacola') and
            not x.id.endswith('cybertruck') and
            not x.id.endswith('t2')
        ]
        # vehicle_bp = sorted(vehicle_bp, key=lambda bp: bp.id)

        spawn_points = self.carla_world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)
        print(f"Available spawn points: {number_of_spawn_points}")

        vehicle_batch = []
        self.vehicle_npcs = []

        for waypoint_idx in waypoints:
            bp = random.choice(vehicle_bp)
            if bp.has_attribute('color'):
                color = random.choice(bp.get_attribute('color').recommended_values)
                bp.set_attribute('color', color)
            if bp.has_attribute('driver_id'):
                driver_id = random.choice(bp.get_attribute('driver_id').recommended_values)
                bp.set_attribute('driver_id', driver_id)
            bp.set_attribute('role_name', 'autopilot')

            # prepare the light state of the cars to spawn
            light_state = carla.VehicleLightState.NONE
            # light_state = vls.Position | vls.LowBeam | vls.LowBeam

            # prepare commands to spawn the cars and set their autopilot and light state all together
            vehicle_batch.append(carla.command.SpawnActor(bp, self.carla_world.get_map().get_spawn_points()[waypoint_idx])
                .then(carla.command.SetAutopilot(carla.command.FutureActor, True, tm.get_port()))
                .then(carla.command.SetVehicleLightState(carla.command.FutureActor, light_state)))

        # create vehicles from batch data
        for response in client.apply_batch_sync(vehicle_batch, False):
            if response.error:
                print(response.error)
            else:
                self.vehicle_npcs.append(response.actor_id)

    def destroy(self):
        self.ego_vehicle.destroy()
        for vehicle in self.vehicle_npcs:
            vehicle.destroy()
