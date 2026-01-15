import carla
import random

client = carla.Client('localhost', 2000)
world = client.get_world()

client.load_world('Town05')

spectator = world.get_spectator()
transform = spectator.get_transform()
location = transform.location
rotation = transform.rotation
spectator.set_transform(carla.Transform())

#객체 생성
vehicle_blueprint = world.get_blueprint_library().filter('*vehicle*')

#스폰 포인트 설정
spawn_points = world.get_map().get_spawn_points()
for i in range(0,50):
    world.try_spawn_actor(random.choice(vehicle_blueprint), random.choice(spawn_points))    

ego_vehicle = world.spawn_actor(random.choice(vehicle_blueprint), random.choice(spawn_points))    

# Create a transform to place the camera on top of the vehicle
camera_init_trans = carla.Transform(carla.Location(z=1.5))
# We create the camera through a blueprint that defines its properties
camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
# We spawn the camera and attach it to our ego vehicle
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=ego_vehicle)
# Start camera with PyGame callback
#camera.listen(lambda image: image.save_to_disk('out/%06d.png' % image.frame))

for vehicle in world.get_actors().filter('*vehicle*'):
    vehicle.set_autopilot(True)


# 1. 트래픽 매니저(TM) 생성 (이게 있어야 오토파일럿이 작동합니다)
tm = client.get_trafficmanager(8000)
tm.set_global_distance_to_leading_vehicle(2.5) # 차간 거리 설정
tm.set_synchronous_mode(False) # 비동기 모드로 설정하여 부하 감소

# 2. 모든 차량을 찾아서 오토파일럿 연결
vehicles = world.get_actors().filter('*vehicle*')
for v in vehicles:
    v.set_autopilot(True, tm.get_port()) # TM 포트를 정확히 지정

print(f"현재 {len(vehicles)}대의 차량이 주행을 시작합니다!")