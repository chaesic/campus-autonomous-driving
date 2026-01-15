import carla
import random
import time
import sys
import os

# CARLA 에이전트 모듈 경로 설정 (본인 경로에 맞춰 확인)
try:
    sys.path.append(os.path.expanduser("~/carla/PythonAPI/carla"))
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    from agents.navigation.controller import VehiclePIDController
except ImportError:
    print("에이전트 모듈을 찾을 수 없습니다.")

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    amap = world.get_map()
    debug = world.debug
    vehicle = None
    spawn_points = amap.get_spawn_points()

    # z값은 바닥에 끼지 않게 2.0 정도로 넉넉히 줍니다.
    start_location = carla.Location(x=-52.1, y=100.2, z=2.0)
    end_location = carla.Location(x=-110.5, y=-3.3, z=2.0)

    try:
        # 1. 출발지와 목적지 선정
        # 맵에서 해당 좌표와 가장 가까운 도로 정보를 가져옵니다.
        start_point = amap.get_waypoint(start_location).transform
        # 차가 바닥에 겹치지 않게 살짝 띄워줍니다.
        start_point.location.z += 0.5 

        # 목적지는 경로 계산용이므로 좌표 그대로 사용합니다.
        end_point_location = end_location

        # 2. 경로 계산 (Global Planning)
        grp = GlobalRoutePlanner(amap, 2.0)
        route = grp.trace_route(start_location, end_location)
        
        # 빨간 점으로 경로 미리 그리기
        for wp, opt in route:
            debug.draw_point(wp.transform.location, size=0.1, color=carla.Color(255, 0, 0), life_time=100.0)

        # 3. 실제 주행할 차량 스폰
        blueprint = world.get_blueprint_library().find('vehicle.tesla.model3')
        vehicle = world.spawn_actor(blueprint, start_point)
        print("내 차량이 스폰되었습니다. 주행을 시작합니다.")

        # 4. PID 제어기 설정 (경로를 따라가게 하는 '운전자' 역할)
        # 차가 경로를 이탈하지 않게 핸들을 조절합니다.
        controller = VehiclePIDController(vehicle, 
            args_lateral={'K_P': 1.95, 'K_I': 0.07, 'K_D': 0.2, 'dt': 0.05},
            args_longitudinal={'K_P': 1.0, 'K_I': 0.05, 'K_D': 0, 'dt': 0.05})

        # 5. 주행 루프: 생성된 경로(route)의 점들을 하나씩 순서대로 따라감
        for waypoint, road_option in route:
            while True:
                vehicle_loc = vehicle.get_location()
                # 현재 위치가 목표 점(waypoint)에서 2미터 이내면 다음 점으로 넘어감
                if vehicle_loc.distance(waypoint.transform.location) < 2.0:
                    break

                # 목표 속도 30km/h로 설정하여 제어 신호 계산
                control = controller.run_step(30.0, waypoint)
                vehicle.apply_control(control)

                time.sleep(0.05)

        print("목적지에 도착했습니다!")

    finally:
        if vehicle is not None:
            vehicle.destroy()
            print("차량을 제거했습니다.")

if __name__ == '__main__':
    main()
