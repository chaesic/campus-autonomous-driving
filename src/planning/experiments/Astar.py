# A* 알고리즘 : 휴리스틱 f=g+h(현재비용+예상비용) = f가 가장 적은 경로를 우선 선택
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors



class Node:
    def __init__(self, pos: tuple, g_cost=0.0, h_cost=0.0): # pos: (x, y) = 그리드 상의 위치
        self.pos = pos # (x, y) 좌표
        self.g_cost = g_cost # 시작점에서 현재까지 온 실제 거리
        self.h_cost = h_cost # 목표까지 남은 거리(휴리스틱)
        self.f_cost = g_cost + h_cost # 총 비용
        self.parent = None # 부모 노드: 목표 도달시, 어느길로 왔는지 역추적

    def __lt__(self, other): # f_cost 기준으로 노드 비교 // less than : 비교 연산자
        return self.f_cost < other.f_cost # 작으면 True 반환 // 노드끼리 비교 시, f_cost가 더 작은쪽 = 더 좋은 노드


class AStar:
    def __init__(self, map_grid):
        self.open = []   # open list 초기화 // open : 아직 방문x, 탐색후보인 노드 목록
        self.closed = [] # closed list 초기화 // closed : 이미 방문한 노드 목록
        self.map_grid = map_grid # 2D numpy 배열로 맵 표현 (0: 이동가능, 1: 장애물)

    def search(self, start_node, goal_node): # A* 탐색 시작
        self.open.append(start_node) # 탐색 시작 시, start node를 open list에 넣고 시작

        while self.open: # open list가 빌 때까지 반복 / 아직 탐색할 노드가 남아있으면 계속 진행
            # f_cost가 가장 낮은 노드 선택
            self.open.sort() # f_cost 기준으로 open list 정렬
            current_node = self.open.pop(0) # f_cost가 가장 노드를 꺼내서 current_node로 설정
            self.closed.append(current_node) # 그 노드를 closed list에 넣어서 탐색 완료 표시

            # 목표 도달 시, 경로 반환
            if current_node.pos == goal_node.pos:
                return self.reconstruct_path(current_node)

            # 목표 노드가 아니라면, 이웃 노드 탐색    
            for neighbor_pos in self.get_neighbors(current_node): # 현재 노드 주변(상하좌우)칸 모두 탐색
                # closed list에 이미 있는지 노드인지 확인
                if any(n.pos == neighbor_pos for n in self.closed): 
                    continue # 이미 완료된 노드면 무시

                g_cost = current_node.g_cost + 1 # 이동 비용 (1로 고정) //현재까지 온거리(g)+ 한칸 이동
                h_cost = self.heuristic(neighbor_pos, goal_node.pos) # 휴리스틱 비용 계산
                f_cost = g_cost + h_cost # 총 비용 계산

                # open list에 이미 있는 노드인지 확인
                existing = next((n for n in self.open if n.pos == neighbor_pos), None) # 기존 노드 찾기

                if existing: # 이미 open list에 있는 노드라면
                    if existing.f_cost > f_cost: # 더 나은 경로 발견 시(새로 계산한 f_cost가 더 작으면)
                        self.update_node(existing, current_node, g_cost, h_cost) # 노드정보를 새값으로 업데이트
                else: # open에 없는 새로운 노드면, open list에 추가
                    new_node = Node(neighbor_pos, g_cost, h_cost) # 새 노드 생성
                    new_node.parent = current_node # 부모 노드 설정
                    self.open.append(new_node) # open list에 추가

        # open리스트가 다 비었는데 목표를 못찾았다면, no path found
        return None

    def get_neighbors(self, node): # 현재 노드 기준 상하좌우 이웃 노드 위치 구하기
        dirs = [(1,0), (0,1), (-1,0), (0,-1)] # 상하좌우 방향 벡터
        neighbors = [] # 이웃 노드 위치들

        for dx, dy in dirs: # 각 방향에 대해
            x, y = node.pos[0] + dx, node.pos[1] + dy # 현재 노드 위치에서 방향 벡터 더해서(각 방향으로 한칸씩 이동해서) 새 좌표 계산
            if 0 <= x < self.map_grid.shape[0] and 0 <= y < self.map_grid.shape[1]: # 맵 범위 내인지 확인
                if self.map_grid[x, y] != 1: # 장애물(1)이 아닌지 확인
                    neighbors.append((x, y)) # 유효한 이웃 노드 위치 추가
        return neighbors # 이웃 노드 위치들 반환

    def heuristic(self, pos, goal_pos): # 휴리스틱 함수 계산
        #: 현재위치와 목표 위치의 거리차를 절대값으로 더함
        return abs(pos[0] - goal_pos[0]) + abs(pos[1] - goal_pos[1]) 
        #가로차이+세로차이 = (xn - xg) + (yn - yg)
        #대각선 이동이 불가능할때 => 휴리스틱 사용

    #목표 도달시 경로 복원
    def reconstruct_path(self, node): # 목표 노드에서 시작 노드까지 경로 복원
        path = [] # 경로 리스트
        current = node # 현재 노드부터 시작
        while current is not None: # 부모 노드가 없을 때까지
            path.append(current.pos) # 현재 노드 위치 추가
            current = current.parent # 부모 노드로 이동
        return path[::-1] # 경로 뒤집어서 반환(시작->목표 순서로)

    # 더 좋은 경로 발견 시, 그 노드 정보(g,h,f 비용)를 업데이트
    def update_node(self, node, parent, g_cost, h_cost): 
        node.g_cost = g_cost 
        node.h_cost = h_cost 
        node.f_cost = g_cost + h_cost
        node.parent = parent # 부모 노드 업데이트


# ==== 예제 맵 ======
grid = np.array([
    [0,0,0,0,1],
    [0,0,0,0,0],
    [0,0,1,0,0],
    [0,0,1,0,0],
    [0,0,0,0,0]
])

start = Node((0,0))
goal = Node((4,4))

astar = AStar(grid) # Astar 객체 생성
path = astar.search(start, goal) # search()로 경로 탐색, 결과는 path에 저장

print("최단 경로:", path)


# ==== 경로 시각화 =====
cmap = colors.ListedColormap(['white', 'black', 'green', 'red', 'blue'])
visual_grid = np.copy(grid)
for x, y in path:
    visual_grid[x, y] = 4  # 파란색 경로 표시
visual_grid[start.pos[0], start.pos[1]] = 2  # 시작점 초록색
visual_grid[goal.pos[0], goal.pos[1]] = 3    # 목표점 빨간색
plt.imshow(visual_grid, cmap=cmap)
plt.title("A* Path Planning")
plt.show()
    
