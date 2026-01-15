# Dynamic A# 알고리즘 예제
import math
from sys import maxsize # 무한대 사용
import matplotlib.pyplot as plt

show_animation = True


class State:

    def __init__(self, x, y): # 좌표
        self.x = x
        self.y = y
        self.parent = None # 부모 노드 : 경로 역추적용
        self.state = "." # 기본상태(new)
        self.t = "new"  # tag : new, open, close (알고리즘이 사용함) 
        self.h = 0 # 휴리스틱 비용 / (f_cost와 비슷한 개념)
        self.k = 0 # 최소 비용 / open list에서 최소 k값을 가진 노드 꺼냄 / 우선순위 개념
    
    # 이동비용(cost) 정의
    def cost(self, state): # self(현재칸)에서 state(다른칸)으로 가는 비용 계산
        if self.state == "#" or state.state == "#": # 장애물 칸(#)이면
            return maxsize # 무한대 비용 반환(이동불가)

        return math.sqrt(math.pow((self.x - state.x), 2) + # 유클리드 거리(대각선 이동 가능) = 두 칸 사이 거리 
                         math.pow((self.y - state.y), 2)) # pow:제곱 / sqrt:루트

    def set_state(self, state): # set_state()함수 : self 노드 상태(state)를 문자열로 바꾸는 함수 (노드에 라벨을 붙이는 함수)
        """
        .: new
        #: obstacle
        e: oparent of current state
        *: closed state
        s: current state
        """
        if state not in ["s", ".", "#", "e", "*"]: # 허용된 문자 아니면 무시
            return
        self.state = state # 현재 노드 상태 표시 = 문자열로(state) / 문자열 state를 self.state에 저장
        # ex) self.state = "#" 

class Map:

    def __init__(self, row, col):
        self.row = row # 맵 행 개수
        self.col = col # 맵 열 개수
        self.map = self.init_map() # 2D 리스트로 맵 생성

    def init_map(self): 
        map_list = [] # 맵 전체를 담을 리스트
        for i in range(self.row): # 세로 행 개수(row)만큼 i개 행이 만들어짐 
            tmp = [] # 행 하나를 임시로 담는 리스트 / 가로 한줄 
            for j in range(self.col): # 가로 열 개수(col)만큼 j개 열이 만들어짐
                tmp.append(State(i, j)) # (i, j) 좌표를 가진 State 객체 생성 후, tmp 리스트에 추가
            map_list.append(tmp) # 한 행이 완성되면, map_list에 추가
        return map_list # 2D 리스트(map_list) 반환
        
        """
        row = 2 (i=0,1)
        col = 3 (j=0,1,2)

        i = 0 → tmp = []
            j = 0 → tmp.append(State(0,0))
            j = 1 → tmp.append(State(0,1))
            j = 2 → tmp.append(State(0,2))
            map_list.append(tmp)

        i = 1 → tmp = []
            j = 0 → tmp.append(State(1,0))
            j = 1 → tmp.append(State(1,1))
            j = 2 → tmp.append(State(1,2))
            map_list.append(tmp)

        return map_list
                최종 map_list = 
                [[State(0,0), State(0,1), State(0,2)],
                 [State(1,0), State(1,1), State(1,2)]]

        """


    def get_neighbors(self, state): # 현재 노드(state)의 주변 8칸을 찾아서 리스트로 반환
        state_list = [] # 이웃 노드들을 담을 리스트(총 8개가 담길 수 있다)
        for i in [-1, 0, 1]: # (위, 가운데, 아래)
            for j in [-1, 0, 1]: # (왼쪽, 가운데, 오른쪽)
                if i == 0 and j == 0:
                    continue # (i, j) = (0, 0) = 현재 노드 자기자신 = 무시
                if state.x + i < 0 or state.x + i >= self.row: # 0 < state.x+i < row
                    continue # 맵 범위 벗어나면 무시 
                if state.y + j < 0 or state.y + j >= self.col: # 0 < state.y+j < col
                    continue # 무시  
                state_list.append(self.map[state.x + i][state.y + j]) # 유효한 이웃 노드면 이웃 리스트에 추가
        return state_list

    def set_obstacle(self, point_list): # 지정된 좌표를 장애물로 설정 함수
        for x, y in point_list: # point_list = (x1, y1), (x2, y2) ....
            if x < 0 or x >= self.row or y < 0 or y >= self.col: # 맵 범위 벗어나면 무시
                continue

            self.map[x][y].set_state("#") 


class Dstar:
    def __init__(self, maps):
        self.map = maps
        self.open_list = set() # open list 초기화 // set()함수 : 중복허용x  // open list : 현재 노드의 이웃노드8개 + 다음 확장한 노드의 이웃노드8개 ...
    
    def process_state(self): # 노드 확장(처리)하려고 할 때 
        x = self.min_state() # min_state() : open list에서 값이 가장 작은 노드!! 선택 = 우선순위(k)가 가장 높은 노드
        # x = 확장된 노드 (closed list에 추가될 노드)

        if x is None: # open list가 비어있으면 None 반환
            return -1

        k_old = self.get_kmin() # open list에서 (우선순위 가장 높은 노드의) 최소 k값!! 가져오기
        self.remove(x) # open list에서 x 노드 제거-> closed list에 추가됨

        if k_old < x.h: # 
            for y in self.map.get_neighbors(x): # 
                if y.h <= k_old and x.h > y.h + x.cost(y): # 
                    x.parent = y # 
                    x.h = y.h + x.cost(y) #
        if k_old == x.h:
            for y in self.map.get_neighbors(x):
                if y.t == "new" or y.parent == x and y.h != x.h + x.cost(y) \
                        or y.parent != x and y.h > x.h + x.cost(y):
                    y.parent = x
                    self.insert(y, x.h + x.cost(y))
        else:
            for y in self.map.get_neighbors(x):
                if y.t == "new" or y.parent == x and y.h != x.h + x.cost(y):
                    y.parent = x
                    self.insert(y, x.h + x.cost(y))
                else:
                    if y.parent != x and y.h > x.h + x.cost(y):
                        self.insert(x, x.h)
                    else:
                        if y.parent != x and x.h > y.h + x.cost(y) \
                                and y.t == "close" and y.h > k_old:
                            self.insert(y, y.h)
        return self.get_kmin()

    def min_state(self):
        if not self.open_list:
            return None
        min_state = min(self.open_list, key=lambda x: x.k)
        return min_state 

    def get_kmin(self):
        if not self.open_list:
            return -1
        k_min = min([x.k for x in self.open_list])
        return k_min

    def insert(self, state, h_new):
        if state.t == "new":
            state.k = h_new
        elif state.t == "open":
            state.k = min(state.k, h_new)
        elif state.t == "close":
            state.k = min(state.h, h_new)
        state.h = h_new
        state.t = "open"
        self.open_list.add(state)

    def remove(self, state):
        if state.t == "open":
            state.t = "close"
        self.open_list.remove(state)

    def modify_cost(self, x):
        if x.t == "close":
            self.insert(x, x.parent.h + x.cost(x.parent))

    def run(self, start, end):

        rx = []
        ry = []

        self.insert(end, 0.0)

        while True:
            self.process_state()
            if start.t == "close":
                break

        start.set_state("s")
        s = start
        s = s.parent
        s.set_state("e")
        tmp = start

        AddNewObstacle(self.map) # add new obstacle after the first search finished

        while tmp != end:
            tmp.set_state("*")
            rx.append(tmp.x)
            ry.append(tmp.y)
            if show_animation:
                plt.plot(rx, ry, "-r")
                plt.pause(0.01)
            if tmp.parent.state == "#":
                self.modify(tmp)
                continue
            tmp = tmp.parent
        tmp.set_state("e")

        return rx, ry

    def modify(self, state):
        self.modify_cost(state)
        while True:
            k_min = self.process_state()
            if k_min >= state.h:
                break

def AddNewObstacle(map:Map):
    ox, oy = [], []
    for i in range(5, 21):
        ox.append(i)
        oy.append(40)
    map.set_obstacle([(i, j) for i, j in zip(ox, oy)])
    if show_animation:
        plt.pause(0.001)
        plt.plot(ox, oy, ".g")

def main():
    m = Map(100, 100)
    ox, oy = [], []
    for i in range(-10, 60):
        ox.append(i)
        oy.append(-10)
    for i in range(-10, 60):
        ox.append(60)
        oy.append(i)
    for i in range(-10, 61):
        ox.append(i)
        oy.append(60)
    for i in range(-10, 61):
        ox.append(-10)
        oy.append(i)
    for i in range(-10, 40):
        ox.append(20)
        oy.append(i)
    for i in range(0, 40):
        ox.append(40)
        oy.append(60 - i)
    m.set_obstacle([(i, j) for i, j in zip(ox, oy)])

    start = [10, 10]
    goal = [50, 50]
    if show_animation:
        plt.plot(ox, oy, ".k")
        plt.plot(start[0], start[1], "og")
        plt.plot(goal[0], goal[1], "xb")
        plt.axis("equal")

    start = m.map[start[0]][start[1]]
    end = m.map[goal[0]][goal[1]]
    dstar = Dstar(m)
    rx, ry = dstar.run(start, end)

    if show_animation:
        plt.plot(rx, ry, "-r")
        plt.show()


if __name__ == '__main__':
    main()