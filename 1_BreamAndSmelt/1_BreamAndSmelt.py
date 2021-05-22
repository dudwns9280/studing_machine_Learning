import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

# 도미의 데이터와 빙어의 데이터 합치기.
length = bream_length + smelt_length
weight = bream_weight + smelt_weight

# 사이킷런 패키지를 사용하기 위해 각 특성의 리스트를 세로 방향으로 늘어뜨린 2차원 리스트로 만들어야함.
fish_data = [[l,w] for l, w in zip(length, weight)]

# 도미는 1로 갱신하고, 빙어는 0으로 갱신함.
fish_target = [1] * 35 + [0] * 14

# k-최근접 이웃 알고리즘으로 훈련 시킴. ( 기본적으로 이웃된 5개의 데이터를 참고하여 결과값 도출)
kn = KNeighborsClassifier()
kn.fit(fish_data, fish_target) 
kn.score(fish_data, fish_target)

# 이웃된 49개의 데이터를 참고하여 결과값 도출. ( 빙어와 도미가 섞여있는데 도미가 다수이기때문에 어떤 데이터를 넣어도 도미로 예측함.)
kn49 = KNeighborsClassifier(n_neighbors=49)
kn49.fit(fish_data, fish_target)
kn49.score(fish_data, fish_target)

# 길이가 30이고 무게가 600인 생선은 ? 도미라고 판단.
kn.predict([[30, 600]])
kn49.predict([[9, 6]])

# 이웃된 데이터 갯수를 조정하여 1.0 아래로 내려가기 시작하는 이웃의 갯수 찾기
kn_test = KNeighborsClassifier()
kn_test.fit(fish_data, fish_target)

for n in range(5,50) :
  kn_test.n_neighbors = n
  score = kn_test.score(fish_data, fish_target)

  if score < 1 :
    print(n, score)


## 빙어와 도미의 산점도 그래프 그리기.
# plt.scatter(bream_length, bream_weight)
# plt.scatter(smelt_length, smelt_weight)
# plt.xlabel("length")
# plt.ylabel("weight")
# plt.show()