# AIC RacingKart 36초 목표 인수인계

갱신: 2026-07-29 (KST)

## 오늘 확인된 것 — 앞의 인수인계에서 정정해야 하는 부분

### 1. 기준선 설정은 문서보다 `output/20260728-172948/d1/autoware.log`가 정확하다

그 로그에는 실행 당시의 config가 주석까지 그대로 찍혀 있고, 이전 인수인계의 서술과 여러 곳이 다르다. 47.28초를 낸 실제 설정:

| 항목 | 기준선(검증됨) | 오늘 아침 작업 트리 |
|---|---|---|
| `v_max` | **33.0** (실규정 상한 35) | 40.0 |
| `delta_max_deg` | **36.0** (차량 한계 0.64 rad) | 32.0 |
| `max_width` | **3.0** | 6.0 |
| `Q` | **[1e6, 1e8, 1.85e6]** | [3e6, 9e7, 1e5] |
| `QN` | **[1e6, 1e6, 1e4]** | [1e6, 1e3, 1e4] |
| ref_vel 직선 | **30** | 40 |

즉 작업 트리에는 검증 안 된 실험값이 여럿 섞여 있었다. **새 실험 전에 먼저 이 로그에서 기준선을 복원할 것.**

그 로그에 남은 다른 확정 사실들:

- `a_max`: 이 차량의 물리적 가속 능력이 **약 1.37 m/s²**(AWSIM handicap)이고, `a_max=2.7`로 올려도 랩타임이 동일했다. 1.35가 실제 최적이며 **가속 한계는 튜닝으로 못 넘는다.**
- `a_min=-2.7`, `N=30`, `R=10000`, `ay_max=28`, `use_max_kappa_pred=false`: 전부 시도했고 전부 무변화. 다시 하지 말 것.
- `smoothing_distance=5`: 76.59초로 크게 악화.

### 2. `load_ref_path`는 psi/kappa도 읽지만 `create_ref_path`가 버린다

```python
wp_x, wp_y, _, _ = load_ref_path(...)
```

psi/kappa는 재샘플링된 x/y에서 다시 계산된다. 그래서 **경로 CSV의 psi가 공식과 90° 달라도 무해하다** — 앞 인수인계가 출발 실패 원인으로 지목한 것은 원인이 아니다.

### 3. CSV 속도 분기에 단위 버그가 있었다 (수정함)

`mpc_controller.py`의 `use_csv_speed_profile` 분기가 config의 km/h `v_max`를 변환 없이 `update_v_max()`에 넘겨, MPC 속도 상한이 11.1 m/s가 아니라 **40 m/s(144 km/h)**가 되고 있었다. 다른 두 호출부는 `kmh_to_m_per_sec()`로 변환한다. 수정 완료.

### 4. 코너 속도는 공짜가 아니었다 — 실패한 실험

36초 주행 telemetry를 구간별로 재보면 우리 프로파일이 거꾸로 붙어 있다:

| 구간 | 기준선 | 36초 주행 min..mean |
|---|---|---|
| s4 (wp155-190) | 22 | 31.0 .. 33.1 |
| s6 (wp265-290) | 22 | 30.4 .. 31.3 |
| s8 (wp320-335) | 25 | 33.3 .. 33.7 |
| s5 (wp190-265) | 30 | 28.6 .. 30.3 |

그래서 s4/s6를 30, s8을 31로 올려 돌렸더니 **47 m 지점에서 완전 정지**(조향 최대 포화 0.93 rad, 가속 명령 최대). 그 속도들은 36초 주행이 **자기 325 m 라인에서** 낸 값이고, 공식 346 m 라인의 코너 형상에서는 통과되지 않는다.

교훈: telemetry 속도는 그 라인과 묶여 있다. 속도만 옮기는 건 안 된다.

### 5. 그래서 실질 차이는 라인 길이다

| | 길이 | \|k\|p95 | 폐곡선 갭 |
|---|---|---|---|
| 공식 final / ver2 / ver3 / ver4 | 346.1 / 345.5 / 346.0 / 344.6 | 0.185~0.194 | 0.00 |
| **traj_top36_line.csv** | **323.6** | **0.187** | **0.00** |
| traj_top36_telemetry.csv | 325.1 | 0.197 | 0.51 |
| traj_top36_fitted.csv | 328.6 | 0.206 | 0.60 |

공식 경로 4개 버전은 서로 1.5 m 안쪽이라 **공식 저장소에 더 빠른 라인은 없다.** 우승자는 트랙 폭을 써서 mincurv 안쪽을 잘랐고, 그 20 m가 남은 격차의 실체다. `traj_top36_line.csv`는 폐곡선 갭 0.00에 곡률 p95가 공식보다 오히려 낮아 기하학적으로 멀쩡한 후보다.

### 6. eval은 벽 복구가 없다 — 제출물 쪽 복구 노드를 배선했다

`aichallenge/simulator_scripts/eval.sh`는 `--wall-recovery off`로 시뮬레이터를 띄운다. 벽에 닿으면 그대로 끝이고, 오늘 두 번의 실패가 정확히 그 형태였다(멈춘 자리에서 세션 종료까지 정지).

`stuck_recovery_controller` 패키지는 **이미 `aichallenge_submit`에 있었지만 아무 launch도 띄우지 않는 죽은 코드**였다. 배선함:

- `control/mpc.launch.xml`: `output_control_cmd` 인자 + remap 추가
- `reference.launch.xml`: mpc 출력을 `nominal_control_cmd`로 보내고 그 사이에 노드 삽입
- `aichallenge_submit_launch/package.xml`: `exec_depend` 추가

우리 트리의 구현(244행)은 upstream `agent/recovery-supervisor` 브랜치(138행)보다 앞서 있다 — 감지 0.4초, 깊은 후진, 쿨다운, 상호 양보 교착 시 크리프 탈출, 다중차량용 도메인별 복구 조향. upstream에서 되가져올 것 없음.

## 오프라인 랩타임 모델 (`tools/lap_time_model.py`)

컨트롤러의 재샘플링·스무딩·곡률·구간 조회를 그대로 재현해서, 빌드+eval 한 사이클(약 10분) 없이 후보를 점수화한다.

검증:

| 조합 | 예측 | 실측 |
|---|---|---|
| 공식 + 기준선 속도 | 45.31 | **47.28** |
| telemetry 라인 + telemetry 속도 | 36.98 | **36.319** |

telemetry 쪽이 정확한 이유는 그 속도가 실제로 달성된 값이라서다. `ref_vel` 예측은 약 **+2초** 낙관적이고, 그 차이가 MPC 추종 손실이다. **ref_vel 후보를 볼 때는 예측에 2초를 더해 읽을 것.**

주의: 이 모델의 "a_max 손실" 항목은 오해를 부른다. 차량이 어차피 1.37 m/s²에 걸리므로 급변동 프로파일이 추가로 손해를 보는 게 아니다. 비교용으로만 쓸 것.

## 현재 작업 트리 상태

- config: 공식 기준선 값 복원 + `csv_path`를 `traj_top36_line.csv`로 변경, `use_csv_speed_profile: false`
- ref_vel: 기준선 속도(30 / s4 22 / s6 22 / s8 25), **wp_id를 새 라인 인덱스로 재매핑**(좌표 이격 최대 1.73 m, 단조 유지, wp 1~55는 이격 0.00으로 출발 구간 동일)
- 예측 42.21초 → 실측 44초 전후 기대
- 복구 노드 배선 완료
- 실험 파일 다수 untracked이므로 broad clean/reset 금지

## 다음에 할 일

1. 위 조합을 eval로 검증한다. 벽에 닿아도 복구 노드가 살려내므로, 결과에서 **어느 구간에서 복구가 발동했는지** 로그로 확인할 것(`stuck detected` 메시지).
2. 복구가 특정 코너에서 반복 발동하면 그 구간만 속도를 내린다. 라인은 유지.
3. 라인이 통하면 그 다음에 코너 속도를 한 구간씩 올린다. 한 번에 한 구간, 한 변수.
4. `dev.sh`는 `--wall-recovery on`이므로 후보 스크리닝은 dev로 먼저 하는 게 싸다. eval은 확정 후보에만 쓸 것.

## 참고

- 리더보드(`https://aichallenge-board.jsae.or.jp/public/live`)는 **로그인이 필요**하다. 헤드리스 브라우저로도 데이터가 안 나온다.
- 36.319초 rosbag은 로컬에 없다. URL은 `https://d3al8lo5i04x19.cloudfront.net/32cc7c53-7277-4c84-8d08-6f9f5d27d331/1/rosbag2_autoware.mcap`. 다만 필요한 telemetry는 이미 `env/final_ver3/traj_top36_telemetry.csv`(544점, 0.6 m 간격)로 추출돼 있다.
- 기준 결과: `output/20260728-172948/d1/{result-summary.json,autoware.log}`
- 진단 대시보드: 이 세션에서 만든 artifact (구간 속도 역전 + 격차 분해 시각화)
