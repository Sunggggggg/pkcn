import torch
import itertools

def cosine_similarity_matrix(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    A: (n, d) 텐서
    B: (m, d) 텐서
    리턴: (n, m) 코사인 유사도 행렬
    """
    A_norm = torch.nn.functional.normalize(A, p=2, dim=1)
    B_norm = torch.nn.functional.normalize(B, p=2, dim=1)
    return torch.matmul(A_norm, B_norm.t())

def pairwise_feature_similarity_sum(A: torch.Tensor, B: torch.Tensor) -> float:
    """
    A: (5, d) 기준 사람 특징 벡터
    B: (5, d) 비교 사람 특징 벡터
    → 코사인 유사도 총합 (float)
    """
    sim = cosine_similarity_matrix(A, B)  # (5, 5)
    return sim.sum().item()

def match_features(baseline: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    """
    baseline: (b, d) 텐서, 기준 뷰의 특징 벡터 (b는 보통 2여야 함)
    features: (m, d) 텐서, 비교 뷰의 특징 벡터 (m는 0, 1 또는 2 이상)
    
    리턴: 길이가 2인 torch.Tensor (dtype=torch.int64)로, 최종 출력은 항상 2개.
      - 만약 baseline이 0개라면 tensor([-1, -1]) 반환.
      - baseline이 1개인 경우: 해당 검출과의 매칭 결과를 복제하여 반환.
      - features가 0개이면 tensor([-1, -1]) 반환.
      - features가 1개이면, 그 검출이 baseline의 어느 사람에 더 가까운지 판단한 후 그 ID를 두 자리에 모두 할당.
      - features가 2개 이상이면, 가능한 두 검출의 조합 중 비용(1-유사도 합)이 최소인 경우에 따라 baseline 매칭 순서를 반환.
    """
    # baseline 처리: baseline이 0개인 경우
    b = baseline.size(0)
    if b == 0:
        return torch.full((2,), -1, dtype=torch.int64)
    # baseline이 1개인 경우 -> 결과는 [0, 0]
    if b == 1:
        baseline = baseline  # 그대로 사용
        default_assignment = torch.tensor([0, 0], dtype=torch.int64)
    else:
        default_assignment = None

    m = features.size(0)
    # features가 없는 경우
    if m == 0:
        return torch.full((2,), -1, dtype=torch.int64)
    
    # features가 1개인 경우: 해당 검출에 대해 각 baseline과의 비용 비교
    if m == 1:
        sim_matrix = cosine_similarity_matrix(baseline, features)  # (b, 1)
        cost_matrix = 1 - sim_matrix  # (b, 1)
        # 만약 baseline이 1개라면 복제된 default_assignment 사용, 
        # baseline이 2개 이상이면 가장 낮은 비용을 가진 baseline의 인덱스를 선택
        if b >= 2:
            best = 0 if cost_matrix[0, 0] <= cost_matrix[1, 0] else 1
            return torch.tensor([best, best], dtype=torch.int64)
        else:
            return default_assignment

    # features가 2개 이상인 경우
    cost_matrix = 1 - cosine_similarity_matrix(baseline, features)  # shape: (b, m)
    best_total = float('inf')
    best_assignment = None  # 최종적으로 길이가 2인 할당 (baseline의 인덱스 2개)
    
    # b가 1 이상일 때, 최종 출력은 2개여야 하므로,
    # 만약 baseline의 검출 수가 2 이상이면 가능한 조합을 고려,
    # baseline이 1개인 경우는 위에서 처리됨.
    if b >= 2:
        # 가능한 두 검출 조합(인덱스 i,j) 중에서 평가
        for i, j in itertools.combinations(range(m), 2):
            cost1 = cost_matrix[0, i] + cost_matrix[1, j]  # baseline0->i, baseline1->j
            cost2 = cost_matrix[0, j] + cost_matrix[1, i]  # baseline0->j, baseline1->i
            if cost1 < best_total:
                best_total = cost1.item()
                best_assignment = torch.tensor([0, 1], dtype=torch.int64)
            if cost2 < best_total:
                best_total = cost2.item()
                best_assignment = torch.tensor([1, 0], dtype=torch.int64)
        return best_assignment
    else:
        # b == 1인 경우는 이미 위에서 처리되었음
        return default_assignment

def match_features_multi(baseline: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    """
    baseline: (b, b*5, d) ← 기준 사람들
    features: (m, m*5, d) ← 비교 사람들
    반환: 길이 2인 torch.Tensor (ID 매핑)
    """
    b, _, d = baseline.shape
    m, _, _ = features.shape

    if b == 0 or m == 0:
        return torch.full((2,), -1, dtype=torch.int64)
    if b == 1:
        return torch.tensor([0, 0], dtype=torch.int64)
    if m == 1:
        sims = [pairwise_feature_similarity_sum(baseline[i].view(5, d), features[0].view(5, d)) for i in range(2)]
        best = int(torch.argmax(torch.tensor(sims)))
        return torch.tensor([best, best], dtype=torch.int64)

    # m >= 2인 경우: 모든 조합 비교 (i, j)
    best_score = float('-inf')
    best_assignment = torch.tensor([-1, -1], dtype=torch.int64)

    for i, j in itertools.combinations(range(m), 2):
        f_i = features[i].view(5, d)
        f_j = features[j].view(5, d)

        # 매칭 1: baseline[0] → i, baseline[1] → j
        s1 = pairwise_feature_similarity_sum(baseline[0].view(5, d), f_i)
        s2 = pairwise_feature_similarity_sum(baseline[1].view(5, d), f_j)
        total1 = s1 + s2

        # 매칭 2: baseline[0] → j, baseline[1] → i
        s1_rev = pairwise_feature_similarity_sum(baseline[0].view(5, d), f_j)
        s2_rev = pairwise_feature_similarity_sum(baseline[1].view(5, d), f_i)
        total2 = s1_rev + s2_rev

        if total1 > best_score:
            best_score = total1
            best_assignment = torch.tensor([0, 1], dtype=torch.int64)
        if total2 > best_score:
            best_score = total2
            best_assignment = torch.tensor([1, 0], dtype=torch.int64)

    return best_assignment

def reassign_ids(data: list) -> torch.Tensor:
    """
    data: 각 뷰별 특징 텐서의 리스트.
          첫 번째 뷰(data[0])는 기준 뷰.
          최종 출력은 (N, 2) 텐서 (dtype=torch.int64).
    """
    num_views = len(data)
    results = []
    # 기준 뷰: 만약 검출 수가 2개 미만이면, 2개로 맞추기 위해 복제 혹은 패딩
    baseline = data[0]
    b = baseline.size(0)
    if b == 0:
        baseline_ids = torch.full((2,), -1, dtype=torch.int64)
    elif b == 1:
        baseline_ids = torch.tensor([0, 0], dtype=torch.int64)
    else:
        baseline_ids = torch.arange(b, dtype=torch.int64)[:2]
    results.append(baseline_ids)
    
    for i in range(1, num_views):
        current = data[i]
        assigned = match_features(baseline, current)
        results.append(assigned)
    return torch.stack(results, dim=0)

def reassign_ids_multi(data: list) -> torch.Tensor:
    """
    data: 각 뷰별 특징 벡터 리스트. shape = [N, N*5, d] 
    반환: (num_views, 2) 매칭 ID
    """
    num_views = len(data)
    results = []

    baseline = data[0]
    b = baseline.size(0)

    if b == 0:
        baseline_ids = torch.full((2,), -1, dtype=torch.int64)
    elif b == 1:
        baseline_ids = torch.tensor([0, 0], dtype=torch.int64)
    else:
        baseline_ids = torch.arange(b, dtype=torch.int64)[:2]
    results.append(baseline_ids)

    for i in range(1, num_views):
        current = data[i]
        assigned = match_features_multi(baseline, current)
        results.append(assigned)

    return torch.stack(results, dim=0)

def match_features_ids(baseline1, baseline2, current1, current2):
    b = baseline1.size(0)
    if b == 0:
        return torch.full((2,), -1, dtype=torch.int64)
    elif b == 1:
        return torch.tensor([0, 0], dtype=torch.int64)
    else:
        sim1 = cosine_similarity_matrix(baseline1, current1)  # [2, 2]
        sim2 = cosine_similarity_matrix(baseline2, current2)  # [2, 2]
        total_score_01 = sim1[0,0] + sim1[1,1] + sim2[0,0] + sim2[1,1]
        total_score_10 = sim1[0,1] + sim1[1,0] + sim2[0,1] + sim2[1,0]
        if total_score_01 >= total_score_10:
            return torch.tensor([0, 1], dtype=torch.int64)
        else:
            return torch.tensor([1, 0], dtype=torch.int64)


def assign_all_ids(feat_list1, feat_list2):
    """
    feat_list1: [N, 2, D] (baseline1, + 나머지 N-1개)
    feat_list2: [N, 2, D] (baseline2, + 나머지 N-1개)
    출력: [N-1, 2] (ID 매칭 결과, 0번 인덱스는 기준이므로 생략)
    """
    N = feat_list1.shape[0]
    results = []
    
    baseline = feat_list1[0]
    b = baseline.size(0)
    
    baseline1 = feat_list1[0]
    baseline2 = feat_list2[0]
    
    if b == 0:
        baseline_ids = torch.full((2,), -1, dtype=torch.int64)
    elif b == 1:
        baseline_ids = torch.tensor([0, 0], dtype=torch.int64)
    else:
        baseline_ids = torch.arange(b, dtype=torch.int64)[:2]
    results.append(baseline_ids)
    
    for n in range(1, N):
        current1 = feat_list1[n]
        current2 = feat_list2[n]
        assigned = match_features_ids(baseline1, baseline2, current1, current2)
        results.append(assigned)
    return torch.stack(results, dim=0)

if __name__ == "__main__":
    x1 = torch.rand((4, 2, 400))
    x2 = torch.rand((4, 2, 400))
    
    ids = assign_all_ids(x1, x2)
    print(ids)