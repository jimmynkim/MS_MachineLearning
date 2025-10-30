import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer # (선택적이지만 포함하는 것이 좋음)

# --------------------------------------------------------
# FunctionTransformer에 사용될 순환형 데이터 인코딩 함수 정의
# --------------------------------------------------------
def cyclical_feature_engineer(X, feature_names):
    """
    주어진 열에 대해 sin/cos 인코딩을 수행하고 NumPy 배열 형식으로 반환
    """
    # X는 ColumnTransformer로부터 NumPy 배열로 전달되므로, DataFrame으로 변환 후 처리합니다.
    # DataFrame으로 변환할 때 columns 인자를 반드시 사용해야 오류를 피할 수 있습니다.
    X_df = pd.DataFrame(X, columns=feature_names) 
    
    X_new_list = []
    
    # 각 순환형 특성에 대해 인코딩 수행 (주기: hour=24, month=12, dayofweek=7)
    cycle_map = {'hour': 24, 'month': 12, 'dayofweek': 7}
    
    for feature in feature_names:
        cycle = cycle_map.get(feature)
        if cycle:
            # sin/cos 값 계산 후 리스트에 추가
            X_new_list.append(np.sin(2 * np.pi * X_df[feature] / cycle).values)
            X_new_list.append(np.cos(2 * np.pi * X_df[feature] / cycle).values)
            
    # 새로 생성된 특성들을 수평(axis=1)으로 합쳐 NumPy 배열로 반환
    return np.stack(X_new_list, axis=1)