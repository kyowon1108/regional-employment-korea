import pandas as pd
import numpy as np

def debug_jeju_integration():
    """제주도 데이터 통합 과정 디버깅"""
    print("=== 제주도 데이터 통합 디버깅 ===")

    # 1. 원본 산업별 종사자 데이터에서 제주도 확인
    print("\n1. 원본 산업별 종사자 데이터:")
    industry_df = pd.read_csv("/Users/kapr/Desktop/DataAnalyze/data/processed/산업별_종사자_전처리완료_utf-8.csv")
    jeju_industry = industry_df[industry_df['시도'].str.contains('제주', na=False)]
    print(f"제주 관련 데이터: {len(jeju_industry)}개")
    if len(jeju_industry) > 0:
        print(jeju_industry[['시도', '시군구', '연도']].head())

    # 2. 전처리 과정에서의 변환 시뮬레이션
    print("\n2. 지역명 변환 테스트:")
    from comprehensive_preprocessing import split_region_name

    test_names = ['제주도 서귀포시', '제주도 제주시', '제주 서귀포시', '제주 제주시']
    for name in test_names:
        result = split_region_name(name)
        print(f"'{name}' → {result}")

    # 3. 매핑 딕셔너리 확인
    print("\n3. 매핑 테이블 확인:")
    mapping_df = pd.read_csv("/Users/kapr/Desktop/DataAnalyze/data/processed/시도_시군구_매핑.csv")
    jeju_mapping = mapping_df[mapping_df['시도'].str.contains('제주', na=False)]
    print("제주 매핑:")
    print(jeju_mapping)

    # 4. 최종 통합 데이터에서 제주도 확인
    print("\n4. 최종 통합 데이터:")
    final_df = pd.read_csv("/Users/kapr/Desktop/DataAnalyze/new_analysis/data/new_processed/comprehensive_integrated_data.csv")
    jeju_final = final_df[final_df['시도'].str.contains('제주', na=False)]
    print(f"최종 제주 데이터: {len(jeju_final)}개")
    print(jeju_final[['시도', '시군구', '연도', 'E9_체류자수', '제조업_종사자수', '서비스업_종사자수']].head())

if __name__ == "__main__":
    debug_jeju_integration()