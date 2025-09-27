import pandas as pd
import numpy as np
from comprehensive_preprocessing import *

def debug_merge_process():
    """데이터 병합 과정 세부 디버깅"""
    print("=== 데이터 병합 과정 디버깅 ===")

    # 데이터 로드
    mapping_df, industry_df, e9_df, employment_raw = load_existing_processed_data()
    employment_df = clean_employment_data(employment_raw, mapping_df)

    # 매핑 딕셔너리
    mapping_dict = dict(zip(mapping_df['시군구'], mapping_df['시도']))

    # 1. 산업 데이터 변환
    print("\n1. 산업별 종사자 데이터 변환:")
    industry_annual = convert_to_annual_data(industry_df)
    industry_annual = standardize_region_names(industry_annual, mapping_dict)
    industry_annual = process_special_regions(industry_annual)

    # 제주 데이터 확인
    jeju_industry = industry_annual[industry_annual['시도'].str.contains('제주', na=False)]
    print(f"변환된 산업 데이터 중 제주: {len(jeju_industry)}개")
    if len(jeju_industry) > 0:
        print(jeju_industry[['시도', '시군구', '연도', '제조업_종사자수', '서비스업_종사자수']].head())

    # 2. 고용률 데이터 변환
    print("\n2. 고용률 데이터 변환:")
    employment_annual = convert_to_annual_data(employment_df)
    employment_annual = standardize_region_names(employment_annual, mapping_dict)

    # 제주 데이터 확인
    jeju_employment = employment_annual[employment_annual['시도'].str.contains('제주', na=False)]
    print(f"변환된 고용률 데이터 중 제주: {len(jeju_employment)}개")
    if len(jeju_employment) > 0:
        print(jeju_employment[['시도', '시군구', '연도', '고용률']].head())

    # 3. 병합 키 확인
    print("\n3. 병합 키 비교:")
    if len(jeju_industry) > 0 and len(jeju_employment) > 0:
        industry_keys = set([f"{row['시도']}|{row['시군구']}|{row['연도']}" for _, row in jeju_industry.iterrows()])
        employment_keys = set([f"{row['시도']}|{row['시군구']}|{row['연도']}" for _, row in jeju_employment.iterrows()])

        print(f"산업 데이터 키 수: {len(industry_keys)}")
        print(f"고용률 데이터 키 수: {len(employment_keys)}")
        print(f"공통 키: {len(industry_keys & employment_keys)}개")

        print("\n산업 데이터 키 (첫 5개):")
        for key in sorted(list(industry_keys))[:5]:
            print(f"  {key}")

        print("\n고용률 데이터 키 (첫 5개):")
        for key in sorted(list(employment_keys))[:5]:
            print(f"  {key}")

    # 4. 실제 병합 테스트
    print("\n4. 실제 병합 테스트:")
    test_merge = employment_annual.merge(
        industry_annual[['시도', '시군구', '연도', '제조업_종사자수', '서비스업_종사자수']],
        on=['시도', '시군구', '연도'],
        how='left'
    )

    jeju_merged = test_merge[test_merge['시도'].str.contains('제주', na=False)]
    print(f"병합 후 제주 데이터: {len(jeju_merged)}개")
    if len(jeju_merged) > 0:
        print(jeju_merged[['시도', '시군구', '연도', '고용률', '제조업_종사자수', '서비스업_종사자수']].head())

if __name__ == "__main__":
    debug_merge_process()